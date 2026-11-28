"""The scout's runner: a coarse channel over the whole span, one fine channel
that dwells window by window, and the rows both produce.

Spectrum only, never IQ.  The coarse stream runs for the whole of ``run``;
its dashes accumulate (dashes, never frames) and every ``coarse_window_s``
(or sooner, once the window holds more than ``coarse_max_held`` dashes) go
through the line search.  The fine stream opens once and retunes for
each dwell.  Both callbacks fold each frame into its view at once, under a
lock, so no frame is buffered.  A row's times come from radiod's
``gps_time`` when every frame of the dwell carries one, else from packet
arrival on the monotonic clock; ``engineering["time_source"]`` says which.

Every stream gets ka9q's default lifetime and ``stop()`` on every exit path,
so a crashed scout leaves no channel behind past ~20 s.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from hamsci_dsp.chirp.streaks import BinBackground, Dash, Streak, find_streaks, frame_dashes

from chirp_recorder.core.writer import ROWS_KEEP_DAYS, _json_default
from chirp_recorder.scout.plan import Dwell, DwellCycle
from chirp_recorder.scout.published import Published, load_published
from chirp_recorder.scout.rows import dwell_row, screen, sweep_row
from chirp_recorder.scout.settings import ScoutSettings

log = logging.getLogger(__name__)

SETTLE_S = 0.5              # frames this soon after a retune may hold the old window
RETRY_S = 5.0               # pause after a dwell that raised
PRODUCER = "chirp-scout"
TABLES = ("sweep", "scout_dwell")


@dataclass(frozen=True)
class ViewCost:
    """What one view may spend, and what it keeps.  A frame with more than
    ``max_dashes`` dashes is a burst; ``rate_max_hz_s`` bounds the line
    search's pairs (the fine view's 2e5 keeps a busy dwell under the unit's
    MemoryMax).  A cap of 60 dropped a quarter of SIGMA's full-window frames
    and with them the CODAR's 1 s period; 100 keeps it (R12).  min_fit 0.6:
    SIGMA's CODAR fits 0.71 (R11).  min_span_hz 10 kHz: narrower streaks
    were SIGMA's drifting carriers (R16)."""
    max_dashes: int
    rate_max_hz_s: float
    min_sweeps: int
    min_fit: float
    min_span_hz: float = 10_000.0


FINE = ViewCost(max_dashes=100, rate_max_hz_s=2.0e5, min_sweeps=5, min_fit=0.6)
COARSE = ViewCost(max_dashes=100, rate_max_hz_s=2.0e6, min_sweeps=2, min_fit=0.6)
COARSE_MAX_HELD = 300_000   # dashes a coarse window may hold before it closes early


def clean_power(power_db) -> Optional[np.ndarray]:
    """power_db with each non-finite bin set to the frame's finite median;
    None when no bin is finite.  A -inf bin would poison its background
    for good."""
    if power_db is None:
        return None
    p = np.asarray(power_db, dtype=float)
    ok = np.isfinite(p)
    if ok.all():
        return p
    if not ok.any():
        return None
    p = p.copy()
    p[~ok] = float(np.median(p[ok]))
    return p


class _Accum:
    """One dwell's (or coarse window's) frames, folded in as they arrive.

    Dashes are stamped with their frame's index; ``result`` restamps them
    with the frame times once the dwell knows its time source.  Frames that
    arrive within SETTLE_S of ``settle_from`` or after ``deadline`` (both on
    the monotonic clock) are discarded.
    """

    def __init__(self, dwell: Dwell, cost: ViewCost, *, settle_from: float,
                 deadline: Optional[float]):
        self.dwell, self.cost = dwell, cost
        self.settle_from, self.deadline = settle_from, deadline
        self.bg: Optional[BinBackground] = None
        self.n_bins = 0
        self.f0_hz = 0.0
        self.dashes: list[Dash] = []
        self.gps: list[Optional[int]] = []
        self.mono: list[float] = []
        self.medians: list[float] = []
        self.counts = {"frames_received": 0, "frames_settle": 0, "frames_late": 0,
                       "frames_skipped": 0, "frames_other_freq": 0, "frames_failed": 0, "bursts": 0}

    def add(self, st, arrival: float) -> None:
        self.counts["frames_received"] += 1
        if arrival - self.settle_from < SETTLE_S:
            self.counts["frames_settle"] += 1
            return
        if self.deadline is not None and arrival > self.deadline:
            self.counts["frames_late"] += 1
            return
        freq = getattr(st, "frequency", None)
        if freq is not None and abs(float(freq) - self.dwell.centre_hz) > self.dwell.bin_hz:
            self.counts["frames_other_freq"] += 1
            return
        p = clean_power(getattr(st.spectrum, "bin_power_db", None))
        if p is None or p.ndim != 1 or p.size == 0:
            self.counts["frames_skipped"] += 1
            return
        if self.bg is None:
            # SPECT2 bins run low to high, the most negative first (radiod
            # encode_byte_data): bin k sits at centre + (k - n/2) * bin_hz.
            self.n_bins = p.size
            self.f0_hz = self.dwell.centre_hz - p.size * self.dwell.bin_hz / 2
            self.bg = BinBackground(p.size)
        elif p.size != self.n_bins:
            self.counts["frames_skipped"] += 1
            return
        k = len(self.mono)
        excess = self.bg.update(p)
        dashes, _, burst = frame_dashes(excess, f0_hz=self.f0_hz, bin_hz=self.dwell.bin_hz,
                                        t_s=float(k), max_dashes=self.cost.max_dashes)
        self.counts["bursts"] += int(burst)
        self.dashes.extend(dashes)
        g = getattr(st, "gps_time", None)
        self.gps.append(int(g) if g else None)
        self.mono.append(arrival)
        self.medians.append(float(np.median(p)))

    @property
    def n_frames(self) -> int:
        return len(self.mono)

    def times(self) -> tuple[np.ndarray, str]:
        """Frame times in seconds from the first frame, and their source."""
        if self.gps and all(g is not None for g in self.gps):
            g = np.array(self.gps, dtype=np.int64)
            return (g - g[0]).astype(float) * 1e-9, "gps_time"
        m = np.array(self.mono, dtype=float)
        return m - m[0], "monotonic"

    def streaks(self, *, guard: bool = False) -> tuple[list[Streak], float, str, str]:
        """The line search over this dwell's dashes: (streaks, frame_s,
        time_source, error).  With ``guard`` a failing search returns no
        streaks and its error, so the dwell still counts as listened."""
        t, source = self.times()
        frame_s = float(np.mean(np.diff(t))) if t.size >= 2 else 0.0
        if t.size < 2 or frame_s <= 0:
            return [], frame_s, source, ""
        dashes = [replace(d, t_s=float(t[int(d.t_s)])) for d in self.dashes]
        try:
            found = find_streaks(dashes, view_lo_hz=self.f0_hz,
                                 view_hi_hz=self.f0_hz + self.n_bins * self.dwell.bin_hz,
                                 bin_hz=self.dwell.bin_hz, frame_s=frame_s,
                                 rate_max_hz_s=self.cost.rate_max_hz_s)
        except Exception as exc:  # noqa: BLE001
            if not guard:
                raise
            log.warning("line search failed on %s %.3f MHz: %r", self.dwell.view,
                        self.dwell.centre_hz / 1e6, exc)
            return [], frame_s, source, f"{type(exc).__name__}: {exc}"
        return found, frame_s, source, ""


def replay(power_db, t_s, *, f0_hz: float, bin_hz: float, view: str = "fine") -> list[Streak]:
    """Run the scout's per-frame steps and line search on a saved waterfall
    (frames x bins, bin 0 at f0_hz), with the given view's caps."""
    p = np.asarray(power_db)
    n_bins = p.shape[1]
    d = Dwell(view, f0_hz + n_bins * bin_hz / 2, n_bins * bin_hz, bin_hz, "replay")
    acc = _Accum(d, FINE if view == "fine" else COARSE, settle_from=-np.inf, deadline=None)
    for row, t in zip(p, np.asarray(t_s, dtype=float)):
        acc.add(_Frame(row), float(t))
    return acc.streaks()[0]


class _Frame:
    """A saved frame dressed as the parts of a ka9q ChannelStatus _Accum reads."""

    def __init__(self, power_db):
        self.spectrum = self
        self.bin_power_db = power_db
        self.gps_time = None


# --- the writer -------------------------------------------------------------

def _sink_writers() -> Optional[dict]:
    """{table: sink Writer} for sweep and scout_dwell, or None.

    hs_uploader.sink.Writer, falling back to sigmond.hamsci_sink, as
    core/writer.py builds them."""
    built: dict = {}
    try:
        try:
            from hs_uploader.sink import Writer  # type: ignore[import-not-found]
        except ImportError:
            from sigmond.hamsci_sink import Writer  # type: ignore[import-not-found]
        for t in TABLES:
            built[t] = Writer.from_env(table=t, mode="sigmondb", schema_version=1,
                                       producer=PRODUCER)
    except ImportError:
        log.warning("no sink writer importable; scout rows go to JSONL only")
        return None
    except Exception as exc:  # noqa: BLE001
        log.warning("scout sink writer init failed (%s); JSONL unaffected", exc)
        for w in built.values():
            try:
                w.close()
            except Exception:  # noqa: BLE001
                pass
        return None
    return built


class ScoutWriter:
    """write(table, row): the row to the sink and to
    <state_dir>/rows/YYYY-MM-DD.jsonl as {"table": ..., "row": ...}.  Files
    dated more than ROWS_KEEP_DAYS before the host's UTC date go at the
    first write of each day.  A sink failure never touches the JSONL."""

    def __init__(self, state_dir: Path, sinks: Optional[dict], *,
                 today: Callable[[], date]):
        self.dir = Path(state_dir) / "rows"
        self.sinks = sinks
        self._today = today
        self._pruned_for: Optional[date] = None
        self._lock = threading.Lock()

    def __call__(self, table: str, row: dict) -> None:
        if table not in TABLES:
            raise ValueError(f"unknown scout table {table!r}")
        with self._lock:
            self._jsonl(table, row)
            w = (self.sinks or {}).get(table)
            if w is None:
                return
            try:
                w.insert([row])
                w.flush()
            except Exception as exc:  # noqa: BLE001
                log.warning("scout sink insert failed (%s); JSONL holds the row", exc)

    def _jsonl(self, table: str, row: dict) -> None:
        try:
            day = self._today()
            if self._pruned_for != day:
                self._prune(day)
            self.dir.mkdir(parents=True, exist_ok=True)
            line = json.dumps({"table": table, "row": row}, separators=(",", ":"),
                              default=_json_default)
            with open(self.dir / f"{str(row['time'])[:10]}.jsonl", "a") as fh:
                fh.write(line + "\n")
        except Exception as exc:  # noqa: BLE001
            log.warning("scout rows JSONL write failed: %s", exc)

    def _prune(self, day: date) -> None:
        self._pruned_for = day
        cutoff = (day - timedelta(days=ROWS_KEEP_DAYS)).isoformat()
        for f in self.dir.glob("????-??-??.jsonl"):
            if f.stem < cutoff:
                try:
                    f.unlink()
                except OSError as exc:
                    log.warning("scout rows prune of %s failed: %s", f.name, exc)

    def close(self) -> None:
        for w in (self.sinks or {}).values():
            try:
                w.close()
            except Exception as exc:  # noqa: BLE001
                log.warning("scout sink close failed: %s", exc)


def make_scout_writer(state_dir, *, today: Optional[Callable[[], date]] = None) -> ScoutWriter:
    """The real writer: the two sink writers plus the daily JSONL."""
    return ScoutWriter(Path(state_dir), _sink_writers(),
                       today=today or (lambda: datetime.now(timezone.utc).date()))


# --- the runner -------------------------------------------------------------

def _default_control(status: str):
    from ka9q import RadiodControl
    return RadiodControl(status, client_id="chirp-recorder")


def _default_stream(control, frequency_hz, bin_count, resolution_bw, *,
                    poll_interval_sec, on_spectrum):
    from ka9q.spectrum_stream import SpectrumStream
    return SpectrumStream(control, frequency_hz, bin_count, resolution_bw,
                          poll_interval_sec=poll_interval_sec, on_spectrum=on_spectrum)


def _add(acc: _Accum, st, arrival: float) -> None:
    """Fold one frame in; a frame that breaks the view is counted, and the
    first per dwell logged, never raised into ka9q's receive thread."""
    try:
        acc.add(st, arrival)
    except Exception as exc:  # noqa: BLE001
        acc.counts["frames_failed"] += 1
        if acc.counts["frames_failed"] == 1:
            log.warning("%s frame failed: %r", acc.dwell.view, exc)


def _stop_stream(s, what: str) -> None:
    try:
        s.stop()
    except Exception as exc:  # noqa: BLE001
        log.warning("%s stream stop failed: %s", what, exc)


class Scout:
    def __init__(self, settings: ScoutSettings, reporter_id: str, *,
                 control_factory: Callable = _default_control,
                 stream_factory: Callable = _default_stream,
                 writer: Callable[[str, dict], None],
                 now: Callable[[], float] = time.time,
                 monotonic: Callable[[], float] = time.monotonic,
                 utcnow: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 published_loader: Optional[Callable[[], Published]] = None,
                 coarse_max_held: int = COARSE_MAX_HELD):
        self.s = settings
        self.reporter_id = reporter_id
        self.coarse_max_held = int(coarse_max_held)
        self._control_factory = control_factory
        self._stream_factory = stream_factory
        self._write = writer
        self._now = now
        self._mono = monotonic
        self._utcnow = utcnow
        self._load = published_loader or (lambda: load_published(
            settings.state_dir, settings.public_hosts, reporter_id, now=now))
        self._stop = threading.Event()
        self._control = None
        self._fine = None
        self._fine_stale = False
        self._fine_acc: Optional[_Accum] = None
        self._fine_lock = threading.Lock()
        self._coarse = None
        self._coarse_acc: Optional[_Accum] = None
        self._coarse_lock = threading.Lock()
        self._coarse_t0 = 0.0
        self._coarse_utc: Optional[datetime] = None
        self.cycle = DwellCycle(settings)
        self.published = Published((), (), None, "none")
        self._published_at: Optional[float] = None

    # -- channels

    def _ctl(self):
        if self._control is None:
            self._control = self._control_factory(self.s.status)
        return self._control

    def _on_fine(self, st) -> None:
        arrival = self._mono()
        with self._fine_lock:
            if self._fine_acc is not None:
                _add(self._fine_acc, st, arrival)

    def _on_coarse(self, st) -> None:
        arrival = self._mono()
        with self._coarse_lock:
            if self._coarse_acc is not None:
                _add(self._coarse_acc, st, arrival)

    def _coarse_dwell(self) -> Dwell:
        return Dwell("coarse", self.s.nyquist_hz / 2, self.s.nyquist_hz,
                     self.s.coarse_bin_hz, "band")

    def _open_coarse(self) -> None:
        self._coarse_t0 = self._mono()
        self._coarse_utc = self._utcnow()
        with self._coarse_lock:
            self._coarse_acc = _Accum(self._coarse_dwell(), COARSE,
                                      settle_from=self._coarse_t0, deadline=None)
        self._coarse = self._stream_factory(
            self._ctl(), self.s.nyquist_hz / 2, int(round(self.s.nyquist_hz / self.s.coarse_bin_hz)),
            self.s.coarse_bin_hz, poll_interval_sec=self.s.coarse_poll_s,
            on_spectrum=self._on_coarse)
        try:
            self._coarse.start()
        except Exception:
            _stop_stream(self._coarse, "coarse")
            self._coarse = None
            raise

    def _tune_fine(self, centre_hz: float) -> None:
        if self._fine is not None and self._fine_stale:
            log.info("fine stream starved last dwell; opening it afresh")
            old, self._fine = self._fine, None
            _stop_stream(old, "fine")
        self._fine_stale = False
        if self._fine is None:
            self._fine = self._stream_factory(
                self._ctl(), centre_hz, int(round(self.s.fine_span_hz / self.s.fine_bin_hz)),
                self.s.fine_bin_hz, poll_interval_sec=self.s.fine_poll_s,
                on_spectrum=self._on_fine)
            try:
                self._fine.start()
            except Exception:
                old, self._fine = self._fine, None
                _stop_stream(old, "fine")
                raise
        else:
            self._fine.set_frequency(centre_hz)

    def close(self) -> None:
        """Stop both streams and drop the control; safe to call twice."""
        with self._fine_lock:
            self._fine_acc = None
        for attr, what in (("_fine", "fine"), ("_coarse", "coarse")):
            s = getattr(self, attr)
            setattr(self, attr, None)
            if s is not None:
                _stop_stream(s, what)
        ctl, self._control = self._control, None
        if ctl is not None and hasattr(ctl, "close"):
            try:
                ctl.close()
            except Exception as exc:  # noqa: BLE001
                log.warning("radiod control close failed: %s", exc)

    # -- rows

    def _rows(self, acc: _Accum, *, start_utc: datetime, duration_s: float,
              cost: ViewCost) -> tuple[list[dict], Optional[dict], list]:
        """(sweep rows, dwell row, screened streaks); ([], None, []) for a
        dwell with fewer than min_frames frames."""
        d = acc.dwell
        if acc.n_frames < self.s.min_frames:
            log.warning("dwell starved: %s %.3f MHz got %d frames, min_frames %d (%s)",
                        d.view, d.centre_hz / 1e6, acc.n_frames, self.s.min_frames,
                        ", ".join(f"{k} {v}" for k, v in acc.counts.items()))
            return [], None, []
        found, frame_s, source, error = acc.streaks(guard=True)
        passed, dropped = screen(found, self.published, nyquist_hz=self.s.nyquist_hz,
                                 min_sweeps=cost.min_sweeps, min_fit=cost.min_fit,
                                 min_span_hz=cost.min_span_hz)
        sweeps = []
        for s, key, eng in passed:
            sweeps.append(sweep_row(s, dwell=d, start_utc=start_utc,
                                    reporter_id=self.reporter_id, frame_s=frame_s,
                                    matched_key=key,
                                    engineering={**eng, "time_source": source}))
        age = (None if self.published.fetched_at is None
               else round(self._now() - self.published.fetched_at))
        eng = {"time_source": source, "reason": d.reason, "streaks_found": len(found),
               **acc.counts, **{f"dropped_{k}": v for k, v in dropped.items()},
               "published_source": self.published.source,
               "published_age_s": "" if age is None else age}
        if error:
            eng["search_error"] = error
        row = dwell_row(dwell=d, start_utc=start_utc, reporter_id=self.reporter_id,
                        frame_s=frame_s, duration_s=duration_s, n_frames=acc.n_frames,
                        noise_db=float(np.median(acc.medians)), n_streaks=len(passed),
                        engineering=eng)
        return sweeps, row, passed

    def run_dwell(self, dwell: Dwell) -> tuple[list[dict], Optional[dict]]:
        """Listen on the fine channel for dwell_s; return (sweep rows, dwell
        row or None).  A starved dwell returns ([], None) and the next dwell
        opens the channel afresh."""
        start_utc = self._utcnow()
        t0 = self._mono()
        acc = _Accum(dwell, FINE, settle_from=t0, deadline=t0 + self.s.dwell_s)
        with self._fine_lock:
            self._fine_acc = acc
        try:
            self._tune_fine(dwell.centre_hz)
            while not self._stop.is_set():
                left = acc.deadline - self._mono()
                if left <= 0:
                    break
                self._stop.wait(min(left, self.s.fine_poll_s))
        finally:
            with self._fine_lock:
                self._fine_acc = None
        if self._stop.is_set():
            return [], None
        sweeps, row, _ = self._rows(acc, start_utc=start_utc, duration_s=self.s.dwell_s,
                                    cost=FINE)
        if row is None:
            self._fine_stale = True
        return sweeps, row

    def _close_coarse_window(self) -> None:
        """Run the window's dashes through the line search, write its rows,
        flag the fine cycle, and start the next window."""
        t1 = self._mono()
        with self._coarse_lock:
            acc = self._coarse_acc
            self._coarse_acc = _Accum(self._coarse_dwell(), COARSE,
                                      settle_from=-np.inf, deadline=None)
        start_utc, self._coarse_utc = self._coarse_utc, self._utcnow()
        t0, self._coarse_t0 = self._coarse_t0, t1
        if acc is None:
            return
        sweeps, row, passed = self._rows(acc, start_utc=start_utc, duration_s=t1 - t0,
                                         cost=COARSE)
        if row is None:
            if self._coarse is not None and acc.counts["frames_received"] == 0:
                log.warning("coarse stream silent for a window; opening it afresh")
                old, self._coarse = self._coarse, None
                _stop_stream(old, "coarse")
                self._open_coarse()
            return
        for r in sweeps:
            self._write("sweep", r)
        self._write("scout_dwell", row)
        half = self.s.fine_span_hz / 2
        targets = [t * 1e6 for t in self.s.targets_mhz]
        for s, key, eng in passed:
            # The fixed windows already cover their bands, where the coarse
            # view mis-measures CODAR; a wideband sweeper stays the coarse view's.
            if key or "local_reject" in eng or s.span_hz >= self.s.fine_span_hz:
                continue
            if any(abs(s.centre_hz - t) <= half for t in targets):
                continue
            log.info("coarse streak at %.3f MHz, %.0f Hz/s: flagged for a fine dwell",
                     s.centre_hz / 1e6, s.rate_hz_per_s)
            self.cycle.flag(s.centre_hz, self._mono())

    def _coarse_held(self) -> int:
        with self._coarse_lock:
            return 0 if self._coarse_acc is None else len(self._coarse_acc.dashes)

    def _refresh_published(self) -> None:
        now = self._mono()
        if self._published_at is not None and now - self._published_at < self.s.refresh_s:
            return
        self._published_at = now
        try:
            self.published = self._load()
        except Exception as exc:  # noqa: BLE001
            log.warning("published chirps not loaded (%s); keeping %d", exc,
                        len(self.published.chirps))

    def run(self, stop: threading.Event, max_dwells: Optional[int] = None) -> None:
        """Scout until ``stop`` is set (or after ``max_dwells`` fine dwells)."""
        self._stop = stop
        dwells = 0
        try:
            self._refresh_published()
            self._open_coarse()
            while not stop.is_set():
                self._refresh_published()
                dwell = self.cycle.next(self.published, self._mono())
                try:
                    sweeps, row = self.run_dwell(dwell)
                except Exception as exc:  # noqa: BLE001
                    log.warning("dwell at %.3f MHz failed: %s", dwell.centre_hz / 1e6, exc)
                    self._fine_stale = True
                    stop.wait(RETRY_S)
                    sweeps, row = [], None
                if row is not None:
                    for r in sweeps:
                        self._write("sweep", r)
                    self._write("scout_dwell", row)
                dwells += 1
                if not stop.is_set():
                    held = self._coarse_held()
                    if held > self.coarse_max_held:
                        log.warning("coarse window closed early at %d dashes", held)
                        self._close_coarse_window()
                    elif self._mono() - self._coarse_t0 >= self.s.coarse_window_s:
                        self._close_coarse_window()
                if max_dwells is not None and dwells >= max_dwells:
                    break
        finally:
            self.close()
