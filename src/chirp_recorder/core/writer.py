"""One writer thread owns the sink connection and the daily JSONL copy.

Callers `put` rows and never block: a full queue drops the batch and counts
it.  The thread writes each sigmondb row to <output_dir>/rows/YYYY-MM-DD.jsonl
first (the station's canonical copy), then hands it to the sink; a sink
failure counts and never touches the JSONL.  Three table writers share the
one thread: arrival, capture and spots (pre-flight C12).
"""
from __future__ import annotations

import json
import logging
import queue
import re
import threading
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from hamsci_dsp import sigmondb_rows as sr

log = logging.getLogger(__name__)

PHASE_KEEP_DAYS = 7
QUEUE_MAX = 256
_STOP = object()


def make_sink_writers() -> Optional[tuple]:
    """(arrival, capture, spots) writers on the local sink, or None.

    hs_uploader.sink.Writer, falling back to sigmond.hamsci_sink.  Rows land
    under producer chirp-recorder; spots go to codar.spots (mode "codar")."""
    try:
        try:
            from hs_uploader.sink import Writer  # type: ignore[import-not-found]
        except ImportError:
            from sigmond.hamsci_sink import Writer  # type: ignore[import-not-found]
        arrival, capture = (
            Writer.from_env(table=t, mode="sigmondb", schema_version=sr.SCHEMA_VERSION,
                            producer="chirp-recorder", batch_rows=200)
            for t in ("arrival", "capture"))
        spots = Writer.from_env(table="spots", mode="codar", schema_version=1,
                                producer="chirp-recorder", batch_rows=200)
        return arrival, capture, spots
    except ImportError:
        log.debug("no sink writer importable; sigmondb rows off")
    except Exception as exc:  # noqa: BLE001
        log.warning("sink writer init failed (%s); JSONL unaffected", exc)
    return None


def _json_default(o):
    """numpy scalars in a row; anything else still raises."""
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


class PhaseStore:
    """Per-sweep phase and amplitude series, kept on the station.  One JSON
    line per arrival per interval, one file per sweep group per UTC day, under
    <root>/<radiod>/<group>/; days older than keep_days go at the first write
    of each new day.  Moved from codar-sounder's core/sigmondb.py; the root is
    now <output_dir>/phase.  The duplicate drops when codar-sounder retires
    (pre-flight C29)."""

    def __init__(self, root, *, radiod: str = "radiod", keep_days: int = PHASE_KEEP_DAYS):
        self.root = Path(root)
        self.radiod = radiod or "radiod"
        self.keep_days = int(keep_days)
        self._lock = threading.Lock()
        self._pruned_for: Optional[date] = None

    @staticmethod
    def _safe(group: str) -> str:
        return re.sub(r"[^0-9A-Za-z.+-]+", "_", group)

    def write(self, *, group, cpi_start_utc, radiod="", sweep_start_ns, bin, period_s, tau_ns,
              delay_ns, level_db, slow) -> str:
        day = cpi_start_utc.date()              # tz-aware UTC
        base = self.root / self._safe(radiod or self.radiod)
        path = base / self._safe(group) / f"{day.isoformat()}.jsonl"
        rec = {"cpi_start_utc": cpi_start_utc.isoformat(), "sweep_start_ns": sweep_start_ns,
               "group": group, "bin": int(bin), "period_s": period_s, "tau_ns": int(tau_ns),
               "delay_ns": int(delay_ns), "level_db": round(level_db, 2),
               "phase_rad": [round(float(x), 4) for x in np.angle(slow)],
               "amp_db": [round(float(x), 2) for x in
                          20 * np.log10(np.maximum(np.abs(slow), 1e-12))]}
        with self._lock:
            if self._pruned_for != day:
                self.prune(today=day)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as fh:
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        return str(path.relative_to(self.root))

    def prune(self, *, today: date) -> None:
        self._pruned_for = today
        cutoff = today - timedelta(days=self.keep_days)
        for f in self.root.glob("*/*/*.jsonl"):
            try:
                if date.fromisoformat(f.stem) < cutoff:
                    f.unlink()
            except (ValueError, OSError):
                continue


class RowWriter(threading.Thread):
    def __init__(self, *, output_dir: Path, sink_writers: Optional[tuple],
                 phase_store: Optional[PhaseStore], queue_max: int = QUEUE_MAX):
        super().__init__(name="chirp-row-writer", daemon=True)
        self.output_dir = Path(output_dir)
        self.sink_writers = sink_writers
        self.phase_store = phase_store
        self._q: queue.Queue = queue.Queue(maxsize=queue_max)
        self._stats = {"written": 0, "sink_failures": 0, "queue_full": 0,
                       "jsonl_failures": 0}
        self._lock = threading.Lock()
        self._closed = False
        self._jsonl_lock = threading.Lock()
        self._jsonl_ok = True
        self.start()

    @property
    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def _bump(self, key, n=1):
        with self._lock:
            self._stats[key] += n

    def put(self, capture: Optional[dict], arrivals: list, phases: list, spots=()) -> None:
        """Write the rows' canonical JSONL copy now, then queue the batch for
        the sink.  Never blocks on the sink: a full queue drops only the sink
        copy and counts it."""
        arrivals = list(arrivals)
        self._jsonl(arrivals, capture)
        try:
            self._q.put_nowait((capture, arrivals, list(phases), list(spots)))
        except queue.Full:
            self._bump("queue_full")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._q.put(_STOP)           # the thread keeps draining, so this ends
        self.join()
        for w in self.sink_writers or ():
            try:
                w.close()
            except Exception as exc:  # noqa: BLE001
                self._bump("sink_failures")
                log.warning("sink writer close failed: %s", exc)

    def run(self) -> None:
        while True:
            item = self._q.get()
            if item is _STOP:
                return
            try:
                self._handle(*item)
            except Exception as exc:  # noqa: BLE001
                log.warning("row writer batch failed: %s", exc)

    def _handle(self, capture, arrivals, phases, spots) -> None:
        if self.phase_store is not None:
            for kw in phases:
                try:
                    self.phase_store.write(**kw)
                except Exception as exc:  # noqa: BLE001
                    log.warning("phase store write failed: %s", exc)
        w = self.sink_writers
        if not w:
            return
        for writer, rows in ((w[0], arrivals), (w[1], [capture] if capture else []),
                             (w[2], spots)):
            if not rows:
                continue
            try:
                writer.insert(rows)
            except Exception as exc:  # noqa: BLE001
                self._bump("sink_failures")
                log.warning("sink insert failed (%s); JSONL holds the rows", exc)

    def _jsonl(self, arrivals, capture) -> None:
        """Append each row to the day's file.  A bad row counts in
        jsonl_failures and never stops the others or the sink inserts."""
        by_day: dict = {}
        ok = True
        for table, row in [("arrival", a) for a in arrivals] + (
                [("capture", capture)] if capture else []):
            try:
                line = json.dumps({"table": table, **row}, separators=(",", ":"),
                                  default=_json_default)
                by_day.setdefault(str(row["time"])[:10], []).append(line)
            except Exception as exc:  # noqa: BLE001
                ok = False
                self._bump("jsonl_failures")
                self._jsonl_state(False, exc)
        with self._jsonl_lock:
            for day, lines in by_day.items():
                try:
                    d = self.output_dir / "rows"
                    d.mkdir(parents=True, exist_ok=True)
                    with open(d / f"{day}.jsonl", "a") as fh:
                        fh.write("\n".join(lines) + "\n")
                        fh.flush()
                    self._bump("written", len(lines))
                except Exception as exc:  # noqa: BLE001
                    ok = False
                    self._bump("jsonl_failures", len(lines))
                    self._jsonl_state(False, exc)
        if ok:
            self._jsonl_state(True, None)

    def _jsonl_state(self, ok: bool, exc) -> None:
        """Log on a change of state, not on every failure."""
        if ok != self._jsonl_ok:
            self._jsonl_ok = ok
            if ok:
                log.info("rows JSONL writing again")
        if not ok and not getattr(self, "_jsonl_said", False):
            self._jsonl_said = True
            log.warning("rows JSONL write failed (%s); sink inserts continue", exc)
        elif ok:
            self._jsonl_said = False
