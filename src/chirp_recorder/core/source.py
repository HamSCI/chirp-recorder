"""The raw IQ channel and its counter.

RawChannelSource opens a radiod channel through ka9q-python and cuts its
stream into fixed-length intervals.  Each Interval carries its own timing
position: the RX888 input-sample index of its first sample, found through a
CounterMap fitted from the channel's own status polls.  Nothing counts
delivered samples, so a lost packet shifts no later label.

Lessons of codar-sounder's 11-18 fixes (58d9e08, 344457b, 7a06713):

* Intervals assemble in the stream callback, which runs continuously.  A
  consumer that drains slowly loses whole intervals, never part of one.
* Status polls run on their own thread, continuously, and each poll belongs
  to the stream generation live when it was taken.  A generation (bumped at
  every gap and restart) gets its own CounterMap from its own polls.  An
  interval waits for its generation's map; when the generation is superseded
  with no map, the interval is discarded and counted.  No interval ever takes
  another generation's map, epoch or status pair.
* Each interval carries its own position (here: first_rtp mapped to the
  input index), never "anchor + count * length".
* Any gap discards the interval it falls in.  After any RTP discontinuity the
  map is fitted again; a map from another radiod run starts a new epoch, and
  no interval projects across epochs.

RTP is 32 bits and wraps.  Gap tests and first_rtp arithmetic work modulo 2**32;
`Interval.first_rtp` stays the wrapped counter value.  Intervals sit on a grid
of `n` samples from the first sample of a run of contiguous RTP.  A gap leaves
the grid where it is and skips to the next boundary; a restart (a jump too
large or too backward to be loss) begins a new grid at the packet it meets.
"""
from __future__ import annotations

import collections
import dataclasses
import logging
import queue
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable, Iterator, Optional

import numpy as np
from hamsci_dsp.counter_map import CounterMap, CounterPoll

from chirp_recorder.core.channels import USABLE_FRACTION, ChannelPlan

log = logging.getLogger(__name__)

_MASK = 0xFFFFFFFF
_MOD = 2**32
CLIENT_ID = "chirp-recorder"
F32LE = 4


def _s32(x: int) -> int:
    x &= _MASK
    return x - _MOD if x >= 2**31 else x


@dataclass(frozen=True)
class StatusPair:
    """The RTP/GPS pair from one status poll (pre-flight C6)."""
    gps_time_ns: int
    rtp_timesnap: int
    output_samprate: int


@dataclass(frozen=True)
class Interval:
    samples: np.ndarray
    first_rtp: int
    first_input_index: float
    input_samprate: int
    input_epoch_gps_s: float
    channel: str
    status_pair: StatusPair


def _make_control(status: str):
    from ka9q.control import RadiodControl
    return RadiodControl(status, client_id=CLIENT_ID)


def _poll_from_reply(reply, output_samprate: int) -> Optional[CounterPoll]:
    """One status reply to one poll, or None when it lacks a field (radiod
    answers a ghost SSRC with frontend status alone)."""
    fe = getattr(reply, "frontend", None)
    gps = getattr(reply, "gps_time", None)
    rtp = getattr(reply, "rtp_timesnap", None)
    n_in = getattr(fe, "input_samples", None)
    f_in = getattr(fe, "input_samprate", None)
    ilen = getattr(reply, "filter_blocksize", None)
    if not gps or rtp is None or n_in is None or not f_in or not ilen:
        return None
    if getattr(reply, "output_samprate", None) != output_samprate:
        return None
    return CounterPoll(rtp_timesnap=int(rtp), input_samples=int(n_in),
                       filter_blocksize=int(ilen), input_samprate=int(f_in),
                       output_samprate=int(output_samprate), gps_time_ns=int(gps))




_NO_SAMPLES = {"empty_payload", "stream_start"}      # gap reports that insert no zeros


class _KaStream:
    """Adapter from ka9q-python's RadiodStream to (on_packet, on_lost).

    ka9q-python zero-fills lost packets, and radiod drops a block as a packet
    of exact zeros.  The gap positions ka9q-python reports are not trustworthy
    (3.27.0 counts lost-packet output twice in `cumulative_samples`, so every
    reported position after the first loss lands late).  So the adapter finds
    the filled spans itself: runs of exact complex zeros at least `min_zero_run`
    long (F32 IQ from a receiver never holds one).  Each run goes to on_lost on
    its own; the clean samples either side go to on_packet with their true RTP.
    A clean piece therefore never holds a packet-length run of zeros.

    Where the batch's own report and the batch disagree (a reported gap with no
    zero run, or a different count of runs), the whole batch goes to on_lost.
    A zero run the stream never reported is a gap all the same.  Reports that
    insert no samples (EMPTY_PAYLOAD) condemn nothing.  A batch without
    `delivered_rtp_start` is never given an RTP from a running count (that
    disables the continuity check, the bug codar-sounder 344457b fixed): it is
    logged as an ERROR and passed to on_lost when its place is known.
    """

    def __init__(self, channel_info, on_packet, on_lost, _stream_cls=None,
                 min_zero_run: int = 64):
        if _stream_cls is None:
            from ka9q.stream import RadiodStream as _stream_cls
        self._on_packet, self._on_lost = on_packet, on_lost
        self._min_zero_run = int(min_zero_run)
        self._next: Optional[int] = None
        self._stream = _stream_cls(channel=channel_info, on_samples=self._on_samples)

    def _zero_runs(self, samples, min_run: int) -> list[tuple[int, int]]:
        arr = np.asarray(samples)
        if arr.size == 0:
            return []
        z = (arr.real == 0) & (arr.imag == 0) if np.iscomplexobj(arr) else (arr == 0)
        edges = np.diff(np.concatenate(([0], z.view(np.int8), [0])))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        return [(int(a), int(b - a)) for a, b in zip(starts, ends)
                if b - a >= min_run]

    def _on_samples(self, samples, quality) -> None:
        n = len(samples)
        rtp0 = getattr(quality, "delivered_rtp_start", None)
        if rtp0 is None:
            log.error("ka9q-python batch of %d samples has no delivered_rtp_start; "
                      "it cannot be placed on the RTP timeline and is discarded "
                      "(ka9q-python too old, or a batch the stream cannot place)", n)
            if self._next is not None:
                self._on_lost(self._next, n, [n])
                self._next = (self._next + n) & _MASK
            return
        rtp0 = int(rtp0) & _MASK
        self._next = (rtp0 + n) & _MASK
        reported = [g for g in (getattr(quality, "batch_gaps", None) or [])
                    if getattr(g.source, "value", g.source) not in _NO_SAMPLES]
        durs = [int(g.duration_samples) for g in reported]
        # a report may name a fill shorter than min_zero_run (the last fragment
        # of a packet): search for runs as short as the shortest report
        runs = self._zero_runs(samples, min([self._min_zero_run] + durs))
        if reported:
            fits = (len(runs) == len(reported)
                    and all(r >= d for r, d in zip(sorted(r for _, r in runs),
                                                   sorted(durs))))
            if not fits:
                log.error("batch of %d samples reports gaps of %s samples but holds "
                          "zero runs of %s; reports and runs do not match one to "
                          "one, the whole batch is discarded", n, sorted(durs),
                          sorted(r for _, r in runs))
                self._on_lost(rtp0, n, durs)
                return
        if runs and not reported:
            log.error("batch of %d samples holds %d unreported run(s) of exact "
                      "zeros; they are treated as gaps", n, len(runs))
        if not runs:
            self._on_packet(rtp0, samples)
            return
        pos = 0
        for off, ln in runs:
            if off > pos:
                self._on_packet((rtp0 + pos) & _MASK, samples[pos:off])
            self._on_lost((rtp0 + off) & _MASK, ln, [ln])
            pos = off + ln
        if pos < n:
            self._on_packet((rtp0 + pos) & _MASK, samples[pos:])

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()


class RawChannelSource:
    QUEUE_MAX = 3                 # finished intervals waiting for the consumer
    MAX_GAP_S = 30.0              # a forward jump beyond this is a restart, not loss
    PLAUSIBLE_S = 30.0            # a poll this far from the stream's RTP is not its run
    POLLS = 5                     # status pairs per fit
    FIT_ATTEMPTS = 5
    RUN_POLLS_KEPT = 512
    LIFETIME_FRAMES = 3000        # radiod frames (~50 Hz): ~60 s; the poller refreshes it
    BURST_SPACING_S = 0.2         # poll spacing while a generation has no map

    def __init__(self, plan: ChannelPlan, *, interval_s: float, control=None,
                 stream_factory: Optional[Callable] = None,
                 queue_max: Optional[int] = None,
                 poll_period_s: Optional[float] = 3.0, poll_spacing_s: float = 0.05,
                 roll_s: float = 3600.0, fit_wait_s: float = 15.0,
                 silence_s: float = 10.0, lifetime_frames: Optional[int] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.plan = plan
        self.rate = int(plan.sample_rate_hz)
        self.n = int(round(interval_s * self.rate))
        if self.n <= 0:
            raise ValueError("interval_s must hold at least one sample")
        self._control = control
        self._owns_control = control is None
        self._stream_factory = stream_factory or _KaStream
        self._poll_period_s = poll_period_s       # None: no thread; poll_once() is called
        self._poll_spacing_s = poll_spacing_s
        self._roll_s = roll_s
        self._fit_wait_s = fit_wait_s
        self._silence_s = silence_s
        self._lifetime = self.LIFETIME_FRAMES if lifetime_frames is None else lifetime_frames
        self._clock = clock
        self._queue: "queue.Queue" = queue.Queue(maxsize=queue_max or self.QUEUE_MAX)
        self._stopped = threading.Event()
        self._lock = threading.RLock()
        self._stats = {"gaps": 0, "restarts": 0, "discarded": 0, "epochs": 0,
                       "delivered": 0, "queue_full": 0, "rolls": 0,
                       "rejected_polls": 0}
        # assembly state, owned by the stream callback
        self._gen = 0
        self._expected_rtp: Optional[int] = None
        self._last_packet_t = clock()
        self._reset_grid(None)
        # polling state (lock): per-generation pending polls, fitted maps, runs
        self._pending: dict[int, list[CounterPoll]] = {}
        self._gen_info: dict[int, tuple[CounterMap, int]] = {}
        self._run_polls: dict[int, "collections.deque[CounterPoll]"] = {}
        self._last_fit: Optional[tuple[CounterMap, int]] = None
        self._runs = 0
        self._refused: set[int] = set()
        self._ssrc: Optional[int] = None
        self._ensure_kwargs: dict = {}
        self._need_ensure = False
        self._last_ensure_t = clock()
        self._started = False
        self._stream = None
        self._poller: Optional[threading.Thread] = None
        self._released = False
        self._fatal: Optional[Exception] = None

    @property
    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def _bump(self, key: str, by: int = 1) -> None:
        with self._lock:
            self._stats[key] += by

    def stop(self) -> None:
        """Stop, and give the channel back: remove it from radiod, close control."""
        self._stopped.set()
        self._release()

    def _release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        poller = self._poller
        if poller is not None and poller is not threading.current_thread():
            poller.join(timeout=5.0)
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception as exc:  # noqa: BLE001
                log.warning("stream stop failed: %s", exc)
        ctl = self._control
        if ctl is not None and self._ssrc is not None:
            try:
                ctl.remove_channel(self._ssrc)
            except Exception as exc:  # noqa: BLE001
                log.warning("remove_channel of %s failed (its lifetime will "
                            "expire it): %s", self.plan.name, exc)
        if ctl is not None and hasattr(ctl, "close"):
            try:
                ctl.close()
            except Exception as exc:  # noqa: BLE001
                log.warning("control close failed: %s", exc)

    # --- assembly (stream callback thread) ---------------------------------

    def _reset_grid(self, r0: Optional[int]) -> None:
        self._r0 = r0                 # RTP of grid position 0
        self._pos = 0                 # grid position of the next expected sample
        self._buf: Optional[np.ndarray] = None
        self._filled = 0
        self._first_rtp = 0
        self._drop_through = 0        # drop samples below this position (a gap's tail)
        self._counted_k = -1          # highest interval index counted as discarded

    def _align(self, rtp: int) -> None:
        """Compare an arriving RTP with the grid; a gap or a restart ends here."""
        if self._r0 is None:
            self._reset_grid(rtp)
            return
        delta = _s32(rtp - ((self._r0 + self._pos) & _MASK))
        if delta:
            if 0 < delta <= self.MAX_GAP_S * self.rate:
                self._lose(delta, rtp)
            else:
                self._restart(rtp, delta)

    def _on_packet(self, rtp: int, samples) -> None:
        if self._stopped.is_set():
            return
        arr = np.asarray(samples, dtype=np.complex64)
        if arr.size == 0:
            return
        if not np.all(np.isfinite(arr)):
            arr = np.where(np.isfinite(arr), arr, np.complex64(0))
        self._last_packet_t = self._clock()
        self._align(rtp & _MASK)
        self._consume(arr)

    def _on_lost(self, rtp: int, n: int, durations=()) -> None:
        """The stream says [rtp, rtp+n) holds no signal (zero-filled)."""
        if self._stopped.is_set():
            return
        self._last_packet_t = self._clock()
        self._align(rtp & _MASK)
        durations = list(durations) or [n]
        for d in durations:
            log.error("gap on %s: %d samples lost at rtp %d (stream zero-filled); "
                      "the interval it falls in is discarded", self.plan.name, d, rtp)
        self._bump("gaps", len(durations))
        self._discard_span(n)
        self._bump_gen()

    def _bump_gen(self) -> None:
        self._gen += 1

    def _discard_span(self, lost: int) -> None:
        """Advance the grid over `lost` missing samples; discard each interval
        they touch and skip the tail of the last."""
        before = self._pos
        after = before + lost
        k_lo, k_hi = before // self.n, (after - 1) // self.n
        first = max(k_lo, self._counted_k + 1)
        if k_hi >= first:
            self._bump("discarded", k_hi - first + 1)
        self._counted_k = max(self._counted_k, k_hi)
        self._buf, self._filled = None, 0
        self._pos = after
        self._drop_through = (k_hi + 1) * self.n
        self._expected_rtp = (self._r0 + self._pos) & _MASK

    def _lose(self, delta: int, rtp: int) -> None:
        log.error("gap on %s: %d samples lost before rtp %d; the interval it "
                  "falls in is discarded", self.plan.name, delta, rtp)
        self._bump("gaps")
        self._discard_span(delta)
        self._bump_gen()

    def _restart(self, rtp: int, delta: int) -> None:
        expected = (self._r0 + self._pos) & _MASK
        log.error("RTP discontinuity on %s: packet at rtp %d, expected %d "
                  "(%+d samples; a radiod restart re-bases RTP); partial "
                  "interval discarded, new grid, channel re-ensured", self.plan.name,
                  rtp, expected, delta)
        self._bump("restarts")
        if self._filled > 0 and self._pos // self.n > self._counted_k:
            self._bump("discarded")
        self._bump_gen()
        self._reset_grid(rtp)
        self._need_ensure = True        # radiod does not keep run-time channels

    def _consume(self, arr: np.ndarray) -> None:
        n, idx, total = self.n, 0, arr.size
        base = self._pos
        while idx < total:
            pos = base + idx
            if pos < self._drop_through:
                idx += min(total - idx, self._drop_through - pos)
                continue
            off = pos % n
            take = min(total - idx, n - off)
            if off == 0:
                self._buf = np.empty(n, dtype=np.complex64)
                self._filled = 0
                self._first_rtp = (self._r0 + pos) & _MASK
            self._buf[off:off + take] = arr[idx:idx + take]
            self._filled += take
            idx += take
            if self._filled == n:
                self._emit(pos // n)
        self._pos = base + total
        self._expected_rtp = (self._r0 + self._pos) & _MASK

    def _emit(self, k: int) -> None:
        try:
            self._queue.put_nowait((self._buf, self._first_rtp, self._gen))
        except queue.Full:
            self._bump("discarded")
            self._bump("queue_full")
            self._counted_k = max(self._counted_k, k)
            log.warning("interval on %s dropped whole: the consumer is %d intervals "
                        "behind", self.plan.name, self._queue.maxsize)
        self._buf, self._filled = None, 0

    # --- polling (its own thread; poll_once is callable directly) ----------

    def _ensure(self) -> None:
        info = self._control.ensure_channel(**self._ensure_kwargs)
        self._last_ensure_t = self._clock()
        if info.ssrc != self._ssrc:
            log.error("ensure_channel for %s returned ssrc %s, not %s: the running "
                      "stream follows the old one; the source stops", self.plan.name,
                      info.ssrc, self._ssrc)
            self._fatal = RuntimeError(f"channel {self.plan.name} changed ssrc "
                                       f"{self._ssrc} -> {info.ssrc} on re-ensure")
            self._stopped.set()
            raise self._fatal
        return info

    def _plausible(self, rtp_timesnap: int) -> bool:
        exp = self._expected_rtp
        return exp is None or abs(_s32(rtp_timesnap - exp)) <= self.PLAUSIBLE_S * self.rate

    def poll_once(self) -> None:
        """One status poll, filed under the generation live while it ran.

        Refused when a discontinuity fell during it, when its RTP is far from
        the stream's (another radiod run), or when its generation already has a
        map that it contradicts.  First re-ensures the channel after a restart
        or a silence."""
        if self._stopped.is_set():
            return
        try:
            silent = (self._started
                      and self._clock() - self._last_packet_t > self._silence_s
                      and self._clock() - self._last_ensure_t > self._silence_s)
            if self._need_ensure or silent:
                self._need_ensure = False
                if silent:
                    log.error("no packets from %s for %.0f s; re-ensuring the channel",
                              self.plan.name, self._clock() - self._last_packet_t)
                self._ensure()
            g0 = self._gen
            reply = self._control.poll_status(self._ssrc, 2.0)
        except Exception as exc:  # noqa: BLE001
            log.warning("status poll of %s failed: %s", self.plan.name, exc)
            return
        p = _poll_from_reply(reply, self.rate)
        if p is None:
            return
        if self._gen != g0 or not self._plausible(p.rtp_timesnap):
            self._bump("rejected_polls")
            return
        self._attach(g0, p)
        if self._lifetime and self._started:
            try:
                self._control.set_channel_lifetime(self._ssrc, int(self._lifetime))
            except Exception as exc:  # noqa: BLE001
                log.warning("lifetime refresh of %s failed: %s", self.plan.name, exc)

    def _attach(self, g: int, p: CounterPoll) -> None:
        with self._lock:
            info = self._gen_info.get(g)
            if info is not None:
                m, run = info
                if abs(m.input_index(p.rtp_timesnap) - p.input_samples) > 0.5:
                    self._stats["rejected_polls"] += 1
                    log.warning("status poll of %s contradicts generation %d's map; "
                                "refused", self.plan.name, g)
                    return
                self._run_polls[run].append(p)
                return
            pend = self._pending.setdefault(g, [])
            pend.append(p)
            del pend[:-16]
            if len(pend) < self.POLLS:
                return
            m = CounterMap.from_polls(pend, min_agree=3)
            if m is None:
                return
            prev = self._last_fit
            if prev is not None and m.same_run(prev[0]):
                run = prev[1]
                m = dataclasses.replace(m, input_epoch_gps_s=prev[0].input_epoch_gps_s)
            else:
                run = self._runs
                self._runs += 1
                self._stats["epochs"] += 1
                self._run_polls[run] = collections.deque(maxlen=self.RUN_POLLS_KEPT)
                if prev is not None:
                    log.error("new radiod run on %s (epoch %.3f s -> %.3f s): new "
                              "epoch, nothing projects across it", self.plan.name,
                              prev[0].input_epoch_gps_s, m.input_epoch_gps_s)
            self._gen_info[g] = (m, run)
            self._last_fit = (m, run)
            self._run_polls[run].extend(
                q for q in pend if abs(m.input_index(q.rtp_timesnap) - q.input_samples) <= 0.5)
            for old in [k for k in self._pending if k <= g]:
                del self._pending[old]

    def _poll_loop(self) -> None:
        while not self._stopped.is_set():
            self.poll_once()
            with self._lock:
                fitted = self._gen in self._gen_info
            self._stopped.wait(self._poll_period_s if fitted else self.BURST_SPACING_S)

    # --- labelling (consumer thread) ---------------------------------------

    def _await_map(self, g: int) -> Optional[tuple[CounterMap, int]]:
        deadline = self._clock() + self._fit_wait_s
        while True:
            with self._lock:
                info = self._gen_info.get(g)
                if info is not None:
                    return info
                if g in self._refused:
                    return None
                if self._gen > g:
                    log.error("generation %d of %s was superseded before a counter "
                              "map existed; its interval is discarded, never "
                              "stamped with another run's map", g, self.plan.name)
                    return None
                if self._stopped.is_set():
                    return None
                if self._clock() >= deadline:
                    self._refused.add(g)
                    log.error("no counter map for generation %d of %s after %.0f s; "
                              "its intervals are discarded", g, self.plan.name,
                              self._fit_wait_s)
                    return None
            time.sleep(0.02)

    def _pair_for(self, run: int, first_rtp: int) -> Optional[StatusPair]:
        with self._lock:
            before = [p for p in self._run_polls.get(run, ())
                      if _s32(first_rtp - p.rtp_timesnap) >= 0]
        if not before:
            return None
        p = min(before, key=lambda q: _s32(first_rtp - q.rtp_timesnap))
        return StatusPair(p.gps_time_ns, p.rtp_timesnap, p.output_samprate)

    def _label(self, item) -> Optional[Interval]:
        buf, first_rtp, g = item
        info = self._await_map(g)
        if info is None:
            self._bump("discarded")
            return None
        m, run = info
        if _s32(first_rtp - m.rtp_ref) >= self._roll_s * self.rate:
            m = m.rolled_forward(first_rtp)
            with self._lock:
                self._gen_info[g] = (m, run)
                self._stats["rolls"] += 1
        pair = self._pair_for(run, first_rtp)
        if pair is None:
            log.error("no status pair at or before rtp %d on %s; interval discarded "
                      "(a pair after it would carry the wrong age)", first_rtp,
                      self.plan.name)
            self._bump("discarded")
            return None
        return Interval(samples=buf, first_rtp=first_rtp,
                        first_input_index=m.input_index(first_rtp),
                        input_samprate=m.input_samprate,
                        input_epoch_gps_s=m.input_epoch_gps_s,
                        channel=self.plan.name, status_pair=pair)

    # --- the iterator -------------------------------------------------------

    def intervals(self) -> Iterator[Interval]:
        if self._control is None:
            self._control = _make_control(self.plan.status)
        half = USABLE_FRACTION * self.rate / 2
        self._ensure_kwargs = dict(
            frequency_hz=float(self.plan.centre_hz), preset="iq",
            sample_rate=self.rate, encoding=F32LE,
            agc_enable=0, gain=0.0, low_edge=-half, high_edge=half)
        if self._lifetime:
            self._ensure_kwargs["lifetime"] = int(self._lifetime)
        try:
            info = self._control.ensure_channel(**self._ensure_kwargs)
            self._ssrc = info.ssrc          # what ensure_channel returns, never computed
            for i in range(self.FIT_ATTEMPTS * self.POLLS):
                if 0 in self._gen_info:
                    break
                if i and self._poll_spacing_s:
                    time.sleep(self._poll_spacing_s)
                self.poll_once()
            if 0 not in self._gen_info:
                raise RuntimeError(f"no counter map for {self.plan.name}: status "
                                   "polls never agreed")
            self._last_packet_t = self._clock()
            self._stream = self._stream_factory(info, self._on_packet, self._on_lost)
            self._started = True
            if self._poll_period_s is not None:
                self._poller = threading.Thread(target=self._poll_loop, daemon=True,
                                                name=f"chirp-poll-{self.plan.name}")
                self._poller.start()
            self._stream.start()
            while not self._stopped.is_set():
                try:
                    item = self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                iv = self._label(item)
                if iv is None:
                    continue
                self._bump("delivered")
                yield iv
            if self._fatal is not None:
                raise self._fatal
        finally:
            self._stopped.set()
            self._release()
