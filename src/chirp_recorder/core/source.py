"""The raw IQ channel and its counter.

RawChannelSource opens a radiod channel through ka9q-python and cuts its
stream into fixed-length intervals.  Each Interval carries its own timing
position: the RX888 input-sample index of its first sample, found through a
CounterMap fitted from the channel's own status polls.  Nothing counts
delivered samples, so a lost packet shifts no later label.

Lessons of codar-sounder's 11-18 fixes (58d9e08, 344457b, 7a06713):

* Intervals assemble in the stream callback, which runs continuously.  A
  consumer that drains slowly loses whole intervals, never part of one.
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


class _KaStream:
    """Adapter from ka9q-python's RadiodStream to (on_packet, on_lost).

    ka9q-python zero-fills lost packets.  A batch that holds any filled gap
    goes to on_lost whole: the zeros are not signal, and the batch's exact
    split is not worth trusting.  That costs the interval, which a gap costs
    anyway.
    """

    def __init__(self, channel_info, on_packet, on_lost):
        from ka9q.stream import RadiodStream
        self._on_packet, self._on_lost = on_packet, on_lost
        self._next: Optional[int] = None
        self._stream = RadiodStream(channel=channel_info, on_samples=self._on_samples)

    def _on_samples(self, samples, quality) -> None:
        n = len(samples)
        rtp0 = getattr(quality, "delivered_rtp_start", None)
        if rtp0 is None:
            rtp0 = self._next if self._next is not None else getattr(
                quality, "first_rtp_timestamp", None)
        if rtp0 is None:
            return
        rtp0 = int(rtp0) & _MASK
        self._next = (rtp0 + n) & _MASK
        gaps = list(getattr(quality, "batch_gaps", None) or [])
        if gaps:
            self._on_lost(rtp0, n, [int(g.duration_samples) for g in gaps])
        else:
            self._on_packet(rtp0, samples)

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()


class RawChannelSource:
    QUEUE_MAX = 3                 # finished intervals waiting for the consumer
    MAX_GAP_S = 30.0              # a forward jump beyond this is a restart, not loss
    POLLS = 5                     # status pairs per fit
    FIT_ATTEMPTS = 5
    PAIRS_KEPT = 8

    def __init__(self, plan: ChannelPlan, *, interval_s: float, control=None,
                 stream_factory: Optional[Callable] = None,
                 queue_max: Optional[int] = None, poll_spacing_s: float = 0.05,
                 roll_s: float = 3600.0):
        self.plan = plan
        self.rate = int(plan.sample_rate_hz)
        self.n = int(round(interval_s * self.rate))
        if self.n <= 0:
            raise ValueError("interval_s must hold at least one sample")
        self._control = control
        self._stream_factory = stream_factory or _KaStream
        self._poll_spacing_s = poll_spacing_s
        self._roll_s = roll_s
        self._queue: "queue.Queue" = queue.Queue(maxsize=queue_max or self.QUEUE_MAX)
        self._stopped = threading.Event()
        self._lock = threading.Lock()
        self._stats = {"gaps": 0, "discarded": 0, "epochs": 0, "delivered": 0,
                       "queue_full": 0, "rolls": 0, "withheld": 0}
        # assembly state, owned by the stream callback
        self._gen = 0
        self._reset_grid(None)
        # labelling state, owned by the consumer
        self._map: Optional[CounterMap] = None
        self._pairs: list[StatusPair] = []
        self._cur_gen = 0

    @property
    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def _bump(self, key: str, by: int = 1) -> None:
        with self._lock:
            self._stats[key] += by

    def stop(self) -> None:
        self._stopped.set()

    # --- assembly (stream callback thread) ---------------------------------

    def _reset_grid(self, r0: Optional[int]) -> None:
        self._r0 = r0                 # RTP of grid position 0
        self._pos = 0                 # grid position of the next expected sample
        self._buf: Optional[np.ndarray] = None
        self._filled = 0
        self._first_rtp = 0
        self._drop_through = 0        # drop samples below this position (a gap's tail)
        self._counted_k = -1          # highest interval index counted as discarded

    def _on_packet(self, rtp: int, samples) -> None:
        if self._stopped.is_set():
            return
        arr = np.asarray(samples, dtype=np.complex64)
        if arr.size == 0:
            return
        if not np.all(np.isfinite(arr)):
            arr = np.where(np.isfinite(arr), arr, np.complex64(0))
        rtp &= _MASK
        if self._r0 is None:
            self._reset_grid(rtp)
        else:
            delta = _s32(rtp - ((self._r0 + self._pos) & _MASK))
            if delta:
                if 0 < delta <= self.MAX_GAP_S * self.rate:
                    self._lose(delta, rtp)
                else:
                    self._restart(rtp, delta)
        self._consume(arr)

    def _on_lost(self, rtp: int, n: int, durations=()) -> None:
        """The stream says [rtp, rtp+n) holds no signal (zero-filled)."""
        if self._stopped.is_set():
            return
        rtp &= _MASK
        if self._r0 is None:
            self._reset_grid(rtp)
        else:
            delta = _s32(rtp - ((self._r0 + self._pos) & _MASK))
            if delta:
                if 0 < delta <= self.MAX_GAP_S * self.rate:
                    self._lose(delta, rtp)
                else:
                    self._restart(rtp, delta)
        for d in (durations or [n]):
            log.error("gap on %s: %d samples lost at rtp %d (stream zero-filled); "
                      "the interval it falls in is discarded", self.plan.name, d, rtp)
        self._bump("gaps", len(durations or [n]))
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
                  "interval discarded, new grid", self.plan.name, rtp, expected, delta)
        self._bump("gaps")
        if self._filled > 0 and self._pos // self.n > self._counted_k:
            self._bump("discarded")
        self._bump_gen()
        self._reset_grid(rtp)

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

    # --- labelling (consumer thread) ---------------------------------------

    def _poll_pairs(self, count: int) -> list[CounterPoll]:
        polls: list[CounterPoll] = []
        for i in range(count):
            if i and self._poll_spacing_s:
                time.sleep(self._poll_spacing_s)
            try:
                reply = self._control.poll_status(self._ssrc, 2.0)
            except Exception as exc:  # noqa: BLE001
                log.warning("status poll of %s failed: %s", self.plan.name, exc)
                continue
            p = _poll_from_reply(reply, self.rate)
            if p is not None:
                polls.append(p)
        return polls

    def _fit(self) -> Optional[tuple[CounterMap, list[CounterPoll]]]:
        for _ in range(self.FIT_ATTEMPTS):
            if self._stopped.is_set():
                return None
            polls = self._poll_pairs(self.POLLS)
            m = CounterMap.from_polls(polls, min_agree=3) if polls else None
            if m is not None:
                return m, polls
        return None

    @staticmethod
    def _pair(p: CounterPoll) -> StatusPair:
        return StatusPair(p.gps_time_ns, p.rtp_timesnap, p.output_samprate)

    def _install(self, fitted) -> None:
        new, polls = fitted
        if self._map is None or not new.same_run(self._map):
            if self._map is not None:
                log.error("new radiod run on %s (epoch %.3f s -> %.3f s): new epoch, "
                          "nothing projects across it", self.plan.name,
                          self._map.input_epoch_gps_s, new.input_epoch_gps_s)
            self._bump("epochs")
            self._map = new
        else:       # same run: same epoch label, fresh intercept
            self._map = replace(new, input_epoch_gps_s=self._map.input_epoch_gps_s)
        self._pairs = [self._pair(p) for p in polls][-self.PAIRS_KEPT:]

    def _pair_for(self, first_rtp: int) -> StatusPair:
        before = [p for p in self._pairs if _s32(first_rtp - p.rtp_timesnap) >= 0]
        if before:
            return min(before, key=lambda p: _s32(first_rtp - p.rtp_timesnap))
        return min(self._pairs, key=lambda p: abs(_s32(first_rtp - p.rtp_timesnap)))

    def _label(self, item) -> Optional[Interval]:
        buf, first_rtp, gen = item
        if gen != self._cur_gen:
            fitted = self._fit()
            if fitted is None:
                log.error("no counter map for %s after a discontinuity; interval "
                          "withheld, never mislabelled", self.plan.name)
                self._bump("withheld")
                return None
            self._install(fitted)
            self._cur_gen = gen
        m = self._map
        if _s32(first_rtp - m.rtp_ref) >= self._roll_s * self.rate:
            self._map = m = m.rolled_forward(first_rtp)
            self._bump("rolls")
            extra = self._poll_pairs(1)
            if extra:
                self._pairs = (self._pairs + [self._pair(extra[0])])[-self.PAIRS_KEPT:]
        return Interval(samples=buf, first_rtp=first_rtp,
                        first_input_index=m.input_index(first_rtp),
                        input_samprate=m.input_samprate,
                        input_epoch_gps_s=m.input_epoch_gps_s,
                        channel=self.plan.name, status_pair=self._pair_for(first_rtp))

    # --- the iterator -------------------------------------------------------

    def intervals(self) -> Iterator[Interval]:
        if self._control is None:
            self._control = _make_control(self.plan.status)
        half = USABLE_FRACTION * self.rate / 2
        info = self._control.ensure_channel(
            frequency_hz=float(self.plan.centre_hz), preset="iq",
            sample_rate=self.rate, encoding=F32LE,
            agc_enable=0, gain=0.0, low_edge=-half, high_edge=half)
        self._ssrc = info.ssrc          # what ensure_channel returns, never computed
        fitted = self._fit()
        if fitted is None:
            raise RuntimeError(f"no counter map for {self.plan.name}: status polls "
                               "never agreed")
        self._install(fitted)
        stream = self._stream_factory(info, self._on_packet, self._on_lost)
        stream.start()
        try:
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
        finally:
            try:
                stream.stop()
            except Exception as exc:  # noqa: BLE001
                log.warning("stream stop failed: %s", exc)
