"""Shared fixtures.  `fake_radiod` drives a RawChannelSource with no radiod."""
from __future__ import annotations

from itertools import islice
from types import SimpleNamespace

import numpy as np
import pytest

from chirp_recorder.core.channels import ChannelPlan
from chirp_recorder.core.source import RawChannelSource

MASK = 0xFFFFFFFF


class FakeRadiod:
    """One fake radiod: a control that answers status polls and a stream that
    delivers (rtp, samples) packets.  The stream delivers synchronously in
    start(), so the consumer sees a finished recording; `current_run` then
    names the run the polls describe, as a live radiod would at that moment.
    """

    def __init__(self, *, rate=64_000, interval_s=1.0, input_rate=129_600_000,
                 olen=None, drop_packets_at_rtp=(), restart_at_rtp=None,
                 restart_base=5_000_000, start_rtp=0, total_s=8.0,
                 queue_max=1000, roll_s=3600.0, poll_every=2,
                 fit_wait_s=0.3):
        self.rate = rate
        self.input_rate = input_rate
        self.olen = olen or max(1, rate // 1000)          # output samples per block
        # ilen input samples per block: olen * input_rate / rate
        self.ilen = self.olen * input_rate // rate
        assert self.ilen * rate == self.olen * input_rate
        self.packet = rate // 20
        self.drops = {d & MASK for d in drop_packets_at_rtp}
        self.restart_at = restart_at_rtp
        self.total = int(total_s * rate)
        # runs: base rtp, input index at that rtp, epoch (gps_s - input/f_in)
        self.runs = [dict(base=start_rtp & MASK, input0=1_000_000_000, epoch=1.7e9),
                     dict(base=restart_base & MASK, input0=3_000_000_000, epoch=1.7e9 + 500.0)]
        self.current_run = 0
        self.run_pos = 0               # run-relative position of the next packet
        self.poll_every = poll_every   # a status poll before every Nth packet
        self.removed: list[int] = []
        self.closed = False
        self.lifetimes: list[tuple] = []
        self.start_rtp = start_rtp
        self.ensure_calls: list[dict] = []
        self.control = SimpleNamespace(
            ensure_channel=self._ensure, poll_status=self._poll,
            remove_channel=self.removed.append,
            set_channel_lifetime=lambda ssrc, n: self.lifetimes.append((ssrc, n)),
            close=self._close)
        self.stream_factory = self._factory
        self.plan = ChannelPlan(name="4.4630MHz", status="radiod.local",
                                centre_hz=4.463e6, sample_rate_hz=rate, groups=())
        self.source = RawChannelSource(
            self.plan, interval_s=interval_s, control=self.control,
            stream_factory=self.stream_factory, queue_max=queue_max,
            poll_period_s=None, poll_spacing_s=0.0, roll_s=roll_s,
            fit_wait_s=fit_wait_s)

    # --- control ---------------------------------------------------------
    def _ensure(self, **kw):
        self.ensure_calls.append(kw)
        return SimpleNamespace(ssrc=0x1234, multicast_address="239.1.1.1", port=5004)

    def _close(self):
        self.closed = True

    def _poll(self, ssrc, timeout=2.0):
        """Status as radiod holds it now: the RTP of the stream's current
        position (floored to a block), on the current run's timeline."""
        run = self.runs[self.current_run]
        blocks = self.run_pos // self.olen
        inp = run["input0"] + blocks * self.ilen
        gps_ns = round((run["epoch"] + inp / self.input_rate) * 1e9)
        return SimpleNamespace(
            gps_time=gps_ns, rtp_timesnap=(run["base"] + blocks * self.olen) & MASK,
            frontend=SimpleNamespace(input_samples=inp, input_samprate=self.input_rate),
            filter_blocksize=self.ilen, output_samprate=self.rate)

    # --- stream ----------------------------------------------------------
    def _factory(self, channel_info, on_packet, on_lost):
        outer = self

        class _Stream:
            def start(self):
                pos = 0
                run = 0
                npk = 0
                while pos < outer.total:
                    if (run == 0 and outer.restart_at is not None
                            and outer.start_rtp + pos >= outer.restart_at):
                        run = 1
                        outer.current_run = 1
                        outer.run_pos = 0
                    if outer.poll_every and npk % outer.poll_every == 0:
                        outer.source.poll_once()
                    npk += 1
                    rtp = (outer.runs[run]["base"] + outer.run_pos) & MASK
                    n = min(outer.packet, outer.total - pos)
                    if rtp not in outer.drops:
                        # sample value = position within its run
                        s = (np.arange(n) + outer.run_pos).astype(np.float32).astype(np.complex64)
                        on_packet(rtp, s)
                    pos += n
                    outer.run_pos += n

            def stop(self):
                pass

        return _Stream()


class Wrapper:
    def __init__(self, fake: FakeRadiod):
        self.fake = fake
        self.source = fake.source

    def take(self, n):
        out = list(islice(self.source.intervals(), n))
        self.source.stop()
        return out

    def take_all(self, wait_s=0.4):
        """Everything the source delivers from a finished recording."""
        import threading
        t = threading.Timer(wait_s, self.source.stop)
        t.start()
        try:
            return list(self.source.intervals())
        finally:
            t.cancel()

    @property
    def stats(self):
        return self.source.stats


@pytest.fixture
def fake_radiod():
    def make(**kw):
        return Wrapper(FakeRadiod(**kw))
    return make


# --- timing fixtures (Task 11); every value is made up -----------------------
import json  # noqa: E402
from types import SimpleNamespace as _NS  # noqa: E402

from chirp_recorder.core.source import Interval, StatusPair  # noqa: E402

ANCHOR_UPDATED_NS = 1_800_000_000 * 10**9          # file's updated_at (whole second)
NOW = ANCHOR_UPDATED_NS + 10 * 10**9               # the test's "host clock": 10 s later
ANCHOR_UTC = 1_800_000_000 * 10**9 - 123_456_789   # anchor_utc_ns
ANCHOR_IDX = 4_000_000_000.0
EPOCH_S = 1_799_999_000.25
RATE_OUT = 64_000
GPS_EPOCH_UNIX_S = 315_964_800     # 1980-01-06 in Unix seconds
GPS_MINUS_UTC_S = 18               # the test's leap-second count, used to build GPS-scale values


def gps_ns_from_unix_ns(unix_ns):
    """What radiod's GPS_TIME would read at this UTC instant (GPS-epoch scale)."""
    return unix_ns - (GPS_EPOCH_UNIX_S - GPS_MINUS_UTC_S) * 10**9


def _iso(ns):
    import datetime as dt
    return dt.datetime.fromtimestamp(ns // 10**9, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _doc(**over):
    d = dict(radiod_status="b4-status.local", radiod_address="239.1.2.3",
             anchor_input_index=ANCHOR_IDX, anchor_utc_ns=ANCHOR_UTC,
             input_samprate=129_600_000, input_epoch_gps_s=EPOCH_S,
             counter_epoch_id="epoch-7", u_epoch_ns=230.0, k=1.0, p=0.5,
             a_level_provenance="measured", updated_at=_iso(ANCHOR_UPDATED_NS))
    d.update(over)
    return d


@pytest.fixture
def anchor_doc():
    return _doc


@pytest.fixture
def anchor_file(tmp_path):
    p = tmp_path / "t6-anchor.json"
    p.write_text(json.dumps(_doc()))
    return p


def make_interval(*, first_rtp=1_000_000, idx=ANCHOR_IDX + 1_296_000.0, epoch=EPOCH_S,
                  pair=None):
    pair = pair or StatusPair(gps_time_ns=gps_ns_from_unix_ns(1_799_999_990 * 10**9 + 5_000),
                              rtp_timesnap=900_000, output_samprate=RATE_OUT)
    return Interval(samples=np.zeros(4, np.complex64), first_rtp=first_rtp,
                    first_input_index=idx, input_samprate=129_600_000,
                    input_epoch_gps_s=epoch, channel="4.4630MHz", status_pair=pair)


@pytest.fixture
def interval():
    return make_interval()


@pytest.fixture
def interval_from_new_run():
    return make_interval(epoch=EPOCH_S + 500.0)


@pytest.fixture
def fake_auth():
    """A stand-in AuthorityReader: read() returns a snapshot that reads usable."""
    snap = _NS(offset_usable=True, rtp_to_utc_offset_ns=-1_234_567, t_level_active="T3",
               sigma_ns=2_500)
    return _NS(sigma_ns=2_500, snapshot=snap, read=lambda: snap)


# --- synthetic chirps (Task 12) ----------------------------------------------
@pytest.fixture
def synth():
    """Gated chirps from hamsci_dsp, summed over arrivals, plus complex noise.

    arrivals: (tau_s, amplitude) pairs.  Each arrival has one random carrier
    phase, constant across sweeps, so a Doppler ramp stays recoverable;
    ``doppler_hz`` adds a per-sweep phase step 2*pi*doppler_hz*period.
    ``centre_offset_hz`` shifts the transmitter's centre from the channel
    centre; the shift runs on n counted from sample 0.  ``wf`` defaults to a
    4.463 MHz, -25733.913 Hz/s, 1 s group gated at 6 ms.
    """
    from hamsci_dsp.chirp.waveform import Waveform, gated, samples_per_period

    def make(*, fs, arrivals, seconds, centre_offset_hz=0.0, doppler_hz=0.0,
             wf=None, noise=3.0, seed=7):
        wf = wf or Waveform(4.463e6, -25_733.913, 1.0, None, 0.006, 0.5)
        rng = np.random.default_rng(seed)
        n = samples_per_period(wf, fs)
        m = int(seconds / wf.period_s)
        x = np.zeros((m, n), complex)
        for tau, amp in arrivals:
            one = gated(wf, fs, tau)
            ph0 = np.exp(2j * np.pi * rng.random())
            ramp = np.exp(2j * np.pi * doppler_hz * wf.period_s * np.arange(m))
            x += amp * ph0 * ramp[:, None] * one[None, :]
        x = x.reshape(-1)
        x = x * np.exp(2j * np.pi * centre_offset_hz * np.arange(x.size) / fs)
        x += noise * (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size)) / np.sqrt(2)
        return x.astype(np.complex64)

    return make
