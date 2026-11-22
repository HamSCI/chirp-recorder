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

    arrivals: (tau_s, amplitude[, doppler_hz]) tuples; a third value
    overrides ``doppler_hz`` for that arrival.  Each arrival has one random carrier
    phase, constant across sweeps, so a Doppler ramp stays recoverable;
    ``doppler_hz`` shifts the frequency continuously through the interval
    (doppler_mode="continuous"; real Doppler) or adds one phase step per
    sweep, 2*pi*doppler_hz*period (doppler_mode="step").
    ``centre_offset_hz`` shifts the transmitter's centre from the channel
    centre; the shift runs on n counted from sample 0.  ``wf`` defaults to a
    4.463 MHz, -25733.913 Hz/s, 1 s group gated at 1945.6 us.
    """
    from hamsci_dsp.chirp.waveform import Waveform, gated, samples_per_period

    def make(*, fs, arrivals, seconds, centre_offset_hz=0.0, doppler_hz=0.0,
             wf=None, noise=3.0, seed=7, doppler_mode="continuous"):
        wf = wf or Waveform(4.463e6, -25_733.913, 1.0, None, 1945.6e-6, 0.5)
        rng = np.random.default_rng(seed)
        n = samples_per_period(wf, fs)
        m = int(seconds / wf.period_s)
        x = np.zeros(m * n, complex)
        idx = np.arange(m * n)
        for arr in arrivals:
            tau, amp = arr[0], arr[1]
            f = arr[2] if len(arr) > 2 else doppler_hz      # per-arrival Doppler
            sweeps_one = np.tile(gated(wf, fs, tau), m)
            ph0 = np.exp(2j * np.pi * rng.random())
            if doppler_mode == "step":       # one phase step per sweep
                ramp = np.repeat(np.exp(2j * np.pi * f * wf.period_s * np.arange(m)), n)
                x += amp * ph0 * ramp * sweeps_one
            else:                            # a continuous frequency offset
                x += amp * ph0 * sweeps_one * np.exp(2j * np.pi * f * idx / fs)
        x = x * np.exp(2j * np.pi * centre_offset_hz * np.arange(x.size) / fs)
        x += noise * (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size)) / np.sqrt(2)
        return x.astype(np.complex64)

    return make


# --- physics stopgap (Task 14); every value is made up -----------------------
import datetime as _dt  # noqa: E402

PHYS_RATE = 64_000
PHYS_RX = (41.0, -76.0)        # a coarse made-up receiver, not a real site
PHYS_TX = (38.0, -75.5)        # a coarse made-up transmitter, ~340 km away
PHYS_SWEEP_RATE = -25_733.913
PHYS_INPUT_RATE = 129_600_000
PHYS_START_NS = 1_800_000_000 * 10**9 + 250_000_000


def phys_tx_block():
    return {"id": "TX1", "center_freq_hz": 4_463_000,
            "sweep_rate_hz_per_s": PHYS_SWEEP_RATE, "sweep_bw_hz": 25_734.0,
            "sweep_repetition_hz": 1.0, "tx_lat_deg": PHYS_TX[0],
            "tx_lon_deg": PHYS_TX[1]}


@pytest.fixture
def tx_block():
    return phys_tx_block()


@pytest.fixture
def ref_kw():
    """codar-sounder's _TransmitterPipeline keywords that match PHYS_SETTINGS."""
    return dict(receiver_lat=PHYS_RX[0], receiver_lon=PHYS_RX[1], radiod_id="b4",
                radiod_status_dns="radiod.local", sample_rate_hz=float(PHYS_RATE),
                coherent_seconds=60.0, range_min_km=200.0, range_max_km=800.0,
                snr_threshold_db=5.0, clutter_window=20, max_peaks_per_cpi=4,
                host_call="AC0G", host_grid="EM38", reporter_id="AC0G/B4",
                processing_version="codar-ref")


@pytest.fixture
def physics_settings():
    return dict(enabled=True, coherent_seconds=60, range_min_km=200, range_max_km=800,
                snr_threshold_db=5.0, max_peaks_per_cpi=4, clutter_window=20,
                # codar-sounder's synthetic TX crosses the channel centre at
                # x[0], 0.25 s after the UTC second (PHYS_START_NS), so its
                # sweep starts half a sweep earlier: 0.75 s after the second.
                interval_s=30, tdma_offset_samples={"TX1": PHYS_RATE * 3 // 4},
                receiver_lat=PHYS_RX[0], receiver_lon=PHYS_RX[1],
                position_source="precise")


@pytest.fixture
def plan_and_txs():
    from chirp_recorder.core.catalogue import Transmitter
    from chirp_recorder.core.channels import ChannelPlan, groups_for
    tx = Transmitter(id="TX1", kind="codar", centre_hz=4_463_000.0,
                     rate_hz_per_s=PHYS_SWEEP_RATE, period_s=1.0, sweep_bw_hz=25_734.0,
                     lat=PHYS_TX[0], lon=PHYS_TX[1])
    plan = ChannelPlan("4.4630MHz", "radiod.local", 4_463_000.0, PHYS_RATE,
                       groups_for([tx]))
    return plan, {"TX1": tx}


@pytest.fixture
def synth_cpi():
    """codar-sounder's SyntheticIQSource, realtime=False: one CPI of samples."""
    from codar_sounder.core.stream import SyntheticIQSource

    def make(*, fs=PHYS_RATE, seconds=60, group_range_km=420.0, seed=0):
        src = SyntheticIQSource(
            sample_rate_hz=float(fs), sweep_rate_hz_per_s=PHYS_SWEEP_RATE,
            sweep_repetition_hz=1.0, cpi_seconds=float(seconds),
            target_group_range_km=group_range_km, realtime=False, seed=seed)
        x, _ = next(iter(src))
        start = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc) + _dt.timedelta(
            microseconds=PHYS_START_NS // 1000)
        return x, start
    return make


def _label(utc_ns):
    from chirp_recorder.core.timing import Label
    return Label(utc_ns=utc_ns, delay_basis="t6", t_level="T6", provenance={}, engineering={})


@pytest.fixture
def split_into_intervals():
    """(Interval, Label) pairs cut from x; skip_s opens a gap before each
    interval after the first by that many intervals' worth of input index."""
    def split(x, start, *, interval_s=30, first_index=4_000_000_000.0, gap_after=None):
        n = int(interval_s * PHYS_RATE)
        step = interval_s * PHYS_INPUT_RATE
        start_ns = int((start - _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc))
                       / _dt.timedelta(microseconds=1)) * 1000
        out = []
        for k in range(len(x) // n):
            idx = first_index + k * step + (step if gap_after is not None and k > gap_after
                                            else 0)
            iv = Interval(samples=x[k * n:(k + 1) * n], first_rtp=k * n,
                          first_input_index=idx, input_samprate=PHYS_INPUT_RATE,
                          input_epoch_gps_s=EPOCH_S, channel="4.4630MHz",
                          status_pair=None)
            out.append((iv, _label(start_ns + k * interval_s * 10**9)))
        return out
    return split


@pytest.fixture
def contiguous_intervals(synth_cpi, split_into_intervals):
    x, start = synth_cpi()
    return split_into_intervals(x, start)


@pytest.fixture
def intervals_with_gap(synth_cpi, split_into_intervals):
    x, start = synth_cpi()
    return split_into_intervals(x, start, gap_after=0)


class FakeSink:
    """Records the rows a writer receives, with the producer it was built with."""
    registry: dict = {}

    def __init__(self, name, producer=None, **kw):
        self.name, self.producer, self.kw = name, producer, kw
        self.rows_in: list = []
        self.closed = False

    def insert(self, rows):
        self.rows_in.extend(dict(r, producer=self.producer) for r in rows)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_sink():
    sinks: dict = {}

    def make(name, producer=None):
        sinks[name] = FakeSink(name, producer)
        return sinks[name]
    make.rows = lambda name: sinks[name].rows_in
    return make
