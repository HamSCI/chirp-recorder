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
                 queue_max=1000, roll_s=3600.0):
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
        self.poll_n = [0, 0]
        self.start_rtp = start_rtp
        self.ensure_calls: list[dict] = []
        self.control = SimpleNamespace(ensure_channel=self._ensure,
                                       poll_status=self._poll)
        self.stream_factory = self._factory
        self.plan = ChannelPlan(name="4.4630MHz", status="radiod.local",
                                centre_hz=4.463e6, sample_rate_hz=rate, groups=())
        self.source = RawChannelSource(
            self.plan, interval_s=interval_s, control=self.control,
            stream_factory=self.stream_factory, queue_max=queue_max,
            poll_spacing_s=0.0, roll_s=roll_s)

    # --- control ---------------------------------------------------------
    def _ensure(self, **kw):
        self.ensure_calls.append(kw)
        return SimpleNamespace(ssrc=0x1234, multicast_address="239.1.1.1", port=5004)

    def _poll(self, ssrc, timeout=2.0):
        run = self.runs[self.current_run]
        blocks = -8 + self.poll_n[self.current_run]
        self.poll_n[self.current_run] += 1
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
                while pos < outer.total:
                    if (run == 0 and outer.restart_at is not None
                            and outer.start_rtp + pos >= outer.restart_at):
                        run = 1
                        outer.current_run = 1
                        run_pos = 0
                    if run == 0:
                        rtp = (outer.start_rtp + pos) & MASK
                    else:
                        rtp = (outer.runs[1]["base"] + run_pos) & MASK
                    n = min(outer.packet, outer.total - pos)
                    if rtp not in outer.drops:
                        # sample value = position within its run: lets a test
                        # check which samples an interval holds
                        base = pos if run == 0 else run_pos
                        s = (np.arange(n) + base).astype(np.float32).astype(np.complex64)
                        on_packet(rtp, s)
                    pos += n
                    if run == 1:
                        run_pos += n

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

    @property
    def stats(self):
        return self.source.stats


@pytest.fixture
def fake_radiod():
    def make(**kw):
        return Wrapper(FakeRadiod(**kw))
    return make
