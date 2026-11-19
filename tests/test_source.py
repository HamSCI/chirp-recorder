import logging

import numpy as np
import pytest

from chirp_recorder.core import source as source_mod
from chirp_recorder.core.source import Interval, StatusPair

IN = 129_600_000
M32 = 2**32


def s32(x):
    x &= 0xFFFFFFFF
    return x - M32 if x >= 2**31 else x


def test_gap_discards_only_its_interval(fake_radiod):
    src = fake_radiod(rate=64_000, interval_s=1.0, drop_packets_at_rtp=[96_000])
    got = list(src.take(3))
    assert [round(i.first_rtp) for i in got] == [0, 128_000, 192_000]   # interval 64k-128k lost
    assert src.stats["discarded"] == 1


def test_index_never_counts_delivered_samples(fake_radiod):
    src = fake_radiod(rate=64_000, interval_s=1.0, drop_packets_at_rtp=[96_000])
    i0, i1 = list(src.take(2))
    assert i1.first_input_index - i0.first_input_index == 2 * 129_600_000


def test_half_sample_ratio_keeps_index_exact(fake_radiod):          # Review Focus 4
    src = fake_radiod(rate=128_000, interval_s=0.5, input_rate=129_600_000)
    a, b = list(src.take(2))
    assert b.first_input_index - a.first_input_index == 0.5 * 129_600_000
    assert a.first_input_index % 1 in (0.0, 0.5)


def test_new_radiod_run_starts_new_epoch(fake_radiod):
    src = fake_radiod(rate=64_000, interval_s=1.0, restart_at_rtp=200_000)
    eps = {i.input_epoch_gps_s for i in src.take(4)}
    assert len(eps) == 2 and src.stats["epochs"] == 2


def test_new_run_with_a_small_forward_jump_is_still_a_new_epoch(fake_radiod):
    # RTP lands 10 000 samples ahead of expectation: it looks like a gap, the
    # refit says otherwise.
    src = fake_radiod(rate=64_000, interval_s=1.0, restart_at_rtp=200_000,
                      restart_base=211_600)
    got = src.take(4)
    assert len({i.input_epoch_gps_s for i in got}) == 2
    assert src.stats["epochs"] == 2


@pytest.mark.parametrize("kw", [dict(drop_packets_at_rtp=[96_000]),
                                dict(restart_at_rtp=300_000)])
def test_interval_carries_its_status_pair(fake_radiod, kw):
    src = fake_radiod(rate=64_000, interval_s=1.0, **kw)
    got = src.take(4)
    assert len(got) == 4
    for iv in got:
        assert isinstance(iv, Interval) and isinstance(iv.status_pair, StatusPair)
        assert iv.status_pair.output_samprate == 64_000
        assert s32(iv.first_rtp - iv.status_pair.rtp_timesnap) >= 0


def test_interval_shape_and_samples(fake_radiod):
    src = fake_radiod(rate=64_000, interval_s=1.0)
    a, b = src.take(2)
    assert a.samples.shape == (64_000,) and a.samples.dtype == np.complex64
    assert a.channel == "4.4630MHz" and a.input_samprate == IN
    assert a.samples[0] == 0 and b.samples[0] == 64_000 and b.samples[-1] == 127_999
    with pytest.raises(Exception):
        a.first_rtp = 1                                    # frozen


def test_samples_per_interval_rounds(fake_radiod):
    src = fake_radiod(rate=96_000, interval_s=0.25)
    assert src.take(1)[0].samples.size == 24_000


def test_gap_logs_one_error_with_samples_lost(fake_radiod, caplog):
    caplog.set_level(logging.ERROR)
    src = fake_radiod(rate=64_000, interval_s=1.0, drop_packets_at_rtp=[96_000])
    src.take(3)
    errs = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errs) == 1 and "3200" in errs[0].getMessage()
    assert src.stats["gaps"] == 1


def test_rtp_wrap_inside_an_interval_is_no_gap(fake_radiod):
    src = fake_radiod(rate=64_000, interval_s=1.0, start_rtp=M32 - 32_000)
    got = src.take(3)
    assert [i.first_rtp for i in got] == [(M32 - 32_000 + k * 64_000) % M32 for k in range(3)]
    assert src.stats["gaps"] == 0 and src.stats["discarded"] == 0
    assert got[1].first_input_index - got[0].first_input_index == 129_600_000
    assert got[2].first_input_index - got[1].first_input_index == 129_600_000
    assert got[0].samples[0] == 0 and got[0].samples[32_000] == 32_000   # contiguous across the wrap


def test_rtp_wrap_across_a_gap(fake_radiod):
    start = M32 - 100_000
    # packet at start+96 000 is lost; the wrap falls at +100 000 inside the next interval
    src = fake_radiod(rate=64_000, interval_s=1.0, start_rtp=start,
                      drop_packets_at_rtp=[(start + 96_000) % M32])
    got = src.take(3)
    assert [i.first_rtp for i in got] == [start % M32, (start + 128_000) % M32,
                                          (start + 192_000) % M32]
    assert src.stats["gaps"] == 1 and src.stats["discarded"] == 1
    assert got[1].first_input_index - got[0].first_input_index == 2 * 129_600_000


def test_hourly_roll_forward_keeps_the_index(fake_radiod):
    rolled = fake_radiod(rate=64_000, interval_s=1.0, roll_s=2.0, total_s=8)
    plain = fake_radiod(rate=64_000, interval_s=1.0, roll_s=1e9, total_s=8)
    a, b = rolled.take(6), plain.take(6)
    assert rolled.stats["rolls"] >= 1 and plain.stats["rolls"] == 0
    assert [i.first_input_index for i in a] == [i.first_input_index for i in b]
    assert len({i.input_epoch_gps_s for i in a}) == 1


def test_full_queue_discards_whole_intervals(fake_radiod):
    src = fake_radiod(rate=64_000, interval_s=1.0, queue_max=2, total_s=8)
    got = src.take(2)
    assert [i.first_rtp for i in got] == [0, 64_000]
    assert all(i.samples.size == 64_000 for i in got)
    assert src.stats["discarded"] == 6 and src.stats["delivered"] == 2


def test_channel_is_asked_for_f32_and_the_usable_band(fake_radiod):
    src = fake_radiod(rate=64_000, interval_s=1.0)
    src.take(1)
    kw = src.fake.ensure_calls[0]
    assert kw["encoding"] == 4 and kw["sample_rate"] == 64_000
    assert kw["low_edge"] == -25_600 and kw["high_edge"] == 25_600
    assert kw["frequency_hz"] == 4.463e6


def test_default_control_carries_the_client_id(monkeypatch):
    seen = {}

    class C:
        def __init__(self, status, **kw):
            seen.update(status=status, **kw)

    import ka9q.control
    monkeypatch.setattr(ka9q.control, "RadiodControl", C)
    source_mod._make_control("radiod.local")
    assert seen == {"status": "radiod.local", "client_id": "chirp-recorder"}
