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


# --- fix round 1 -----------------------------------------------------------

from types import SimpleNamespace

from ka9q.stream_quality import GapSource

RUN0_EPOCH, RUN1_EPOCH = 1.7e9, 1.7e9 + 500.0


def test_status_pair_is_a_recent_poll_at_or_before(fake_radiod):
    # polls follow the stream (one per 6400 samples), so the pair sits just
    # before the interval, never 8 blocks before a stream that has not begun
    src = fake_radiod(rate=64_000, interval_s=1.0, drop_packets_at_rtp=[96_000])
    for iv in src.take(3):
        age = s32(iv.first_rtp - iv.status_pair.rtp_timesnap)
        assert 0 <= age <= 6_400


def test_no_old_run_interval_carries_the_new_epoch(fake_radiod):
    # the reviewer's repro: a gap, then a radiod restart
    src = fake_radiod(rate=64_000, interval_s=1.0, drop_packets_at_rtp=[96_000],
                      restart_at_rtp=300_000)
    got = src.take_all()
    old = [i for i in got if i.first_rtp < 1_000_000]
    new = [i for i in got if i.first_rtp >= 1_000_000]
    assert [i.first_rtp for i in old] == [0, 128_000, 192_000]
    assert new, "the new run delivers"
    for i in old:
        assert abs(i.input_epoch_gps_s - RUN0_EPOCH) < 1.0
        assert abs(i.first_input_index - (1_000_000_000 + i.first_rtp * 2025)) < 1
        assert i.samples[0] == i.first_rtp
    for i in new:
        assert abs(i.input_epoch_gps_s - RUN1_EPOCH) < 1.0
        assert abs(i.first_input_index
                   - (3_000_000_000 + (i.first_rtp - 5_000_000) * 2025)) < 1
        assert i.samples[0] == i.first_rtp - 5_000_000
    assert src.stats["epochs"] == 2


def test_interval_with_no_pair_at_or_before_is_discarded_not_stamped(fake_radiod, caplog):
    # nothing can be polled before the first packet of a restarted run
    caplog.set_level(logging.ERROR)
    src = fake_radiod(rate=64_000, interval_s=1.0, restart_at_rtp=200_000)
    got = src.take_all()
    assert 5_000_000 not in [i.first_rtp for i in got]        # first run-1 interval
    assert 5_064_000 in [i.first_rtp for i in got]
    assert any("no status pair at or before rtp 5000000" in r.getMessage()
               for r in caplog.records)
    for i in got:
        assert s32(i.first_rtp - i.status_pair.rtp_timesnap) >= 0


def test_superseded_generation_without_a_map_is_discarded(fake_radiod, caplog):
    caplog.set_level(logging.ERROR)
    # no polls during the stream: generations 1 and 2 never get a map
    src = fake_radiod(rate=64_000, interval_s=1.0, drop_packets_at_rtp=[96_000],
                      restart_at_rtp=300_000, poll_every=0)
    got = src.take_all()
    assert [i.first_rtp for i in got] == [0]
    assert src.stats["epochs"] == 1
    # gen 0: intervals 0 (delivered); gen 1: 128k, 192k (superseded) and the
    # restart's partial (counted at the restart); gen 2 waits then is refused
    assert src.stats["discarded"] >= 6
    assert any("superseded" in r.getMessage() for r in caplog.records)


def test_restart_re_ensures_the_channel(fake_radiod):
    src = fake_radiod(rate=64_000, interval_s=1.0, restart_at_rtp=200_000)
    src.take_all()
    assert len(src.fake.ensure_calls) == 2 and src.stats["restarts"] == 1
    gap = fake_radiod(rate=64_000, interval_s=1.0, drop_packets_at_rtp=[96_000])
    gap.take_all()
    assert len(gap.fake.ensure_calls) == 1          # a gap leaves the channel alone


def test_silence_re_ensures_the_channel(fake_radiod):
    src = fake_radiod(rate=64_000, interval_s=1.0, poll_every=0)
    import time
    now = [time.monotonic()]
    src.source._clock = lambda: now[0]
    src.source._silence_s = 10.0
    src.take(1)
    src.source._started = True
    src.source._stopped.clear()
    before = len(src.fake.ensure_calls)
    now[0] += 30.0
    src.source.poll_once()
    assert len(src.fake.ensure_calls) == before + 1


def test_stop_removes_the_channel_and_closes_control(fake_radiod):
    src = fake_radiod(rate=64_000, interval_s=1.0)
    src.take(1)
    assert src.fake.removed == [0x1234] and src.fake.closed
    assert src.fake.ensure_calls[0]["lifetime"] == src.source.LIFETIME_FRAMES
    src.source.stop()                                  # idempotent
    assert src.fake.removed == [0x1234]


def test_poller_refreshes_the_lifetime(fake_radiod):
    src = fake_radiod(rate=64_000, interval_s=1.0)
    src.take_all()
    assert src.fake.lifetimes and src.fake.lifetimes[0] == (0x1234, 3000)


# --- _KaStream (I4) ----------------------------------------------------------

class _Cls:
    def __init__(self, channel, on_samples):
        self.on_samples = on_samples


def _ka(log_calls):
    from chirp_recorder.core.source import _KaStream
    return _KaStream(None, lambda r, s: log_calls.append(("pkt", r, len(s))),
                     lambda r, n, d: log_calls.append(("lost", r, n, tuple(d))),
                     _stream_cls=_Cls)


def _q(rtp0, start, gaps=()):
    return SimpleNamespace(delivered_rtp_start=rtp0, batch_start_sample=start,
                           batch_gaps=list(gaps))


def _gap(src, pos, dur):
    return SimpleNamespace(source=src, position_samples=pos, duration_samples=dur)


def _batch(n, *zero_spans):
    a = np.ones(n, np.complex64) * (1 + 2j)
    for off, ln in zero_spans:
        a[off:off + ln] = 0
    return a


def test_kastream_gap_costs_only_its_span():
    calls = []
    ka = _ka(calls)
    ka._on_samples(_batch(1000, (300, 200)),
                   _q(5_000, 100, [_gap(GapSource.NETWORK_LOSS, 400, 200)]))
    assert calls == [("pkt", 5_000, 300), ("lost", 5_300, 200, (200,)),
                     ("pkt", 5_500, 500)]


def test_kastream_ignores_a_wrong_reported_position():
    # ka9q-python 3.27.0 reports positions late after a loss: here by 200
    calls = []
    ka = _ka(calls)
    ka._on_samples(_batch(1000, (300, 200)),
                   _q(5_000, 100, [_gap(GapSource.NETWORK_LOSS, 600, 200)]))
    assert calls == [("pkt", 5_000, 300), ("lost", 5_300, 200, (200,)),
                     ("pkt", 5_500, 500)]


def test_kastream_reported_gap_without_a_zero_run_condemns_the_batch():
    calls = []
    ka = _ka(calls)
    ka._on_samples(_batch(500),
                   _q(3_000, 0, [_gap(GapSource.NETWORK_LOSS, 9_999, 50)]))
    assert calls == [("lost", 3_000, 500, (50,))]


def test_kastream_run_count_disagreement_condemns_the_batch():
    calls = []
    ka = _ka(calls)
    ka._on_samples(_batch(1000, (100, 100)),
                   _q(3_000, 0, [_gap(GapSource.NETWORK_LOSS, 100, 100),
                                 _gap(GapSource.NETWORK_LOSS, 600, 100)]))
    assert calls == [("lost", 3_000, 1000, (100, 100))]


def test_kastream_unreported_zero_run_is_never_shipped_clean():
    calls = []
    ka = _ka(calls)
    ka._on_samples(_batch(600, (200, 150)), _q(7_000, 0))
    assert calls == [("pkt", 7_000, 200), ("lost", 7_200, 150, (150,)),
                     ("pkt", 7_350, 250)]


def test_kastream_short_zero_runs_are_signal():
    calls = []
    ka = _ka(calls)
    ka._on_samples(_batch(600, (200, 10)), _q(7_000, 0))
    assert calls == [("pkt", 7_000, 600)]


def test_kastream_empty_payload_condemns_nothing():
    calls = []
    ka = _ka(calls)
    ka._on_samples(_batch(640),
                   _q(1_000, 0, [_gap(GapSource.EMPTY_PAYLOAD, 100, 320)]))
    assert calls == [("pkt", 1_000, 640)]


def test_kastream_missing_rtp_start_is_loud_and_never_synthesised(caplog):
    caplog.set_level(logging.ERROR)
    calls = []
    ka = _ka(calls)
    ka._on_samples(_batch(100), _q(None, 0))       # nowhere to place it
    assert calls == [] and "no delivered_rtp_start" in caplog.text
    ka._on_samples(_batch(100), _q(2_000, 0))
    ka._on_samples(_batch(100), _q(None, 100))
    assert calls == [("pkt", 2_000, 100), ("lost", 2_100, 100, (100,))]


def test_kastream_with_the_real_resequencer_ships_no_hole():
    """Drive ka9q-python's own resequencer through two lost packets (the
    reviewer's double-count case) and feed each emitted chunk, with the
    quality it would carry, to the adapter."""
    from ka9q.resequencer import PacketResequencer, RTPPacket
    per = 100
    rs = PacketResequencer(buffer_size=8, samples_per_packet=per, sample_rate=16_000)
    calls, delivered = [], 0
    ka = _ka(calls)
    ka._min_zero_run = per
    lost_seq = {10, 40}
    for seq in range(80):
        if seq in lost_seq:
            continue
        sig = (np.ones(per) * (1 + seq) + 1j).astype(np.complex64)
        out, gaps = rs.process_packet(RTPPacket(seq, seq * per, 1, sig))
        if out is not None and len(out):
            ka._on_samples(out, _q(rs.last_chunk_rtp_start, delivered, gaps))
            delivered += len(out)
    pkts = [c for c in calls if c[0] == "pkt"]
    lost = [c for c in calls if c[0] == "lost"]
    assert [c[1] for c in lost] == [10 * per, 40 * per] and all(c[2] == per for c in lost)
    # every shipped piece is signal: its samples cover everything but the holes
    assert sum(c[2] for c in pkts) == delivered - 2 * per
    assert sum(c[2] for c in calls) == delivered


def test_changed_ssrc_on_re_ensure_stops_the_source(fake_radiod, caplog):
    caplog.set_level(logging.ERROR)
    src = fake_radiod(rate=64_000, interval_s=1.0, restart_at_rtp=200_000)
    orig = src.fake._ensure

    def changing(**kw):
        info = orig(**kw)
        if len(src.fake.ensure_calls) > 1:
            info.ssrc = 0x9999
        return info
    src.fake.control.ensure_channel = changing
    with pytest.raises(RuntimeError, match="changed ssrc"):
        src.take_all()
    assert "returned ssrc" in caplog.text
