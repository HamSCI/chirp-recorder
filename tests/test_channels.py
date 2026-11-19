from chirp_recorder.core.catalogue import Transmitter
from chirp_recorder.core.channels import form_channels, channel_errors


def tx(i, mhz, rate=-25_733.913, bw=25_734, period=1.0):
    return Transmitter(i, "codar", mhz * 1e6, rate, period, bw, 40.0, -74.0)


def test_kfs_lesson_2_far_centres_never_share_a_channel():
    chans = form_channels([tx("A", 4.550), tx("B", 4.785)], "omni-status.local")
    assert len(chans) == 2


def test_tdma_sites_share_one_group():
    (ch,) = form_channels([tx("R1", 4.463), tx("R2", 4.463)], "s")
    assert len(ch.groups) == 1 and set(ch.groups[0].tx_ids) == {"R1", "R2"}


def test_13mhz_neighbours_pack_into_one_wider_channel():
    txs = [tx("U", 13.475, 95_580, 47_790, 0.5), tx("F", 13.4499, -99_260, 49_630, 0.5),
           tx("P", 13.500, -99_260, 49_630, 0.5)]
    (ch,) = form_channels(txs, "s")
    assert ch.sample_rate_hz in (128_000, 192_000) and len(ch.groups) == 3


def test_validation_refuses_what_the_channel_cannot_hear():
    (ch,) = form_channels([tx("A", 4.550)], "s")
    bad = type(ch)(ch.name, ch.status, ch.centre_hz, ch.sample_rate_hz,
                   ch.groups + form_channels([tx("B", 4.785)], "s")[0].groups)
    errs = channel_errors(bad, {"A": tx("A", 4.550), "B": tx("B", 4.785)})
    assert any("B" in e for e in errs)


def test_placeholder_status_refused():
    (ch,) = form_channels([tx("A", 4.463)], "<configure-via-config-init>")
    assert channel_errors(ch, {"A": tx("A", 4.463)})


def test_good_channel_has_no_errors_and_smallest_rate():
    (ch,) = form_channels([tx("A", 4.463)], "s")
    assert ch.sample_rate_hz == 64_000
    assert channel_errors(ch, {"A": tx("A", 4.463)}) == []


def test_channel_centres_on_its_span():
    (ch,) = form_channels([tx("A", 4.463), tx("B", 4.470)], "s")
    lo, hi = 4.463e6 - 12_867, 4.470e6 + 12_867
    assert abs(ch.centre_hz - (lo + hi) / 2) < 1


def test_fractional_sample_period_refused():
    t = tx("A", 4.463, period=1 / 3)
    (ch,) = form_channels([t], "s")
    assert any("whole" in e for e in channel_errors(ch, {"A": t}))


def test_unplaceable_wide_sweep_is_refused_not_dropped():
    t = tx("W", 5.0, bw=400_000)
    (ch,) = form_channels([t], "s")
    assert any("W" in e for e in channel_errors(ch, {"W": t}))
