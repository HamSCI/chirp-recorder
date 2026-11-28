from chirp_recorder.scout.plan import DwellCycle
from pytest import approx
from chirp_recorder.scout.published import Published
from chirp_recorder.scout.settings import ScoutSettings
from hamsci_dsp.chirp.scout_screen import Signature

SET = ScoutSettings(targets_mhz=(4.86, 9.33, 13.50, 16.15, 25.40))
NONE = Published((), (), None, "none")


def chirp(mhz, key="g"):
    return Signature(mhz * 1e6, 1e5, 0.5, None, key)


def test_flag_goes_first_once():
    c = DwellCycle(SET)
    c.flag(7.0e6, 100.0)
    d = c.next(NONE, 110.0)
    assert (d.reason, d.centre_hz, d.view) == ("flag", 7.0e6, "fine")
    assert c.next(NONE, 111.0).reason == "band"


def test_stale_flag_is_dropped():
    c = DwellCycle(SET)
    c.flag(7.0e6, 0.0)
    assert c.next(NONE, 61 * 60.0).reason == "band"


def test_flag_window_is_clamped():
    c = DwellCycle(SET)
    c.flag(100e3, 0.0)
    c.flag(SET.nyquist_hz, 0.0)
    assert c.next(NONE, 1.0).centre_hz == SET.fine_span_hz / 2
    assert c.next(NONE, 1.0).centre_hz == SET.nyquist_hz - SET.fine_span_hz / 2


def test_cycle_visits_every_target():
    c = DwellCycle(SET)
    got = [c.next(NONE, 0.0).centre_hz for _ in range(5)]
    assert sorted(got) == sorted(t * 1e6 for t in SET.targets_mhz)
    assert c.next(NONE, 0.0).centre_hz == got[0]


def test_chirp_windows_interleave():
    c = DwellCycle(SET)
    pub = Published((chirp(7.0), chirp(4.463)), (), 0.0, "x")
    seq = [(d.reason, d.centre_hz) for d in (c.next(pub, 0.0) for _ in range(4))]
    assert seq == [("band", 4.86e6), ("chirp", 7.0e6), ("band", 9.33e6),
                   ("chirp", 7.0e6)]


def test_near_duplicate_chirps_merge_and_position_survives_changes():
    c = DwellCycle(SET)
    pub = Published((chirp(7.0), chirp(7.5), chirp(11.0)), (), 0.0, "x")
    assert [d.centre_hz for d in (c.next(pub, 0) for _ in range(4))] == [
        4.86e6, 7.0e6, 9.33e6, 11.0e6]
    # The list changes between calls; the cycle carries on, not from the top.
    pub2 = Published((chirp(11.0),), (), 1.0, "x")
    assert c.next(pub2, 1).centre_hz == 13.50e6
    assert c.next(pub2, 1).centre_hz == 11.0e6
    assert c.next(NONE, 2).centre_hz == approx(16.15e6)
    assert c.next(NONE, 2).centre_hz == 25.40e6
