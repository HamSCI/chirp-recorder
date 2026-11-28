import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from chirp_recorder.scout.plan import Dwell
from chirp_recorder.scout.published import Published
from chirp_recorder.scout.rows import dwell_row, screen, sweep_row
from hamsci_dsp.chirp.scout_screen import Signature
from hamsci_dsp.chirp.streaks import Streak

# Column lists copied from wsprdaemon-server schema/sigmondb.sql (commit
# 5d6ab95; the tables first appear in 2994704), less received_at and ingested_at.
SWEEP_COLS = ["time", "reporter_id", "view", "centre_hz", "span_hz",
              "rate_hz_per_s", "period_s", "duration_s", "snr_db", "n_sweeps",
              "fit_score", "clipped", "matched_key", "bin_hz", "frame_s",
              "detector_version", "schema_version", "engineering"]
DWELL_COLS = ["time", "reporter_id", "view", "centre_hz", "span_hz", "bin_hz",
              "frame_s", "duration_s", "n_frames", "noise_db", "n_streaks",
              "detector_version", "schema_version", "engineering"]

T0 = datetime(2026, 11, 28, 3, 4, 5, 123456, tzinfo=timezone.utc)
D = Dwell("fine", 9.33e6, 2e6, 150.0, "band")


def streak(centre=9.33e6, rate=1e5, period=0.5, snr=30.0, n=8, fit=0.95):
    return Streak(centre, 50e3, rate, period, None, snr, n, fit, False, (0.0, 0.5))


def test_sweep_row_matches_ddl():
    r = sweep_row(streak(), dwell=D, start_utc=T0, reporter_id="AC0G/T7",
                  frame_s=0.15, matched_key="", engineering={"n": 3})
    assert list(r) == SWEEP_COLS
    assert r["time"] == "2026-11-28T03:04:05.123Z"
    assert r["engineering"] == {"n": "3"} and r["bin_hz"] == 150.0
    assert (r["detector_version"], r["schema_version"], r["clipped"]) == ("scout-1", 1, False)
    json.dumps(r)


def test_dwell_row_matches_ddl():
    r = dwell_row(dwell=D, start_utc=T0, reporter_id="AC0G/T7", frame_s=0.15,
                  duration_s=45.0, n_frames=np.int64(300), noise_db=None,
                  n_streaks=np.int64(2), engineering={"time_source": "gps_time"})
    assert list(r) == DWELL_COLS
    assert r["noise_db"] is None
    json.dumps(r)


def test_naive_or_non_utc_time_is_refused():
    with pytest.raises(ValueError):
        dwell_row(dwell=D, start_utc=datetime(2026, 1, 1), reporter_id="x",
                  frame_s=1, duration_s=1, n_frames=1, noise_db=0, n_streaks=0,
                  engineering={})
    tz = timezone(timedelta(hours=-6))
    with pytest.raises(ValueError):
        dwell_row(dwell=D, start_utc=T0.astimezone(tz), reporter_id="x",
                  frame_s=1, duration_s=1, n_frames=1, noise_db=0, n_streaks=0,
                  engineering={})


def test_rate_sign_survives_and_numpy_is_plain():
    s = streak(rate=np.float64(-1.2e5), snr=np.float32(25.0), n=np.int64(6))
    r = sweep_row(s, dwell=D, start_utc=T0, reporter_id="x", frame_s=0.15,
                  matched_key="", engineering={})
    assert r["rate_hz_per_s"] == -1.2e5
    assert type(r["snr_db"]) is float and type(r["n_sweeps"]) is int
    json.dumps(r)


def test_screen_keeps_rejected_with_tag():
    s = streak()
    pub = Published((Signature(9.33e6, 1e5, 0.5, None, "g1"),),
                    (Signature(9.33e6, 1e5, 0.5, None, "r1"),), 0.0, "x")
    kept, dropped = screen([s], pub, nyquist_hz=64.8e6, min_sweeps=5, min_fit=0.8)
    assert kept == [(s, "g1", {"local_reject": "r1"})]
    assert sum(dropped.values()) == 0
    kept, _ = screen([s], Published((), (), None, "none"), nyquist_hz=None, min_sweeps=5, min_fit=0.8)
    assert kept == [(s, "", {})]


def test_screen_drops_carriers_and_counts_them():
    carrier = streak(rate=50.0)
    weak = streak(centre=5e6, n=3)
    good = streak(centre=7e6)
    kept, dropped = screen([carrier, weak, good], Published((), (), None, "none"),
                           nyquist_hz=64.8e6, min_sweeps=5, min_fit=0.8)
    assert [k[0] for k in kept] == [good]
    assert dropped == {"carrier": 1, "alias": 0, "not_persistent": 1,
                       "subharmonic": 0, "narrow": 0}
    kept, dropped = screen([weak], Published((), (), None, "none"),
                           nyquist_hz=64.8e6, min_sweeps=2, min_fit=0.8)
    assert len(kept) == 1


def test_screen_counts_alias():
    strong = streak(centre=60e6, rate=1e5, snr=40.0)
    mirror = streak(centre=2 * 64.8e6 - 60e6, rate=-1e5, snr=10.0)
    kept, dropped = screen([strong, mirror], Published((), (), None, "none"),
                           nyquist_hz=64.8e6, min_sweeps=5, min_fit=0.8)
    assert dropped["alias"] == 1 and [k[0] for k in kept] == [strong]


def _row(s):
    return sweep_row(s, dwell=D, start_utc=T0, reporter_id="x", frame_s=0.15,
                     matched_key="", engineering={})


def test_screen_drops_a_streak_without_period():
    s = streak(n=9, period=None)
    kept, dropped = screen([s], Published((), (), None, "none"),
                           nyquist_hz=64.8e6, min_sweeps=5, min_fit=0.8)
    assert kept == [] and dropped["not_persistent"] == 1


def test_screen_min_fit_passes_through():
    s = streak(fit=0.7)
    none = Published((), (), None, "none")
    kept, _ = screen([s], none, nyquist_hz=64.8e6, min_sweeps=5, min_fit=0.6)
    assert [k[0] for k in kept] == [s]
    kept, dropped = screen([s], none, nyquist_hz=64.8e6, min_sweeps=5, min_fit=0.8)
    assert kept == [] and dropped["not_persistent"] == 1


def test_sweep_row_refuses_missing_period():
    with pytest.raises(ValueError, match="period"):
        _row(streak(period=None))


def test_sweep_row_allows_missing_duration():
    r = _row(streak())
    assert r["duration_s"] is None
    assert json.loads(json.dumps(r))["duration_s"] is None


def test_screen_drops_a_subharmonic_twin():
    """One source, found again at twice its period (every other sweep):
    keep the shorter period, count the other as "subharmonic"."""
    short = streak(centre=5.2631e6, rate=-25920.0, period=1.0)
    long = streak(centre=5.2637e6, rate=-25475.0, period=1.9991)
    kept, dropped = screen([long, short], Published((), (), None, "none"),
                           nyquist_hz=64.8e6, min_sweeps=5, min_fit=0.6)
    assert [k[0] for k in kept] == [short]
    assert dropped["subharmonic"] == 1


def test_screen_keeps_a_non_integer_period_ratio():
    a = streak(centre=5.2631e6, rate=-25920.0, period=1.0)
    b = streak(centre=5.2631e6, rate=-25920.0, period=1.5)
    kept, dropped = screen([a, b], Published((), (), None, "none"),
                           nyquist_hz=64.8e6, min_sweeps=5, min_fit=0.6)
    assert len(kept) == 2 and dropped["subharmonic"] == 0


def test_screen_keeps_a_twice_period_streak_elsewhere():
    a = streak(centre=5.2631e6, rate=-25920.0, period=1.0)
    b = streak(centre=5.2831e6, rate=-25920.0, period=2.0)
    kept, dropped = screen([a, b], Published((), (), None, "none"),
                           nyquist_hz=64.8e6, min_sweeps=5, min_fit=0.6)
    assert len(kept) == 2 and dropped["subharmonic"] == 0


def test_screen_drops_a_narrow_streak():
    narrow = Streak(5.3595e6, 5000.0, -760.0, 0.58, None, 26.0, 11, 0.9, False, (0.0,))
    kept, dropped = screen([narrow], Published((), (), None, "none"),
                           nyquist_hz=64.8e6, min_sweeps=5, min_fit=0.6)
    assert kept == [] and dropped["narrow"] == 1 and dropped["not_persistent"] == 0


def test_screen_keeps_a_codar_wide_streak():
    codar = Streak(4.4626e6, 25_700.0, -25_730.0, 1.0, None, 17.0, 11, 0.69, False, (0.0,))
    kept, dropped = screen([codar], Published((), (), None, "none"),
                           nyquist_hz=64.8e6, min_sweeps=5, min_fit=0.6, min_span_hz=10_000.0)
    assert [k[0] for k in kept] == [codar] and dropped["narrow"] == 0
