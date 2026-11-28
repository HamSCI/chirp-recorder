import pytest

from chirp_recorder.config import ConfigError, load_instance
from chirp_recorder.scout.settings import ScoutSettings, scout_settings
from tests.test_config import GOOD, write

SCOUT_ONLY = """
[instance]
reporter_id = "AC0G/T7"
[station]
receiver_lat = 38.0
receiver_lon = -95.0
[paths]
output_dir = "{out}"
"""


def cfg_for(tmp_path, extra="", base=GOOD):
    from chirp_recorder.core.catalogue import load_seed
    a = next(t for t in load_seed().values() if 4.4e6 < t.centre_hz < 4.6e6)
    text = base.replace('4463000', str(int(a.centre_hz))).replace(
        '4785000', str(int(a.centre_hz) + 1000)).replace('"AGL1"', f'"{a.id}"')
    return load_instance("AC0G-T7", write(tmp_path, text + extra))


def test_defaults_when_no_scout_table(tmp_path):
    cfg = cfg_for(tmp_path)
    s = scout_settings(cfg)
    assert s == ScoutSettings(status="omni-status.local",
                              state_dir=cfg.output_dir / "scout")
    assert not s.enabled and s.nyquist_hz == 64.8e6 and s.dwell_s == 45.0
    assert s.targets_mhz == (4.86, 9.33, 13.50, 16.15, 25.40)
    assert s.public_hosts[0] == "https://db2.wspr.live"


def test_status_defaults_to_first_channel(tmp_path):
    cfg = cfg_for(tmp_path, '\n[scout]\nenabled = true\ndwell_s = 30\n')
    s = scout_settings(cfg)
    assert s.status == "omni-status.local" and s.enabled and s.dwell_s == 30.0
    cfg = cfg_for(tmp_path, '\n[scout]\nstatus = "other.local"\nstate_dir = "/x/y"\n')
    s = scout_settings(cfg)
    assert s.status == "other.local" and str(s.state_dir) == "/x/y"


def test_bad_values_all_listed(tmp_path):
    cfg = cfg_for(tmp_path, '\n[scout]\ndwell_s = -1\nfine_bin_hz = 10\n'
                  'targets_mhz = [70.0]\ncolour = 3\n')
    with pytest.raises(ConfigError) as e:
        scout_settings(cfg)
    m = str(e.value)
    assert "dwell_s" in m and "fine_bin_hz" in m and "targets_mhz" in m and "colour" in m


def test_coarse_bin_count_checked(tmp_path):
    cfg = cfg_for(tmp_path, '\n[scout]\ncoarse_bin_hz = 1000\n')
    with pytest.raises(ConfigError, match="coarse_bin_hz"):
        scout_settings(cfg)


def test_no_channel_enabled_is_an_error_disabled_is_allowed(tmp_path):
    from types import SimpleNamespace
    base = dict(channels=[], output_dir=tmp_path)
    s = scout_settings(SimpleNamespace(raw={}, **base))
    assert s.status == ""
    with pytest.raises(ConfigError, match="status"):
        scout_settings(SimpleNamespace(raw={"scout": {"enabled": True}}, **base))
    s = scout_settings(SimpleNamespace(
        raw={"scout": {"enabled": True, "status": "s.local"}}, **base))
    assert s.status == "s.local"
