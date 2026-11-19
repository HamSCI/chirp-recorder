import logging
import pytest

from chirp_recorder.config import ConfigError, load_instance

# Made-up coordinates throughout.
GOOD = """
[instance]
reporter_id = "AC0G/T7"
[station]
callsign = "AC0G"
receiver_lat = 38.0
receiver_lon = -95.0
[paths]
output_dir = "{out}"
[[channel]]
name = "low"
status = "omni-status.local"
centre_hz = 4463000
sample_rate_hz = 64000
transmitters = ["AGL1"]
[[channel]]
name = "high"
status = "omni-status.local"
centre_hz = 4785000
sample_rate_hz = 64000
transmitters = ["AGL1"]
"""


def write(tmp_path, text, key="AC0G-T7"):
    (tmp_path / f"{key}.toml").write_text(text.format(out=tmp_path / "out"))
    return tmp_path


def test_two_channels_load(tmp_path):
    from chirp_recorder.core.catalogue import load_seed
    seed = load_seed()
    a, b = [t for t in seed.values() if 4.4e6 < t.centre_hz < 4.6e6][:1] * 2
    text = GOOD.replace('4463000', str(int(a.centre_hz))).replace(
        '4785000', str(int(a.centre_hz) + 1000)).replace('"AGL1"', f'"{a.id}"')
    cfg = load_instance("AC0G-T7", write(tmp_path, text))
    assert [c.name for c in cfg.channels] == ["low", "high"]
    assert cfg.interval_s == 30 and cfg.picker_phase == "establish" and cfg.max_channels == 2
    assert cfg.position_source == "precise"
    assert (cfg.receiver_lat, cfg.receiver_lon) == (38.0, -95.0)


def _one(tmp_path, station, extra_channel=""):
    from chirp_recorder.core.catalogue import load_seed
    a = next(t for t in load_seed().values() if 4.4e6 < t.centre_hz < 4.6e6)
    text = f"""
[instance]
reporter_id = "AC0G/T7"
[station]
{station}
[[channel]]
name = "low"
status = "omni-status.local"
centre_hz = {int(a.centre_hz)}
sample_rate_hz = 64000
transmitters = ["{a.id}"]
{extra_channel}
"""
    (tmp_path / "AC0G-T7.toml").write_text(text)
    return tmp_path


def test_precise_position(tmp_path):
    cfg = load_instance("AC0G-T7", _one(tmp_path, "receiver_lat = 38.5\nreceiver_lon = -95.5\ngrid_square = \"EM18\""))
    assert cfg.position_source == "precise"
    assert (cfg.receiver_lat, cfg.receiver_lon) == (38.5, -95.5)
    assert cfg.grid == "EM18"


def test_grid_only_uses_centre_and_warns(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        cfg = load_instance("AC0G-T7", _one(tmp_path, 'grid_square = "EM28ab"'))
    assert cfg.position_source == "grid_centre"
    assert cfg.receiver_lat == pytest.approx(38.0625) and cfg.receiver_lon == pytest.approx(-95.9583333, abs=1e-6)
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1
    assert "AC0G-T7" in warns[0].getMessage() and "withheld" in warns[0].getMessage()


def test_neither_position_is_an_error(tmp_path):
    with pytest.raises(ConfigError) as e:
        load_instance("AC0G-T7", _one(tmp_path, 'callsign = "AC0G"'))
    assert "position" in str(e.value) or "grid" in str(e.value)


@pytest.mark.parametrize("lat,lon", [("91", "0"), ("0", "181"), ("nan", "0"), ("0", "inf")])
def test_bad_precise_position_refused(tmp_path, lat, lon):
    with pytest.raises(ConfigError):
        load_instance("AC0G-T7", _one(tmp_path, f"receiver_lat = {lat}\nreceiver_lon = {lon}"))


def test_one_coordinate_alone_refused(tmp_path):
    with pytest.raises(ConfigError):
        load_instance("AC0G-T7", _one(tmp_path, "receiver_lat = 38.0\ngrid_square = \"EM28\""))


def test_missing_reporter_id_refused(tmp_path):
    text = GOOD.replace('reporter_id = "AC0G/T7"', "")
    with pytest.raises(ConfigError, match="reporter_id"):
        load_instance("AC0G-T7", write(tmp_path, text))


def test_unknown_transmitter_named(tmp_path):
    text = GOOD.replace('"AGL1"', '"NOPE9"')
    with pytest.raises(ConfigError, match="NOPE9"):
        load_instance("AC0G-T7", write(tmp_path, text))


def test_duplicate_channel_name_and_all_problems_listed(tmp_path):
    text = GOOD.replace('name = "high"', 'name = "low"').replace('"AGL1"', '"NOPE9"')
    with pytest.raises(ConfigError) as e:
        load_instance("AC0G-T7", write(tmp_path, text))
    msg = str(e.value)
    assert "NOPE9" in msg and "low" in msg and "same name" in msg


def test_tdma_offsets_stay_in_raw(tmp_path):
    _one(tmp_path, 'grid_square = "EM28"')
    from chirp_recorder.core.catalogue import load_seed
    a = next(t for t in load_seed().values() if 4.4e6 < t.centre_hz < 4.6e6)
    extra = f'tdma_offset_samples = {{ {a.id} = 12 }}\n'
    (tmp_path / "AC0G-T7.toml").write_text(
        (tmp_path / "AC0G-T7.toml").read_text().replace(
            "transmitters =", extra + "transmitters ="))
    cfg = load_instance("AC0G-T7", tmp_path)
    assert cfg.raw["channel"][0]["tdma_offset_samples"] == {a.id: 12}


def test_missing_file_is_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_instance("AC0G-T7", tmp_path)
