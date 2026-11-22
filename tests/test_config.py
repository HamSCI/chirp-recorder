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


def test_non_table_sections_join_the_problem_list(tmp_path):
    (tmp_path / "AC0G-T7.toml").write_text(
        'station = 3\nchannel = [1]\n[instance]\nreporter_id = "AC0G/T7"\n')
    with pytest.raises(ConfigError) as e:
        load_instance("AC0G-T7", tmp_path)
    assert "[station] must be a table" in str(e.value) and "must be a table" in str(e.value).split("[station]")[1]


def _picker(tmp_path, body):
    return _one(tmp_path, "receiver_lat = 38.0\nreceiver_lon = -95.0",
                "[picker]\n" + body)


def test_picker_reach_km_loads(tmp_path):
    cfg = load_instance("AC0G-T7", _picker(
        tmp_path, "reach_km = [[3e6, 6e6, 500.0], [6e6, 60e6, 100.0]]"))
    assert cfg.reach_km == ((3e6, 6e6, 500.0), (6e6, 60e6, 100.0))
    assert cfg.reach_source == "config"


def test_picker_reach_km_defaults(tmp_path):
    from chirp_recorder.core.picker import DEFAULT_REACH_KM
    cfg = load_instance("AC0G-T7", _picker(tmp_path, "phase = \"spread\""))
    assert cfg.reach_km == DEFAULT_REACH_KM and cfg.reach_source == "default"


def test_picker_reach_km_inverted_row_refused(tmp_path):
    with pytest.raises(ConfigError, match="reach_km"):
        load_instance("AC0G-T7", _picker(tmp_path, "reach_km = [[6e6, 3e6, 500.0]]"))


def test_physics_defaults_and_settings_for_a_channel(tmp_path):
    from chirp_recorder.config import physics_settings
    _one(tmp_path, 'grid_square = "EM28"')
    from chirp_recorder.core.catalogue import load_seed
    a = next(t for t in load_seed().values() if 4.4e6 < t.centre_hz < 4.6e6)
    p = tmp_path / "AC0G-T7.toml"
    p.write_text(p.read_text().replace(
        "transmitters =", f"tdma_offset_samples = {{ {a.id} = 12 }}\ntransmitters ="))
    cfg = load_instance("AC0G-T7", tmp_path)
    assert cfg.physics == {"enabled": True, "coherent_seconds": 60, "range_min_km": 200,
                           "range_max_km": 800, "snr_threshold_db": 10.0,
                           "max_peaks_per_cpi": 4, "clutter_window": 20}
    s = physics_settings(cfg, cfg.channels[0])
    assert s["tdma_offset_samples"] == {a.id: 12} and s["interval_s"] == 30


def test_physics_coherent_seconds_must_be_whole_intervals(tmp_path):
    _one(tmp_path, 'grid_square = "EM28"', "")
    p = tmp_path / "AC0G-T7.toml"
    p.write_text("[physics]\ncoherent_seconds = 45\n" + p.read_text())
    with pytest.raises(ConfigError, match="whole number"):
        load_instance("AC0G-T7", tmp_path)


def test_physics_unknown_key_and_bad_tdma_listed(tmp_path):
    _one(tmp_path, 'grid_square = "EM28"')
    p = tmp_path / "AC0G-T7.toml"
    p.write_text("[physics]\nbogus = 1\n" + p.read_text().replace(
        "transmitters =", "tdma_offset_samples = { NOPE = 1 }\ntransmitters ="))
    with pytest.raises(ConfigError) as e:
        load_instance("AC0G-T7", tmp_path)
    assert "bogus" in str(e.value) and "tdma_offset_samples" in str(e.value)
