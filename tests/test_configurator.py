"""config init per receiver, --from-codar, and the position rules (pre-flight C28)."""
from __future__ import annotations

import json

from chirp_recorder.config import load_instance
from cli_support import *  # noqa: F401,F403  (fixtures)
from cli_support import (CODAR_KFS_CENTRES, CODAR_KFS_PROCESSING, CODAR_KFS_TDMA,
                         COORD_LAT, COORD_LON)

INIT = ("config", "init", "--non-interactive")


def _init(run_cli, key="AC0G-B4", rid="AC0G/B4", status="b4-status.local", *extra):
    return run_cli(*INIT, "--instance", key, "--reporter-id", rid,
                   "--radiod-status", status, *extra)


def test_two_instances_on_one_radiod(tmp_etc, run_cli, run_cli_out, monkeypatch):
    monkeypatch.setenv("STATION_GRID", "EM38ww")
    for key, rid in (("AC0G-B1A", "AC0G/B1A"), ("AC0G-B1A2", "AC0G/B1A2")):
        assert run_cli(*INIT, "--instance", key, "--reporter-id", rid,
                       "--radiod-status", "b1a-status.local") == 0
    inv = json.loads(run_cli_out("inventory", "--json"))
    assert {i["instance"] for i in inv["instances"]} == {"AC0G-B1A", "AC0G-B1A2"}


def test_refuses_to_guess_radiod(tmp_etc, run_cli, monkeypatch):
    monkeypatch.setenv("STATION_GRID", "EM38ww")
    assert run_cli(*INIT, "--instance", "AC0G-B4", "--reporter-id", "AC0G/B4") != 0
    assert not (tmp_etc / "AC0G-B4.toml").exists()


def test_refuses_without_a_position(tmp_etc, run_cli, run_cli_full):
    rc, _, err = run_cli_full(*INIT, "--instance", "AC0G-B4", "--reporter-id", "AC0G/B4",
                              "--radiod-status", "b4-status.local")
    assert rc != 0 and "position" in err
    assert not (tmp_etc / "AC0G-B4.toml").exists()


def test_refuses_to_guess_reporter(tmp_etc, run_cli, monkeypatch):
    monkeypatch.setenv("STATION_GRID", "EM38ww")
    assert run_cli(*INIT, "--instance", "AC0G-B4", "--radiod-status", "b4-status.local") != 0
    # STATION_REPORTER_ID of another key never stands in
    monkeypatch.setenv("STATION_REPORTER_ID", "AC0G/B1A")
    assert run_cli(*INIT, "--instance", "AC0G-B4", "--radiod-status", "b4-status.local") != 0
    assert run_cli(*INIT, "--instance", "AC0G-B4", "--reporter-id", "AC0G/B1A",
                   "--radiod-status", "b4-status.local") != 0


def test_env_fallbacks(tmp_etc, run_cli, monkeypatch):
    monkeypatch.setenv("SIGMOND_INSTANCE", "AC0G-B4")
    monkeypatch.setenv("STATION_REPORTER_ID", "AC0G/B4")
    monkeypatch.setenv("SIGMOND_RADIOD_STATUS", "b4-status.local")
    monkeypatch.setenv("STATION_GRID", "EM38ww")
    monkeypatch.setenv("STATION_CALL", "AC0G")
    monkeypatch.setenv("CHIRP_MAX_CHANNELS", "3")
    assert run_cli(*INIT) == 0
    cfg = load_instance("AC0G-B4", tmp_etc)
    assert cfg.reporter_id == "AC0G/B4" and cfg.max_channels == 3
    assert len(cfg.channels) == 3
    assert cfg.raw["station"]["callsign"] == "AC0G"


def test_coordination_env_gives_a_precise_position(tmp_etc, run_cli, run_cli_full, monkeypatch):
    monkeypatch.setenv("STATION_LAT", str(COORD_LAT))
    monkeypatch.setenv("STATION_LON", str(COORD_LON))
    monkeypatch.setenv("STATION_GRID", "EM38ww")
    rc, out, err = run_cli_full(*INIT, "--instance", "AC0G-B4", "--reporter-id", "AC0G/B4",
                                "--radiod-status", "b4-status.local")
    assert rc == 0, err
    cfg = load_instance("AC0G-B4", tmp_etc)
    assert cfg.position_source == "precise"
    assert (cfg.receiver_lat, cfg.receiver_lon) == (COORD_LAT, COORD_LON)
    assert str(COORD_LAT) not in out + err and str(COORD_LON) not in out + err
    assert "precise" in out


def test_coordination_env_file_and_toml(tmp_path, tmp_etc, run_cli, monkeypatch):
    from chirp_recorder import configurator
    env = tmp_path / "coordination.env"
    env.write_text(f"STATION_CALL=AC0G\nSTATION_LAT={COORD_LAT}\nSTATION_LON={COORD_LON}\n")
    monkeypatch.setattr(configurator, "COORDINATION_ENV", env)
    assert _init(run_cli) == 0
    assert load_instance("AC0G-B4", tmp_etc).position_source == "precise"
    toml = tmp_path / "coordination.toml"
    toml.write_text(f"[host]\ncall = \"AC0G\"\nlat = {COORD_LAT}\nlon = {COORD_LON}\n")
    monkeypatch.setattr(configurator, "COORDINATION_ENV", tmp_path / "gone.env")
    monkeypatch.setattr(configurator, "COORDINATION_TOML", toml)
    assert _init(run_cli, "AC0G-B1A", "AC0G/B1A") == 0
    assert load_instance("AC0G-B1A", tmp_etc).position_source == "precise"


def test_grid_only_says_so(tmp_etc, run_cli, run_cli_full, monkeypatch):
    monkeypatch.setenv("STATION_GRID", "EM38ww")
    rc, out, err = run_cli_full(*INIT, "--instance", "AC0G-B4", "--reporter-id", "AC0G/B4",
                                "--radiod-status", "b4-status.local")
    assert rc == 0, err
    cfg = load_instance("AC0G-B4", tmp_etc)
    assert cfg.position_source == "grid_centre"
    assert "grid centre" in out and "EM38ww" in out and "withheld" in out


def test_existing_config_stays_without_reconfig(tmp_etc, run_cli, monkeypatch):
    monkeypatch.setenv("STATION_GRID", "EM38ww")
    assert _init(run_cli) == 0
    path = tmp_etc / "AC0G-B4.toml"
    path.write_text(path.read_text() + "\n# operator note\n")
    assert _init(run_cli) == 0
    assert "operator note" in path.read_text()
    assert _init(run_cli, "AC0G-B4", "AC0G/B4", "b4-status.local", "--reconfig") == 0
    assert "operator note" not in path.read_text()


def test_max_channels_caps_the_pick(tmp_etc, run_cli, monkeypatch):
    monkeypatch.setenv("STATION_GRID", "EM38ww")
    assert _init(run_cli, "AC0G-B4", "AC0G/B4", "b4-status.local", "--max-channels", "1") == 0
    cfg = load_instance("AC0G-B4", tmp_etc)
    assert len(cfg.channels) == 1 and cfg.max_channels == 1
    assert cfg.channels[0].status == "b4-status.local"


def test_from_codar_carries_channels_exactly(tmp_etc, run_cli, codar_kfs_file):
    assert run_cli(*INIT, "--instance", "KFS-MAIN", "--reporter-id", "KFS/MAIN",
                   "--radiod-status", "kfs-omni-status.local",
                   "--from-codar", str(codar_kfs_file)) == 0
    cfg = load_instance("KFS-MAIN", tmp_etc)
    assert [c.centre_hz for c in cfg.channels] == CODAR_KFS_CENTRES
    assert [c.status for c in cfg.channels] == ["kfs-omni-status.local"] * 2
    assert [c.sample_rate_hz for c in cfg.channels] == [64000, 64000]
    assert [t for g in cfg.channels[0].groups for t in g.tx_ids] == ["ASSA", "CEDR", "LISL"]
    for k in ("coherent_seconds", "range_min_km", "range_max_km", "snr_threshold_db",
              "max_peaks_per_cpi", "clutter_window"):
        assert cfg.raw["physics"][k] == CODAR_KFS_PROCESSING[k]
    # codar-sounder offsets do not carry over (test_from_codar_drops_tdma_offsets)
    assert all("tdma_offset_samples" not in c for c in cfg.raw["channel"])
    # codar's own receiver position travels with the channels
    assert cfg.position_source == "precise"
    assert (cfg.receiver_lat, cfg.receiver_lon) == (COORD_LAT, COORD_LON)


def test_from_codar_drops_tdma_offsets(tmp_etc, run_cli_full, codar_kfs_file):
    """A codar-sounder offset has the opposite sign and no UTC reference, so no
    conversion rescues it: the import drops it and says so, one line each."""
    rc, _, err = run_cli_full(*INIT, "--instance", "KFS-MAIN", "--reporter-id", "KFS/MAIN",
                              "--from-codar", str(codar_kfs_file))
    assert rc == 0
    cfg = load_instance("KFS-MAIN", tmp_etc)
    assert all("tdma_offset_samples" not in c for c in cfg.raw["channel"])
    lines = [ln for ln in err.splitlines() if "tdma_offset_samples" in ln]
    assert len(lines) == len(CODAR_KFS_TDMA)
    for tx, old in CODAR_KFS_TDMA.items():
        ln = next(ln for ln in lines if f" {tx} " in ln)
        assert str(old) in ln
        assert "opposite sign" in ln and "UTC" in ln
        assert "delay_ns" in ln and "path delay" in ln


def test_from_codar_ignores_a_placeholder_position(tmp_path, tmp_etc, run_cli,
                                                   run_cli_full, codar_kfs_file, monkeypatch):
    text = codar_kfs_file.read_text().replace(f"receiver_lat = {COORD_LAT}", "receiver_lat = 0.0") \
        .replace(f"receiver_lon = {COORD_LON}", "receiver_lon = 0.0")
    codar_kfs_file.write_text(text)
    rc, out, err = run_cli_full(*INIT, "--instance", "KFS-MAIN", "--reporter-id", "KFS/MAIN",
                                "--from-codar", str(codar_kfs_file))
    assert rc == 0, err
    cfg = load_instance("KFS-MAIN", tmp_etc)
    assert cfg.position_source == "grid_centre" and cfg.grid == "EM38ww"
    assert "grid centre" in out


def test_from_codar_refuses_a_block_the_channel_cannot_hear(tmp_etc, run_cli_full,
                                                            codar_kfs_file):
    # ACI1 sweeps near 13.475 MHz: a channel tuned to 4.4505 MHz cannot hear it.
    codar_kfs_file.write_text(codar_kfs_file.read_text().replace(
        "[processing]", """[[radiod.transmitter]]
id             = "ACI1"
center_freq_hz = 4450500
sweep_rate_hz_per_s = 95583.106
sweep_bw_hz    = 47792
sweep_repetition_hz = 2
tx_lat_deg     = 34.0158
tx_lon_deg     = -119.359333

[processing]""").replace("center_freq_hz = 4450500", "center_freq_hz = 13475000"))
    rc, _, err = run_cli_full(*INIT, "--instance", "KFS-MAIN", "--reporter-id", "KFS/MAIN",
                              "--from-codar", str(codar_kfs_file))
    assert rc != 0 and "codar-4450khz" in err and "ACI1" in err
    assert not (tmp_etc / "KFS-MAIN.toml").exists()


def test_from_codar_refuses_an_unknown_transmitter_naming_the_block(tmp_etc, run_cli_full,
                                                                    codar_kfs_file):
    codar_kfs_file.write_text(codar_kfs_file.read_text().replace('"OBSP"', '"ZZZZ"'))
    rc, _, err = run_cli_full(*INIT, "--instance", "KFS-MAIN", "--reporter-id", "KFS/MAIN",
                              "--from-codar", str(codar_kfs_file))
    assert rc != 0 and "codar-4450khz" in err and "ZZZZ" in err


def test_from_codar_missing_file_refuses(tmp_path, run_cli):
    assert run_cli(*INIT, "--instance", "KFS-MAIN", "--reporter-id", "KFS/MAIN",
                   "--from-codar", str(tmp_path / "nope.toml")) != 0


def test_interactive_init_prompts(tmp_etc, run_cli, monkeypatch):
    from chirp_recorder import configurator
    monkeypatch.setenv("STATION_GRID", "EM38ww")
    monkeypatch.setattr(configurator, "_wizard_available", lambda args=None: False)
    answers = iter(["AC0G/B4", "b4-status.local", "1"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert run_cli("config", "init", "--instance", "AC0G-B4") == 0
    cfg = load_instance("AC0G-B4", tmp_etc)
    assert cfg.reporter_id == "AC0G/B4" and cfg.max_channels == 1


def test_config_edit_prints_in_non_interactive(tmp_etc, configured, run_cli_out):
    out = run_cli_out("config", "edit", "--non-interactive", "--instance", "AC0G-B4")
    assert "[instance]" in out and "AC0G/B4" in out


# --- fix round 1 -------------------------------------------------------------

def test_from_codar_prefers_coordination_over_a_stale_file(tmp_etc, run_cli_full,
                                                           codar_kfs_file, monkeypatch):
    monkeypatch.setenv("STATION_LAT", "41.5")        # made up; differs from the file's
    monkeypatch.setenv("STATION_LON", "-90.25")
    rc, out, err = run_cli_full(*INIT, "--instance", "KFS-MAIN", "--reporter-id", "KFS/MAIN",
                                "--from-codar", str(codar_kfs_file))
    assert rc == 0, err
    cfg = load_instance("KFS-MAIN", tmp_etc)
    assert (cfg.receiver_lat, cfg.receiver_lon) == (41.5, -90.25)
    assert "sigmond coordination" in out and "41.5" not in out


def test_from_codar_names_the_file_when_coordination_is_empty(tmp_etc, run_cli_full,
                                                              codar_kfs_file):
    rc, out, _ = run_cli_full(*INIT, "--instance", "KFS-MAIN", "--reporter-id", "KFS/MAIN",
                              "--from-codar", str(codar_kfs_file))
    assert rc == 0 and "codar-sounder file" in out


def test_config_is_not_world_readable(tmp_etc, run_cli, monkeypatch):
    import grp
    monkeypatch.setenv("STATION_LAT", str(COORD_LAT))
    monkeypatch.setenv("STATION_LON", str(COORD_LON))
    monkeypatch.setattr(grp, "getgrnam", lambda n: (_ for _ in ()).throw(KeyError(n)))
    assert _init(run_cli) == 0
    assert (tmp_etc / "AC0G-B4.toml").stat().st_mode & 0o777 == 0o600
    assert not list(tmp_etc.glob("*.part"))


def test_config_group_readable_when_chirprec_exists(tmp_etc, run_cli, monkeypatch):
    import grp
    import os
    from types import SimpleNamespace
    monkeypatch.setenv("STATION_GRID", "EM38ww")
    monkeypatch.setattr(grp, "getgrnam", lambda n: SimpleNamespace(gr_gid=os.getgid()))
    assert _init(run_cli) == 0
    assert (tmp_etc / "AC0G-B4.toml").stat().st_mode & 0o777 == 0o640


def test_part_file_removed_on_failure(tmp_etc, run_cli, monkeypatch):
    from pathlib import Path
    monkeypatch.setenv("STATION_GRID", "EM38ww")
    real = Path.replace

    def boom(self, target):
        if str(self).endswith(".part"):
            raise OSError("disk full")
        return real(self, target)
    monkeypatch.setattr(Path, "replace", boom)
    try:
        _init(run_cli)
    except OSError:
        pass
    assert not list(tmp_etc.glob("*.part"))


def test_from_codar_warns_when_radiod_status_is_ignored(tmp_etc, run_cli_full, codar_kfs_file):
    rc, _, err = run_cli_full(*INIT, "--instance", "KFS-MAIN", "--reporter-id", "KFS/MAIN",
                              "--radiod-status", "other-status.local",
                              "--from-codar", str(codar_kfs_file))
    assert rc == 0 and "ignored" in err
    assert load_instance("KFS-MAIN", tmp_etc).channels[0].status == "kfs-omni-status.local"


def test_from_codar_block_centre_key_beats_first_transmitter(tmp_etc, run_cli, codar_kfs_file):
    codar_kfs_file.write_text(codar_kfs_file.read_text().replace(
        'channel_name = "codar-4463khz"', 'channel_name = "codar-4463khz"\ncenter_freq_hz = 4463500'))
    assert run_cli(*INIT, "--instance", "KFS-MAIN", "--reporter-id", "KFS/MAIN",
                   "--from-codar", str(codar_kfs_file)) == 0
    assert load_instance("KFS-MAIN", tmp_etc).channels[0].centre_hz == 4_463_500.0
