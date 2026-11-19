"""Helpers shared by test_configurator.py and test_contract.py.  Every
coordinate in this file is made up; none names a real site."""
from __future__ import annotations

import contextlib
import io
import json

import pytest

# Made-up receiver position, written only into tmp files.
COORD_LAT, COORD_LON = 38.25, -80.5

CODAR_KFS_CENTRES = [4_463_000.0, 4_450_500.0]
CODAR_KFS_TDMA = {"ASSA": 21385, "CEDR": 42884, "LISL": 175}
CODAR_KFS_PROCESSING = {"sample_rate_hz": 64000, "coherent_seconds": 90, "range_min_km": 150,
                        "range_max_km": 700, "snr_threshold_db": 6.5,
                        "max_peaks_per_cpi": 3, "clutter_window": 10}

CODAR_KFS_TOML = f'''\
[instance]
reporter_id = "KFS/MAIN"

[station]
callsign     = "KFS"
grid_square  = "EM38ww"
receiver_lat = {COORD_LAT}
receiver_lon = {COORD_LON}

[paths]
output_dir = "/var/lib/codar-sounder"
log_dir    = "/var/log/codar-sounder"

[[radiod]]
status       = "kfs-omni-status.local"
channel_name = "codar-4463khz"

[[radiod.transmitter]]
id                  = "ASSA"
center_freq_hz      = 4463000
sweep_rate_hz_per_s = -25733.913
sweep_bw_hz         = 25734
sweep_repetition_hz = 1
tx_lat_deg          = 38.205017
tx_lon_deg          = -75.1529
tdma_offset_samples = 21385

[[radiod.transmitter]]
id                  = "CEDR"
center_freq_hz      = 4463000
sweep_rate_hz_per_s = -25733.913
sweep_bw_hz         = 25734
sweep_repetition_hz = 1
tx_lat_deg          = 37.6733
tx_lon_deg          = -75.593033
tdma_offset_samples = 42884

[[radiod.transmitter]]
id                  = "LISL"
center_freq_hz      = 4463000
sweep_rate_hz_per_s = -25733.913
sweep_bw_hz         = 25734
sweep_repetition_hz = 1
tx_lat_deg          = 36.691717
tx_lon_deg          = -75.922633
tdma_offset_samples = 175

[[radiod]]
status       = "kfs-omni-status.local"
channel_name = "codar-4450khz"

[[radiod.transmitter]]
id                  = "OBSP"
center_freq_hz      = 4450500
sweep_rate_hz_per_s = -25733.913
sweep_bw_hz         = 25734
sweep_repetition_hz = 1
tx_lat_deg          = 30.249433
tx_lon_deg          = -87.66845

[processing]
sample_rate_hz   = 64000
coherent_seconds = 90
range_min_km     = 150
range_max_km     = 700
snr_threshold_db = 6.5
max_peaks_per_cpi = 3
clutter_window   = 10
'''


def _run(argv, etc):
    out, err = io.StringIO(), io.StringIO()
    from chirp_recorder.cli import main
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = main(["--etc", str(etc), *argv])
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 1
    return rc, out.getvalue(), err.getvalue()


@pytest.fixture(autouse=True)
def isolated_station(tmp_path, monkeypatch):
    """No host coordination, env bag or timing env leaks into a test."""
    from chirp_recorder import configurator
    from chirp_recorder.core import timing
    monkeypatch.setattr(configurator, "COORDINATION_ENV", tmp_path / "none.env")
    monkeypatch.setattr(configurator, "COORDINATION_TOML", tmp_path / "none.toml")
    monkeypatch.setattr(timing, "COORDINATION_ENV", tmp_path / "none-timing.env")
    for v in ("STATION_LAT", "STATION_LON", "STATION_GRID", "STATION_CALL",
              "STATION_REPORTER_ID", "SIGMOND_INSTANCE", "SIGMOND_RADIOD_STATUS",
              "CHIRP_MAX_CHANNELS"):
        monkeypatch.delenv(v, raising=False)


@pytest.fixture
def tmp_etc(tmp_path):
    d = tmp_path / "etc"
    d.mkdir()
    return d


@pytest.fixture
def run_cli(tmp_etc):
    def go(*argv, etc=None):
        return _run(argv, etc or tmp_etc)[0]
    return go


@pytest.fixture
def run_cli_full(tmp_etc):
    def go(*argv, etc=None):
        return _run(argv, etc or tmp_etc)
    return go


@pytest.fixture
def run_cli_out(tmp_etc):
    def go(*argv, etc=None):
        return _run(argv, etc or tmp_etc)[1]
    return go


@pytest.fixture
def configured(tmp_etc, monkeypatch):
    monkeypatch.setenv("STATION_GRID", "EM38ww")
    rc, _, err = _run(["config", "init", "--non-interactive", "--instance", "AC0G-B4",
                       "--reporter-id", "AC0G/B4", "--radiod-status", "b4-status.local"],
                      tmp_etc)
    assert rc == 0, err
    return tmp_etc


@pytest.fixture
def codar_kfs_file(tmp_path):
    p = tmp_path / "KFS-MAIN.toml"
    p.write_text(CODAR_KFS_TOML)
    return p


def load_json(text):
    return json.loads(text)
