import importlib.util
from pathlib import Path

import pytest

from chirp_recorder.config import grid_centre
from chirp_recorder.core.catalogue import Transmitter
from chirp_recorder.core.channels import form_channels
from chirp_recorder.core.picker import (T6Station, VendoredMenu, load_t6_stations,
                                        rank, reach_km)

# Made-up coordinates throughout.
NJ = Transmitter("NJ", "codar", 4.463e6, -25_733.913, 1.0, 25_734, 39.5, -74.3)
CA = Transmitter("CA", "codar", 4.80e6, -25_733.913, 1.0, 25_734, 34.4, -119.9)
TXS = {"NJ": NJ, "CA": CA}
T6 = [T6Station("K3LR", 41.4, -80.4), T6Station("W2NAF", 40.3, -74.6),
      T6Station("ME", 38.0, -95.0)]
PLANS = form_channels([NJ], "s") + form_channels([CA], "s")


def test_establish_puts_shared_channel_first_even_when_farther():
    r = rank(PLANS, TXS, receiver_lat=38.0, receiver_lon=-95.0, self_reporter="ME", t6=T6)
    assert r[0].plan.groups[0].tx_ids == ("NJ",) and r[0].shared_t6 == 2
    assert r[0].reason.startswith("2 other T6 stations")


def test_spread_reverses_the_preference():
    r = rank(PLANS, TXS, receiver_lat=38.0, receiver_lon=-95.0, self_reporter="ME",
             t6=T6, phase="spread")
    assert r[0].plan.groups[0].tx_ids == ("CA",)


def test_bad_phase_raises():
    with pytest.raises(ValueError):
        rank(PLANS, TXS, receiver_lat=38.0, receiver_lon=-95.0, self_reporter="ME",
             t6=T6, phase="sideways")


def test_self_never_counts():
    r = rank(PLANS, TXS, receiver_lat=40.3, receiver_lon=-74.6, self_reporter="W2NAF", t6=T6)
    assert next(c for c in r if c.plan.groups[0].tx_ids == ("NJ",)).shared_t6 == 1


def test_reach_by_band():
    assert reach_km(4.463e6) == 900.0 and reach_km(13.45e6) == 300.0 and reach_km(25e6) == 150.0


def test_config_reach_changes_the_count_and_is_logged(caplog):
    caplog.set_level("INFO")
    short = ((3e6, 6e6, 100.0), (6e6, 15e6, 300.0), (15e6, 60e6, 150.0))
    r = rank(PLANS, TXS, receiver_lat=38.0, receiver_lon=-95.0, self_reporter="ME", t6=T6,
             reach_table=short, reach_source="config")
    assert next(c for c in r if c.plan.groups[0].tx_ids == ("NJ",)).shared_t6 == 1
    assert "reach 100 km at 4.463 MHz (config)" in caplog.text


def test_reason_names_transmitters_heard():
    r = rank(PLANS, TXS, receiver_lat=38.0, receiver_lon=-95.0, self_reporter="ME", t6=T6)
    assert r[0].reason == "2 other T6 stations hear NJ"


def test_load_t6_stations(tmp_path):
    p = tmp_path / "t6.toml"
    p.write_text('as_of = "2026-11-19"\n[stations."K3LR"]\ngrid = "EN91"\nlat = 41.5\nlon = -81.0\n')
    s = load_t6_stations(p)
    assert s == [T6Station("K3LR", 41.5, -81.0)]


def test_shipped_t6_file_loads():
    for s in load_t6_stations():
        assert -90 <= s.lat <= 90 and -180 <= s.lon <= 180


def test_vendored_menu_limits_to_4000_km_and_ranks():
    menu = VendoredMenu(seed=TXS, t6=T6)
    ch = menu.choices(receiver_lat=38.0, receiver_lon=-95.0, reporter_id="ME",
                      status="s", hour_utc=0)
    ids = [c.plan.groups[0].tx_ids[0] for c in ch]
    assert ids == ["NJ", "CA"][:len(ids)] and "NJ" in ids
    # CA at ~2000 km lies within 4000 km; a receiver in Europe hears neither
    far = menu.choices(receiver_lat=50.0, receiver_lon=10.0, reporter_id="X",
                       status="s", hour_utc=0)
    assert far == []


def _load_script():
    p = Path(__file__).resolve().parents[1] / "scripts" / "rebuild-t6-stations.py"
    spec = importlib.util.spec_from_file_location("rebuild_t6", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_writer_row_is_grid6_centre_never_the_precise_position():
    m = _load_script()
    precise = (41.4437, -81.6213)
    row = m.build_row("K3LR", *precise)
    assert row["grid"] == "EN91ek" or len(row["grid"]) == 6
    assert (row["lat"], row["lon"]) == grid_centre(row["grid"])
    assert (row["lat"], row["lon"]) != precise
    # the centre lies inside the grid6 square (1/24 deg lon, 1/48... half cell)
    assert abs(row["lat"] - precise[0]) <= 1 / 48 + 1e-9
    assert abs(row["lon"] - precise[1]) <= 1 / 24 + 1e-9


def test_writer_prefers_gpsdo_over_typed_grid_and_emits_no_precise_digits():
    m = _load_script()
    rows = [("K3LR", "typed_grid", 41.0, -81.0), ("K3LR", "gpsdo", 41.4437, -81.6213),
            ("W2NAF", "typed_grid", 40.3, -74.6)]
    text = m.render(m.pick_positions(rows), as_of="2026-11-19")
    assert "41.4437" not in text and "-81.6213" not in text
    assert text.count("[stations.") == 2
