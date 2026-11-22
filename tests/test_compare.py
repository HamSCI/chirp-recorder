"""The side-by-side comparison: chirp-recorder's T6 clusters against
codar-sounder's, as the retirement check reads them (decision 3, 2026-11-19)."""
from __future__ import annotations

import json
import math

from hamsci_dsp import sigmondb_rows as sr

from chirp_recorder.core.compare import (Cluster, Verdict, clusters, compare,
                                         read_sink_rows)

G = sr.group_key(4.463e6, -25_733.913)
REPORTER = "AC0G/B4"
SCATTER_NS = (-300, -180, -60, 60, 180, 300)       # six arrivals per delay, +-300 ns


def rows(producer, delays, *, doppler_hz=0.1, s4=0.2, sigma_phi=0.05, t6=True):
    """Six T6 arrivals per delay, one per capture, for one reporter and group G."""
    prov = sr.provenance("T6", "epoch-1") if t6 else sr.provenance(None)
    out = []
    for d in delays:
        for i, s in enumerate(SCATTER_NS):
            out.append(sr.arrival_row(
                time=f"2026-11-19T{1 + i:02d}:00:00Z", reporter_id=REPORTER, group_key=G,
                centre_hz=4.463e6, rate_hz_per_s=-25_733.913, delay_ns=d + s,
                level_db=30.0, doppler_hz=doppler_hz, s4=s4, sigma_phi_rad=sigma_phi,
                detector_version=f"{producer}/test", **prov))
    return out


def make_sink(tmp_path, *, producer, rows):
    """A sink.db written through hs_uploader's own Writer, so the reader meets
    the real schema."""
    from hs_uploader.sink.writer import SqliteConfig, Writer
    db = tmp_path / "sink.db"
    w = Writer("sigmondb", "arrival", schema_version=sr.SCHEMA_VERSION, producer=producer,
               local=True, config=SqliteConfig(path=str(db)))
    w.insert(rows)
    w.close()
    return db


def test_matching_clusters_pass():
    v = compare(rows("chirp", [4_120_000, 42_906_000, 50_939_000]),
                rows("codar", [4_121_500, 42_904_000, 50_940_200]), rung_s={G: 19.973e-3})
    assert v.status == "pass" and len(v.matched) == 3


def test_rungs_and_kfs_twins_are_explained_not_missed():
    v = compare(rows("chirp", [4_120_000, 42_906_000, 50_939_000]),
                rows("codar", [4_120_000, 42_906_000, 50_939_000, 42_906_000 + 19_973_000,
                               42_906_000 + 1_632_000]), rung_s={G: 19.973e-3})
    assert v.status == "pass" and len(v.explained) == 2


def test_a_6_us_shift_fails():
    v = compare(rows("chirp", [4_120_000, 42_906_000, 50_939_000]),
                rows("codar", [4_126_000, 42_912_000, 50_945_000]), rung_s={G: 19.973e-3})
    assert v.status == "fail"


def test_doppler_off_by_more_than_0_02_hz_fails():
    v = compare(rows("chirp", [4_120_000, 42_906_000, 50_939_000], doppler_hz=0.10),
                rows("codar", [4_120_000, 42_906_000, 50_939_000], doppler_hz=0.13),
                rung_s={G: 19.973e-3})
    assert v.status == "fail" and next(r for r in v.rows if r.name == "doppler").passed is False


def test_s4_and_sigma_phi_report_but_never_gate():
    v = compare(rows("chirp", [4_120_000, 42_906_000, 50_939_000], s4=0.1),
                rows("codar", [4_120_000, 42_906_000, 50_939_000], s4=0.6), rung_s={G: 19.973e-3})
    assert v.status == "pass" and next(r for r in v.rows if r.name == "s4").gating is False


def test_too_few_t6_clusters_is_insufficient():
    assert compare(rows("chirp", [4_120_000]), rows("codar", [4_120_000]),
                   rung_s={G: 0.02}).status == "insufficient"


def test_reads_rows_from_a_real_sink(tmp_path):
    db = make_sink(tmp_path, producer="chirp-recorder", rows=rows("chirp", [4_120_000]))
    assert len(read_sink_rows(db, producer="chirp-recorder", reporter_id="AC0G/B4",
                              since_iso="2026-11-19T00:00:00Z",
                              until_iso="2026-11-20T00:00:00Z")) == 6


# --- beyond the brief's seven ------------------------------------------------

def test_the_sink_reader_keeps_to_its_producer_reporter_and_window(tmp_path):
    from hs_uploader.sink.writer import SqliteConfig, Writer
    db = make_sink(tmp_path, producer="chirp-recorder", rows=rows("chirp", [4_120_000]))
    w = Writer("sigmondb", "arrival", schema_version=1, producer="codar-sounder",
               local=True, config=SqliteConfig(path=str(db)))
    w.insert(rows("codar", [9_000_000]))
    w.close()
    get = lambda **kw: read_sink_rows(db, **{**dict(                      # noqa: E731
        producer="chirp-recorder", reporter_id="AC0G/B4",
        since_iso="2026-11-19T00:00:00Z", until_iso="2026-11-20T00:00:00Z"), **kw})
    assert {r["delay_ns"] // 1000 for r in get(producer="codar-sounder")} == {8999, 9000}
    assert get(reporter_id="AC0G/B1A") == []
    assert len(get(since_iso="2026-11-19T03:30:00Z")) == 3           # 04Z, 05Z, 06Z
    assert len(get(until_iso="2026-11-19T03:00:00Z")) == 2           # until is exclusive


def test_clusters_take_t6_rows_only():
    assert clusters(rows("chirp", [4_120_000], t6=False), producer="chirp-recorder") == []
    (c,) = clusters(rows("chirp", [4_120_000]), producer="chirp-recorder")
    assert c == Cluster(reporter_id=REPORTER, group_key=G, producer="chirp-recorder",
                        median_delay_ns=4_120_000, n=6, median_doppler_hz=0.1,
                        median_s4=0.2, median_sigma_phi_rad=0.05)


def test_clusters_split_at_the_gap_and_need_min_n_captures():
    r = rows("chirp", [4_120_000, 4_120_000 + 121_000])
    assert [c.median_delay_ns for c in clusters(r, producer="x")] == [4_120_000, 4_241_000]
    assert len(clusters(rows("chirp", [4_120_000, 4_120_000 + 100_000]), producer="x")) == 1
    assert clusters(r[:5], producer="x") == []               # five captures, not six


def test_a_lone_rung_is_not_explained_by_its_own_weaker_twin():
    """Explanation runs from the stronger cluster to the weaker only; a missed
    cluster with nothing stronger beside it stays missed."""
    chirp = rows("chirp", [4_120_000, 42_906_000, 50_939_000])
    codar = rows("codar", [4_120_000, 42_906_000, 50_939_000]) + \
        rows("codar", [60_000_000])[:6]
    v = compare(chirp, codar, rung_s={G: 19.973e-3})
    assert [c.median_delay_ns for c in v.codar_unmatched] == [60_000_000]
    assert next(r for r in v.rows if r.name == "coverage").passed is False   # 3 of 4
    assert v.status == "fail"


def test_verdict_to_dict_is_json_and_names_every_row():
    v = compare(rows("chirp", [4_120_000, 42_906_000, 50_939_000]),
                rows("codar", [4_121_500, 42_904_000, 50_940_200, 42_906_000 + 1_632_000]),
                rung_s={G: 19.973e-3})
    d = json.loads(json.dumps(v.to_dict(), allow_nan=False))
    assert d["status"] == "pass"
    assert {r["name"] for r in d["rows"]} >= {"matched", "coverage", "doppler", "s4", "sigma_phi"}
    assert {r["name"]: r["gating"] for r in d["rows"]}["sigma_phi"] is False
    assert [m[2] for m in d["matched"]] == [-1_500, 2_000, -1_200]
    assert d["matched"][0][3] == G
    assert d["explained"] == [[42_906_000 + 1_632_000, "KFS twin of 42.904 ms"]]
    assert d["codar_unmatched"] == [] and d["chirp_only"] == []


def test_verdict_rows_hold_numbers():
    v = compare(rows("chirp", [4_120_000, 42_906_000, 50_939_000], s4=0.1),
                rows("codar", [4_121_500, 42_904_000, 50_940_200], s4=0.6), rung_s={G: 0.02})
    by = {r.name: r for r in v.rows}
    assert by["matched"].value == 2_000 and by["matched"].limit == 5_000
    assert by["coverage"].value == 1.0 and by["coverage"].limit == 0.9
    assert math.isclose(by["s4"].value, 0.5) and by["s4"].passed is True
    assert isinstance(v, Verdict)


# --- the compare subcommand --------------------------------------------------
from cli_support import configured, isolated_station, run_cli_full, tmp_etc  # noqa: E402,F401


def _sink_with_both(tmp_path, chirp, codar):
    from hs_uploader.sink.writer import SqliteConfig, Writer
    db = tmp_path / "sink.db"
    for producer, rs in (("chirp-recorder", chirp), ("codar-sounder", codar)):
        w = Writer("sigmondb", "arrival", schema_version=1, producer=producer, local=True,
                   config=SqliteConfig(path=str(db)))
        w.insert(rs)
        w.close()
    return db


def test_compare_subcommand_reads_the_stations_sink(tmp_path, monkeypatch, configured,
                                                    run_cli_full):
    db = _sink_with_both(tmp_path, rows("chirp", [4_120_000, 42_906_000, 50_939_000]),
                         rows("codar", [4_121_500, 42_904_000, 50_940_200]))
    monkeypatch.setenv("HAMSCI_SINK_PATH", str(db))
    rc, out, err = run_cli_full("compare", "--instance", "AC0G-B4",
                                "--since", "2026-11-19T00:00:00Z",
                                "--until", "2026-11-20T00:00:00Z", "--json")
    d = json.loads(out)
    assert rc == 0, err
    assert d["status"] == "pass" and len(d["matched"]) == 3
    assert {r["name"] for r in d["rows"]} >= {"matched", "coverage", "doppler"}


def test_compare_subcommand_exits_1_unless_it_passes(tmp_path, monkeypatch, configured,
                                                     run_cli_full):
    db = _sink_with_both(tmp_path, rows("chirp", [4_120_000]), rows("codar", [4_120_000]))
    monkeypatch.setenv("HAMSCI_SINK_PATH", str(db))
    rc, out, _ = run_cli_full("compare", "--instance", "AC0G-B4",
                              "--since", "2026-11-19T00:00:00Z", "--json")
    assert rc == 1 and json.loads(out)["status"] == "insufficient"


def test_compare_subcommand_without_a_sink_says_so(tmp_path, monkeypatch, configured,
                                                   run_cli_full):
    monkeypatch.setenv("HAMSCI_SINK_PATH", str(tmp_path / "absent.db"))
    rc, out, err = run_cli_full("compare", "--instance", "AC0G-B4",
                                "--since", "2026-11-19T00:00:00Z", "--json")
    assert rc == 2 and out == "" and "absent.db" in err
