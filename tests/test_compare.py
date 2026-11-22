"""The side-by-side comparison: chirp-recorder's T6 clusters against
codar-sounder's, as the retirement check reads them (decision 3, 2026-11-19)."""
from __future__ import annotations

import json
import math

from hamsci_dsp import sigmondb_rows as sr

from chirp_recorder.core.compare import (Cluster, Verdict, clusters, codar_bin_ns, compare,
                                         read_sink_rows)

G = sr.group_key(4.463e6, -25_733.913)
REPORTER = "AC0G/B4"
SCATTER_NS = (-300, -180, -60, 60, 180, 300)       # six arrivals per delay, +-300 ns


K = -25_733.913


def rows(producer, delays, *, doppler_hz=0.1, s4=0.2, sigma_phi=0.05, t6=True):
    """Six T6 arrivals per true delay, one per capture, for one reporter and group G,
    each written in its producer's convention.  chirp-recorder's delay_ns holds
    the true delay (detect.py removes the Doppler); codar-sounder's holds
    tau - f/k, because its beat tone sits at -k*tau + f (sigmondb.py:92-93)."""
    prov = sr.provenance("T6", "epoch-1") if t6 else sr.provenance(None)
    shift = round(-doppler_hz / K * 1e9) if (producer == "codar" and doppler_hz) else 0
    out = []
    for d in delays:
        for i, s in enumerate(SCATTER_NS):
            out.append(sr.arrival_row(
                time=f"2026-11-19T{1 + i:02d}:00:00Z", reporter_id=REPORTER, group_key=G,
                centre_hz=4.463e6, rate_hz_per_s=K, delay_ns=d + s + shift,
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
                rows("codar", [4_126_000, 42_912_000, 50_945_000]), rung_s={G: 19.973e-3},
                match="exact")
    assert v.status == "fail"


# --- fix round 1: C1 (one convention), C2 (codar's range bin), M3 ------------

def test_a_0_3_hz_pair_measured_correctly_on_both_sides_matches():
    """|f| = 0.3 Hz puts codar-sounder's delay 11.66 us from the true one
    (tau - f/k).  Both clients measured the same paths correctly, so compare
    must set them side by side on one convention and match them exactly."""
    for f in (0.3, -0.3):
        v = compare(rows("chirp", [4_120_000, 42_906_000, 50_939_000], doppler_hz=f),
                    rows("codar", [4_120_000, 42_906_000, 50_939_000], doppler_hz=f),
                    rung_s={G: 19.973e-3}, match="exact")
        assert v.status == "pass", v.to_dict()
        assert [abs(d) for _, _, d in v.matched] == [0, 0, 0]


def test_chirp_rows_without_a_doppler_stay_out_of_the_match():
    chirp = rows("chirp", [4_120_000, 42_906_000, 50_939_000])
    for r in chirp[:6]:
        r["doppler_hz"] = None
    v = compare(chirp, rows("codar", [4_120_000, 42_906_000, 50_939_000]),
                rung_s={G: 19.973e-3})
    assert [c.median_delay_ns for c in v.codar_unmatched] == [4_120_000 + round(-0.1 / K * 1e9)]


def test_codar_bin_comes_from_codar_sounders_own_range_axis():
    import numpy as np
    from codar_sounder.core.dechirp import C_KM_PER_S, dechirp
    x = (np.random.default_rng(1).standard_normal(4 * 64_000) + 0j).astype(np.complex64)
    r = dechirp(x, sample_rate_hz=64_000.0, sweep_rate_hz_per_s=K, sweep_repetition_hz=1.0)
    step_ns = (r.range_axis_km[1] - r.range_axis_km[0]) / C_KM_PER_S * 1e9
    assert abs(codar_bin_ns(K, 1.0) - step_ns) < 1e-6
    assert round(codar_bin_ns(K, 1.0)) == 38_859
    assert round(codar_bin_ns(99_259.376, 0.5)) == 20_149


def test_bin_mode_is_the_default_and_allows_half_a_codar_bin_plus_tol():
    true = [4_120_000, 42_906_000, 50_939_000]
    half = codar_bin_ns(K, 1.0) / 2 + 5_000            # 24.4 us
    inside = compare(rows("chirp", true), rows("codar", [d + 24_000 for d in true]),
                     rung_s={G: 19.973e-3})
    outside = compare(rows("chirp", true), rows("codar", [d + 25_000 for d in true]),
                      rung_s={G: 19.973e-3})
    assert inside.status == "pass" and outside.status == "fail"
    row = next(r for r in inside.rows if r.name == "matched")
    assert math.isclose(row.limit, half) and row.value == 24_000
    for v in (inside, outside):
        d = v.to_dict()
        assert d["match"] == "bin" and round(d["codar_bin_ns"][G]) == 38_859


def test_exact_mode_keeps_the_5_us_rule_and_says_so():
    true = [4_120_000, 42_906_000, 50_939_000]
    v = compare(rows("chirp", true), rows("codar", [d + 6_000 for d in true]),
                rung_s={G: 19.973e-3}, match="exact")
    assert v.status == "fail" and v.to_dict()["match"] == "exact"
    assert next(r for r in v.rows if r.name == "matched").limit == 5_000
    assert compare(rows("chirp", true), rows("codar", [d + 4_000 for d in true]),
                   rung_s={G: 19.973e-3}, match="exact").status == "pass"


def test_an_unknown_match_mode_is_refused():
    import pytest
    with pytest.raises(ValueError, match="match"):
        compare([], [], rung_s={}, match="loose")


def test_no_doppler_on_the_codar_side_is_insufficient_not_fail():
    codar = rows("codar", [4_120_000, 42_906_000, 50_939_000], doppler_hz=0.0)
    for r in codar:
        r["doppler_hz"] = None
    v = compare(rows("chirp", [4_120_000, 42_906_000, 50_939_000], doppler_hz=0.0), codar,
                rung_s={G: 19.973e-3})
    assert v.status == "insufficient"


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
    # codar rows hold tau - f/k: 9 ms + 3.886 us
    assert {r["delay_ns"] // 1000 for r in get(producer="codar-sounder")} == {9003, 9004}
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
    assert [c.median_delay_ns for c in v.codar_unmatched] == [60_000_000 + 3_886]   # tau - f/k
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
    # codar medians hold tau - f/k (+3.886 us at 0.1 Hz)
    assert d["explained"] == [[42_906_000 + 1_632_000 + 3_886, "KFS twin of 42.908 ms"]]
    assert d["codar_unmatched"] == [] and d["chirp_only"] == []


def test_verdict_rows_hold_numbers():
    v = compare(rows("chirp", [4_120_000, 42_906_000, 50_939_000], s4=0.1),
                rows("codar", [4_121_500, 42_904_000, 50_940_200], s4=0.6), rung_s={G: 0.02},
                match="exact")
    by = {r.name: r for r in v.rows}
    assert by["matched"].value == 2_000 and by["matched"].limit == 5_000
    assert by["coverage"].value == 1.0 and by["coverage"].limit == 0.9
    assert math.isclose(by["s4"].value, 0.5) and by["s4"].passed is True
    assert isinstance(v, Verdict)


# --- the compare subcommand --------------------------------------------------
from cli_support import configured, isolated_station, run_cli_full, tmp_etc  # noqa: E402,F401


def _sink_with_both(tmp_path, chirp, codar, etc=None):
    """codar-sounder's rows in the sink's queue; chirp-recorder's in its own
    daily JSONL under the instance's output_dir (the copy that stays)."""
    from hs_uploader.sink.writer import SqliteConfig, Writer
    db = tmp_path / "sink.db"
    w = Writer("sigmondb", "arrival", schema_version=1, producer="codar-sounder", local=True,
               config=SqliteConfig(path=str(db)))
    w.insert(codar)
    w.close()
    if etc is not None:
        out = tmp_path / "chirp-data"
        cfg = etc / "AC0G-B4.toml"
        text = cfg.read_text()
        line = f'output_dir = "{out}"\n'
        text = text.replace("[paths]\n", "[paths]\n" + line) if "[paths]\n" in text \
            else text + "\n[paths]\n" + line
        cfg.write_text(text)
        _write_jsonl(out / "rows", chirp)
    return db


def _write_jsonl(d, rs):
    d.mkdir(parents=True, exist_ok=True)
    by_day = {}
    for r in rs:
        by_day.setdefault(r["time"][:10], []).append(json.dumps({"table": "arrival", **r}))
    for day, lines in by_day.items():
        with open(d / f"{day}.jsonl", "a") as fh:
            fh.write("\n".join(lines) + "\n")


def test_jsonl_rows_read_by_reporter_window_and_table(tmp_path):
    from chirp_recorder.core.compare import read_jsonl_rows
    _write_jsonl(tmp_path, rows("chirp", [4_120_000]))
    with open(tmp_path / "2026-11-19.jsonl", "a") as fh:
        fh.write(json.dumps({"table": "capture", "time": "2026-11-19T02:00:00.000Z",
                             "reporter_id": REPORTER}) + "\nnot json\n")
    get = lambda **kw: read_jsonl_rows(tmp_path, **{**dict(                 # noqa: E731
        reporter_id=REPORTER, since_iso="2026-11-19T00:00:00Z",
        until_iso="2026-11-20T00:00:00Z"), **kw})
    got = get()
    assert len(got) == 6 and all("table" not in r for r in got)
    assert get(reporter_id="AC0G/B1A") == []
    assert len(get(since_iso="2026-11-19T03:30:00Z")) == 3
    assert read_jsonl_rows(tmp_path / "absent", reporter_id=REPORTER,
                           since_iso="2026-11-19T00:00:00Z", until_iso="2026-11-20T00:00:00Z") == []


def test_compare_subcommand_reads_the_stations_sink(tmp_path, monkeypatch, configured,
                                                    run_cli_full):
    db = _sink_with_both(tmp_path, rows("chirp", [4_120_000, 42_906_000, 50_939_000]),
                         rows("codar", [4_121_500, 42_904_000, 50_940_200]), configured)
    monkeypatch.setenv("HAMSCI_SINK_PATH", str(db))
    rc, out, err = run_cli_full("compare", "--instance", "AC0G-B4",
                                "--since", "2026-11-19T00:00:00Z",
                                "--until", "2026-11-20T00:00:00Z", "--json")
    d = json.loads(out)
    assert rc == 0, err
    assert d["status"] == "pass" and len(d["matched"]) == 3
    assert {r["name"] for r in d["rows"]} >= {"matched", "coverage", "doppler"}
    assert d["match"] == "bin"
    # chirp-recorder's rows come from its own JSONL, which nothing drains;
    # codar-sounder's from the sink queue, which its upload drains
    assert d["sources"]["chirp-recorder"].endswith("chirp-data/rows")
    assert "pending_uploads" in d["sources"]["codar-sounder"]


def test_compare_subcommand_takes_the_match_mode(tmp_path, monkeypatch, configured,
                                                 run_cli_full):
    true = [4_120_000, 42_906_000, 50_939_000]
    db = _sink_with_both(tmp_path, rows("chirp", true), rows("codar", [d + 6_000 for d in true]),
                         configured)
    monkeypatch.setenv("HAMSCI_SINK_PATH", str(db))
    window = ("--instance", "AC0G-B4", "--since", "2026-11-19T00:00:00Z", "--json")
    rc_bin, out_bin, _ = run_cli_full("compare", *window)
    rc_ex, out_ex, _ = run_cli_full("compare", *window, "--match", "exact")
    assert (rc_bin, json.loads(out_bin)["match"], json.loads(out_bin)["status"]) == (0, "bin", "pass")
    assert (rc_ex, json.loads(out_ex)["match"], json.loads(out_ex)["status"]) == (1, "exact", "fail")


def test_compare_subcommand_exits_1_unless_it_passes(tmp_path, monkeypatch, configured,
                                                     run_cli_full):
    db = _sink_with_both(tmp_path, rows("chirp", [4_120_000]), rows("codar", [4_120_000]),
                         configured)
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


def test_clusters_can_take_every_row_for_a_counter_placed_night():
    r = rows("chirp", [4_120_000], t6=False)
    assert clusters(r, producer="x") == []
    (c,) = clusters(r, producer="x", t6_only=False)
    assert c.median_delay_ns == 4_120_000 and c.spread_ns == 600


def test_the_seeds_mid_atlantic_gate_gives_compare_its_rung_spacing():
    """I5: with the seed's gate, compare explains codar-sounder's rungs."""
    from chirp_recorder.core.catalogue import load_seed
    from chirp_recorder.core.channels import groups_for
    from chirp_recorder.core.compare import rung_spacings
    seed = load_seed()
    rung = rung_spacings(groups_for([seed["ASSA"], seed["CEDR"], seed["BMLR"]]))
    assert set(rung) == {G} and math.isclose(rung[G], 1 / (25_733.913 * 1.9456e-3))
    true = [4_120_000, 42_906_000, 50_939_000]
    v = compare(rows("chirp", true), rows("codar", true + [42_906_000 + 2 * 19_973_000]),
                rung_s=rung)
    assert v.status == "pass" and [why for _, why in v.explained] == ["rung +2 of 42.910 ms"]
    assert rung_spacings(groups_for([seed["BMLR"]])) == {}        # no gate, no rung
