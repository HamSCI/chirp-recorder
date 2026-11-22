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
HALF_SWEEP_NS = 500_000_000      # codar-sounder's replica: sweep start at the centre


def rows(producer, delays, *, doppler_hz=0.1, s4=0.2, sigma_phi=0.05, t6=True):
    """Six T6 arrivals per true delay, one per capture, for one reporter and group G,
    each written in its producer's convention.  chirp-recorder's delay_ns holds
    the true delay (detect.py removes the Doppler); codar-sounder's holds
    tau - f/k, because its beat tone sits at -k*tau + f (sigmondb.rows_for_cpi),
    plus half a sweep as releases before 0.8.0 wrote it: their replica starts
    the sweep at the channel centre rather than the band edge (round 2; test
    below).  The detector_version "codar/test" names no codar-sounder release,
    so compare corrects these rows as it does 0.7.0 ones."""
    prov = sr.provenance("T6", "epoch-1") if t6 else sr.provenance(None)
    shift = round(-doppler_hz / K * 1e9) if (producer == "codar" and doppler_hz) else 0
    if producer == "codar":
        shift += HALF_SWEEP_NS
    out = []
    for d in delays:
        for i, s in enumerate(SCATTER_NS):
            out.append(sr.arrival_row(
                time=f"2026-11-19T{1 + i:02d}:00:00Z", reporter_id=REPORTER, group_key=G,
                centre_hz=4.463e6, rate_hz_per_s=K, delay_ns=(d + s + shift) % 10**9,
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
    """No Doppler, no C1 conversion: the row stays out of the fine tier.
    Round 2 (R1-I4) puts it in the coarse tier, which covers but never matches."""
    chirp = rows("chirp", [4_120_000, 42_906_000, 50_939_000])
    for r in chirp[:6]:
        r["doppler_hz"] = None
    codar = rows("codar", [4_120_000, 42_906_000, 50_939_000])
    one = compare(chirp, codar, rung_s={G: 19.973e-3}, coarse=False)
    assert [c.median_delay_ns for c in one.codar_unmatched] == [4_120_000 + round(-0.1 / K * 1e9)]
    two = compare(chirp, codar, rung_s={G: 19.973e-3})
    assert two.codar_unmatched == [] and len(two.matched) == 2 and len(two.coarse) == 1
    assert two.status == "insufficient"                 # two fine clusters, not three


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
    # codar rows hold tau - f/k + P/2: 509 ms + 3.886 us
    assert {r["delay_ns"] // 1000 for r in get(producer="codar-sounder")} == {509003, 509004}
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


# --- round 2: two tiers (R1-I3, R1-I4), mean offset, codar JSONL (R1-I5) -------
def _flag(rs, **eng):
    """Mark rows as detect.py leaves them uncorrected (gated off, ambiguous,
    clipped); a gated-off row also loses its published Doppler."""
    out = []
    for r in rs:
        e = dict(r["engineering"], **{k: "1" for k in eng})
        r = dict(r, engineering=e)
        if "doppler_low_coherence" in eng:
            r["doppler_hz"] = None
        out.append(r)
    return out


TRUE3 = [4_120_000, 42_906_000, 50_939_000]


def test_an_ambiguous_row_is_never_converted_and_goes_to_the_coarse_tier():
    """detect.py kept the Doppler of an ambiguous row but skipped the
    correction, so the row already holds tau - g f/k.  Subtracting f/k again
    would move it twice; it joins the coarse tier on its own delay."""
    amb = _flag(rows("chirp", [60_000_000], doppler_hz=0.5), doppler_ambiguous=True)
    v = compare(rows("chirp", TRUE3) + amb, rows("codar", TRUE3), rung_s={G: 0.02})
    assert len(v.matched) == 3                       # the fine tier alone
    assert [c.median_delay_ns for c in v.chirp_only] == [60_000_000]    # unconverted
    clipped = _flag(rows("chirp", [60_000_000], doppler_hz=0.1), doppler_clipped=True)
    v = compare(rows("chirp", TRUE3) + clipped, rows("codar", TRUE3), rung_s={G: 0.02})
    assert [c.median_delay_ns for c in v.chirp_only] == [60_000_000]


def _coarse_case(d_ns, **kw):
    """Three fine pairs, plus a codar cluster at f = 0.15 Hz whose only
    chirp-recorder partner sits d_ns away in the coarse tier (gated off)."""
    codar_extra = rows("codar", [70_000_000], doppler_hz=0.15)
    codar_delay = (codar_extra[0]["delay_ns"] - SCATTER_NS[0] - HALF_SWEEP_NS) % 10**9
    coarse = _flag(rows("chirp", [codar_delay + d_ns], doppler_hz=0.15),
                   doppler_low_coherence=True)
    return compare(rows("chirp", TRUE3) + coarse, rows("codar", TRUE3) + codar_extra,
                   rung_s={G: 0.02}, match="exact", **kw)


def test_a_coarse_cluster_7_us_off_at_0_15_hz_covers_and_leaves_matched_alone():
    v = _coarse_case(7_000)                     # limit 5 + 2.0 * 5.83 = 16.7 us
    by = {r.name: r for r in v.rows}
    assert v.status == "pass" and by["coverage"].value == 1.0
    assert len(v.matched) == 3 and by["matched"].value <= 5_000
    assert by["matched_coarse"].gating is False and by["matched_coarse"].passed is True
    assert by["matched_coarse"].value == 7_000
    # |1 - g| <= 2 for |g| <= 1 (real captures showed g down to -0.7)
    assert math.isclose(by["matched_coarse"].limit, 5_000 + 2.0 * 0.15 / 25_733.913 * 1e9)
    (row,) = v.to_dict()["matched_coarse"]
    assert row[2] == 7_000 and row[4] == G


def test_the_same_coarse_cluster_20_us_off_is_missed():
    assert _coarse_case(15_000).status == "pass"        # inside 16.7 us with the 2.0 factor
    v = _coarse_case(20_000)
    by = {r.name: r for r in v.rows}
    assert by["coverage"].value == 0.75 and by["coverage"].passed is False
    assert by["matched_coarse"].passed is False and v.status == "fail"


def test_one_tier_leaves_the_coarse_cluster_uncovered():
    v = _coarse_case(7_000, coarse=False)
    assert {r.name: r for r in v.rows}["coverage"].value == 0.75
    assert v.to_dict()["tiers"] == ["fine"]
    assert _coarse_case(7_000).to_dict()["tiers"] == ["fine", "coarse"]


def test_the_coarse_tier_never_feeds_matched_or_doppler():
    """Only fine rows support the precision claim: a night of gated rows alone
    is insufficient, however well it covers."""
    gated = _flag(rows("chirp", TRUE3), doppler_low_coherence=True)
    v = compare(gated, rows("codar", TRUE3), rung_s={G: 0.02})
    assert v.status == "insufficient" and v.matched == []
    assert {r.name: r for r in v.rows}["coverage"].value == 1.0


def _twenty(offset_ns):
    true = [1_000_000 + 2_000_000 * i for i in range(20)]
    return rows("chirp", true), rows("codar", [d - offset_ns for d in true])


def test_mean_offset_is_reported_and_gates_only_when_asked():
    chirp, codar = _twenty(6_000)            # each pair inside bin/2 + 5 us; mean 6 us
    v = compare(chirp, codar, rung_s={})
    row = {r.name: r for r in v.rows}["mean_offset"]
    assert v.status == "pass" and row.gating is False and row.passed is False
    assert row.value == 6_000 and row.limit == 5_000
    assert v.to_dict()["mean_offset"] == {"gating": False, "n": 20, "mean_ns": 6_000.0}
    assert compare(chirp, codar, rung_s={}, mean_offset=True).status == "fail"
    chirp, codar = _twenty(1_000)
    assert compare(chirp, codar, rung_s={}, mean_offset=True).status == "pass"


def test_mean_offset_needs_twenty_matched_clusters():
    v = compare(rows("chirp", TRUE3), rows("codar", TRUE3), rung_s={G: 0.02}, mean_offset=True)
    assert v.status == "insufficient"


def _write_codar_arrivals(d, rs):
    """codar-sounder's layout (bf3f360): <dir>/arrivals-YYYYMMDD.jsonl."""
    d.mkdir(parents=True, exist_ok=True)
    for r in rs:
        with open(d / f"arrivals-{r['time'][:10].replace('-', '')}.jsonl", "a") as fh:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")


def test_codar_rows_read_from_daily_jsonl(tmp_path):
    """codar-sounder's daily files (<dir>/arrivals-YYYYMMDD.jsonl, one
    sink-insert row dict per line), read with the reporter and window rules."""
    from chirp_recorder.core.compare import read_codar_jsonl_rows
    d = tmp_path / "arrivals"
    rs = rows("codar", [4_120_000])
    _write_codar_arrivals(d, rs)
    with open(d / "arrivals-20261119.jsonl", "a") as fh:
        fh.write("not json\n")
    (d / "2026-11-19.jsonl").write_text(json.dumps(rs[0]) + "\n")       # not its name: ignored
    (d / "notes.txt").write_text("ignored\n")
    window = dict(since_iso="2026-11-19T00:00:00Z", until_iso="2026-11-20T00:00:00Z")
    assert read_codar_jsonl_rows(d, reporter_id=REPORTER, **window) == rs
    assert read_codar_jsonl_rows(d, reporter_id="AC0G/B1A", **window) == []
    assert len(read_codar_jsonl_rows(d, reporter_id=REPORTER, since_iso="2026-11-19T03:30:00Z",
                                     until_iso="2026-11-20T00:00:00Z")) == 3


def test_codar_rows_read_what_codar_sounders_emitter_writes(tmp_path):
    """The fixture comes from codar-sounder's own Emitter (its _keep_arrivals),
    so the reader meets the real file name and row format."""
    import numpy as np
    from datetime import datetime, timezone
    from codar_sounder.core.dechirp import dechirp
    from codar_sounder.core.sigmondb import Emitter
    from hamsci_dsp.chirp.waveform import Waveform, gated
    from chirp_recorder.core.compare import read_codar_jsonl_rows

    class Sink:
        def __init__(self):
            self.rows = []

        def insert(self, rs):
            self.rows.extend(rs)

    arr_w, cap_w = Sink(), Sink()
    em = Emitter((arr_w, cap_w), reporter_id=REPORTER, detector_version="codar-ref",
                 state_dir=tmp_path)
    wf = Waveform(4.463e6, K, 1.0, None, 1945.6e-6, 0.5)
    x = np.tile(gated(wf, 64_000, 0.0429), 8).astype(np.complex64)
    res = dechirp(x, sample_rate_hz=64_000.0, sweep_rate_hz_per_s=K, sweep_repetition_hz=1.0)
    n = em.emit(result=res, cpi_start_utc=datetime(2026, 11, 19, 5, 0, tzinfo=timezone.utc),
                tdma_offset_samples=0, centre_hz=4.463e6, period_s=1.0, expected=["X"],
                duration_s=8.0, anchor=None, radiod="r")
    assert n > 0 and arr_w.rows
    assert [p.name for p in (tmp_path / "arrivals").iterdir()] == ["arrivals-20261119.jsonl"]
    got = read_codar_jsonl_rows(tmp_path / "arrivals", reporter_id=REPORTER,
                                since_iso="2026-11-19T00:00:00Z",
                                until_iso="2026-11-20T00:00:00Z")
    assert got == json.loads(json.dumps(arr_w.rows))


def test_compare_subcommand_reads_codar_rows_from_jsonl(tmp_path, monkeypatch, configured,
                                                        run_cli_full):
    _sink_with_both(tmp_path, rows("chirp", TRUE3), [], configured)
    monkeypatch.setenv("HAMSCI_SINK_PATH", str(tmp_path / "absent.db"))   # no sink needed
    d = tmp_path / "codar-rows"
    _write_codar_arrivals(d, rows("codar", TRUE3))
    rc, out, err = run_cli_full("compare", "--instance", "AC0G-B4", "--since",
                                "2026-11-19T00:00:00Z", "--codar-rows", str(d), "--json")
    doc = json.loads(out)
    assert rc == 0, err
    assert doc["status"] == "pass" and doc["sources"]["codar-sounder"] == str(d)



def test_codar_rows_land_on_the_sweep_start_and_old_rows_get_corrected():
    """codar-sounder's replica sweeps from the channel centre (f = k t from
    t = 0, dechirp.make_replica); the transmitted chirp, as hamsci-dsp and
    clean.py model it, sweeps through the centre at mid-sweep.  So its beat
    reads tau + sweep/2.  codar-sounder 0.8.0 takes that half sweep off in
    rows_for_cpi; compare passes its rows through and takes the half sweep
    off a 0.7.0 row (the same row, half a sweep later)."""
    import numpy as np
    from datetime import datetime, timezone
    from codar_sounder import __version__ as codar_version
    from codar_sounder.core.dechirp import dechirp
    from codar_sounder.core.sigmondb import rows_for_cpi
    from hamsci_dsp.chirp.waveform import Waveform, gated
    from chirp_recorder.core.compare import codar_bin_ns, from_codar_convention
    wf = Waveform(4.463e6, K, 1.0, None, 1945.6e-6, 0.5)
    tol = codar_bin_ns(K, 1.0) / 2
    for tau in (0.0429, 0.3, 0.7):
        x = np.tile(gated(wf, 64_000, tau), 8).astype(np.complex64)
        res = dechirp(x, sample_rate_hz=64_000.0, sweep_rate_hz_per_s=K, sweep_repetition_hz=1.0)
        _, arrs = rows_for_cpi(result=res, cpi_start_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
                               tdma_offset_samples=0, centre_hz=4.463e6, period_s=1.0,
                               reporter_id="X", t_level="T6", expected=[],
                               detector_version=f"codar-sounder/{codar_version}", duration_s=8)
        raw = arrs[0]                                     # its strongest peak
        assert abs(raw["delay_ns"] - round(tau * 1e9)) <= tol
        (same,) = from_codar_convention([raw])            # 0.8.0: passes through
        assert same["delay_ns"] == raw["delay_ns"]
        old = dict(raw, detector_version="codar-sounder/0.7.0",
                   delay_ns=(raw["delay_ns"] + HALF_SWEEP_NS) % 10**9)
        (back,) = from_codar_convention([old])
        assert abs(back["delay_ns"] - round(tau * 1e9)) <= tol


def test_the_half_sweep_correction_stops_at_the_fixed_codar_sounder_version(monkeypatch):
    """CODAR_HALF_SWEEP_FIXED_FROM names codar-sounder 0.8.0, the first fixed
    release: its rows and later ones pass through, 0.7.0 rows and rows that
    name no codar-sounder release get corrected.  None corrects them all."""
    import chirp_recorder.core.compare as C
    from chirp_recorder.core.compare import from_codar_convention
    assert C.CODAR_HALF_SWEEP_FIXED_FROM == "codar-sounder/0.8.0"
    row = dict(rows("codar", [4_120_000], doppler_hz=0.0)[0], delay_ns=504_120_000)
    old = dict(row, detector_version="codar-sounder/0.7.0")
    new = dict(row, detector_version="codar-sounder/0.8.0")
    later = dict(row, detector_version="codar-sounder/0.10.1")
    odd = dict(row, detector_version="something-else")
    got = [r["delay_ns"] for r in from_codar_convention([old, new, later, odd])]
    assert got == [4_120_000, 504_120_000, 504_120_000, 4_120_000]
    monkeypatch.setattr(C, "CODAR_HALF_SWEEP_FIXED_FROM", None)
    assert [r["delay_ns"] for r in from_codar_convention([old, new, later, odd])] \
        == [4_120_000] * 4


def test_a_suffixed_codar_sounder_version_still_counts_as_fixed():
    """A local or dev build (0.8.0+g1234, 0.8.1.dev3, 0.9.0rc1) parses on its
    leading X.Y.Z, so compare never takes the half sweep off its rows twice."""
    from chirp_recorder.core.compare import _version, from_codar_convention
    assert _version("codar-sounder/0.8.0+g1234") == (0, 8, 0)
    assert _version("codar-sounder/0.8.1.dev3") == (0, 8, 1)
    assert _version("codar-sounder/0.9.0rc1") == (0, 9, 0)
    assert _version("codar-sounder/0.7.0+local") == (0, 7, 0)
    assert _version("codar-sounder/garbage") is None
    assert _version("other/0.8.0") is None
    row = dict(rows("codar", [4_120_000], doppler_hz=0.0)[0], delay_ns=504_120_000)
    vers = ["codar-sounder/0.8.0+g1234", "codar-sounder/0.8.1.dev3", "codar-sounder/0.7.0+local"]
    got = [r["delay_ns"] for r in from_codar_convention(
        [dict(row, detector_version=v) for v in vers])]
    assert got == [504_120_000, 504_120_000, 4_120_000]


# --- round 3: the readers run as root on files another account can write ----
def _hostile_dir(tmp_path):
    import os
    d = tmp_path / "arrivals"
    good = rows("codar", [4_120_000])
    _write_codar_arrivals(d, good)                                    # arrivals-20261119.jsonl
    os.symlink("/dev/zero", d / "arrivals-20261118.jsonl")           # endless bytes
    os.mkfifo(d / "arrivals-20261117.jsonl")                         # would block on open
    with open(d / "arrivals-20261116.jsonl", "wb") as fh:             # past the 64 MiB cap
        fh.truncate(64 * 1024 * 1024 + 1)
    with open(d / "arrivals-20261115.jsonl", "w") as fh:              # nesting past recursion
        fh.write("[" * 200_000 + "\n" + json.dumps(good[0]) + "\n")
    return d, good


def test_the_codar_reader_skips_symlinks_fifos_oversized_files_and_deep_json(tmp_path, caplog):
    import logging
    import threading
    from chirp_recorder.core.compare import MAX_JSONL_BYTES, read_codar_jsonl_rows
    assert MAX_JSONL_BYTES == 64 * 1024 * 1024
    d, good = _hostile_dir(tmp_path)
    out = {}

    def run():
        out["rows"] = read_codar_jsonl_rows(d, reporter_id=REPORTER,
                                            since_iso="2026-11-15T00:00:00Z",
                                            until_iso="2026-11-20T00:00:00Z")
    with caplog.at_level(logging.WARNING):
        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(20)
    assert not t.is_alive(), "the reader hung"
    # the good file's six rows, plus the one row after the deep line
    assert len(out["rows"]) == 7 and all(r in out["rows"] for r in good)
    text = caplog.text
    assert "arrivals-20261118.jsonl" in text and "arrivals-20261117.jsonl" in text
    assert "arrivals-20261116.jsonl" in text


def test_the_chirp_rows_reader_shares_the_hardening(tmp_path):
    import os
    from chirp_recorder.core.compare import read_jsonl_rows
    d = tmp_path / "rows"
    _write_jsonl(d, rows("chirp", [4_120_000]))
    os.symlink("/dev/zero", d / "2026-11-18.jsonl")
    os.mkfifo(d / "2026-11-17.jsonl")
    got = read_jsonl_rows(d, reporter_id=REPORTER, since_iso="2026-11-15T00:00:00Z",
                          until_iso="2026-11-20T00:00:00Z")
    assert len(got) == 6
