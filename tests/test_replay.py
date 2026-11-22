"""replay(): codar_grab captures through the detector, each labelled the way
relabel.py labels it, or from its own index.txt line when no minute rows exist.
Every time and index here is made up."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest
from hamsci_dsp.chirp.waveform import Waveform

from chirp_recorder.core.channels import Group
from chirp_recorder.core.replay import replay

FS = 64_000
WF = Waveform(4.463e6, -25_733.913, 1.0, None, 1945.6e-6, 0.5)
GROUPS = [Group(WF, ("TX1",))]
TAU_S = 0.0123456
RATE = 129_600_000
EPOCH = 1_479_000_000.25
T0 = "2026-11-20T05:30:01.303230+00:00"
T0_NS = int(datetime(2026, 11, 20, 5, 30, 1, tzinfo=timezone.utc).timestamp()) * 10**9 + 303_230_000


def _line(name, *, t0=T0, f=4_463_000, n=6 * FS, tier="T6", sigma_ns=1000,
          in_idx0=4_637_344_608_000.0):
    lab = {"t0_raw_utc": 0.0, "cpi_minus_anchor_s": 0.0, "rtp0": 1, "ssrc": 2,
           "in_idx0": in_idx0, "in_rate": RATE, "in_epoch": EPOCH, "n_agree": 5,
           "tier": tier, "sigma_ns": sigma_ns}
    return (f"saved /var/tmp/chirp-golden/{name} {n} samples fs={FS} f={f} t0={t0} "
            f"source=rtp_to_utc+authority offset_s=-0.020862 "
            f"labels={json.dumps(lab, separators=(',', ':'))}\n")


@pytest.fixture
def capture_dir(tmp_path, synth):
    x = synth(fs=FS, arrivals=[(TAU_S, 1.0)], seconds=6, noise=0.05)
    for name in ("a.npy", "b.npy"):
        np.save(tmp_path / name, x)
    return tmp_path


def _arrival(rows, capture):
    (r,) = [r for r in rows if r["engineering"]["capture"] == capture]
    return r


def test_without_minutes_each_capture_takes_its_own_index_label(capture_dir):
    (capture_dir / "index.txt").write_text(
        _line("a.npy") + _line("b.npy", tier="T5", sigma_ns=10_206_461))
    rows = replay(capture_dir, [], GROUPS, fs=FS, channel_centre_hz=4.463e6)
    a, b = _arrival(rows, "a.npy"), _arrival(rows, "b.npy")
    for r in (a, b):
        assert r["engineering"]["label_source"] == "index"
        assert r["engineering"]["label_utc_ns"] == str(T0_NS)
        assert abs(r["delay_ns"] - (T0_NS + round(TAU_S * 1e9)) % 10**9) < 50
    # tier T6 gets the T6 block, with the label's sigma as its clock-tie error
    assert a["u_epoch_ns"] == 1000.0 and a["origin"] == "t6_anchor"
    assert a["engineering"]["label_tier"] == "T6"
    # any other tier gets the null block and names its basis
    assert b["u_epoch_ns"] is None and b["engineering"]["delay_basis"] == "authority"
    assert b["engineering"]["label_sigma_ns"] == "10206461"


def test_with_minutes_the_nearest_anchor_in_the_run_labels_the_capture(capture_dir):
    idx0 = 4_637_344_608_000.0
    # Anchor one minute earlier on the counter, placed 2.5 ms off the index label.
    anchor_idx = idx0 - 60 * RATE
    anchor_ns = T0_NS - 60 * 10**9 + 2_500_000
    minutes = capture_dir / "minutes-20261120.jsonl"
    minutes.write_text("\n".join(json.dumps(r) for r in (
        {"t3_shadow": {"t6_anchor_input_index": anchor_idx, "t6_anchor_utc_ns": anchor_ns}},
        {"t3_shadow": {"t6_anchor_input_index": anchor_idx + 60 * RATE,
                       "t6_anchor_utc_ns": anchor_ns + 60 * 10**9}},
        {"t3_shadow": {}},                                   # minutes with no anchor
        {"other": 1})) + "\n")
    (capture_dir / "index.txt").write_text(_line("a.npy", in_idx0=idx0))
    (r,) = replay(capture_dir, [minutes], GROUPS, fs=FS, channel_centre_hz=4.463e6)
    assert r["engineering"]["label_source"] == "relabel"
    assert r["engineering"]["label_utc_ns"] == str(T0_NS + 2_500_000)
    assert abs(r["delay_ns"] - (T0_NS + 2_500_000 + round(TAU_S * 1e9)) % 10**9) < 50
    assert r["origin"] == "t6_anchor"


def test_an_anchor_from_another_run_skips_the_capture(capture_dir, caplog):
    minutes = capture_dir / "m.jsonl"
    minutes.write_text(json.dumps({"t3_shadow": {
        "t6_anchor_input_index": 1_000.0, "t6_anchor_utc_ns": T0_NS}}) + "\n")
    (capture_dir / "index.txt").write_text(_line("a.npy"))
    assert replay(capture_dir, [minutes], GROUPS, fs=FS, channel_centre_hz=4.463e6) == []
    assert "another radiod run" in caplog.text


def test_neighbouring_anchors_that_disagree_skip_the_capture(capture_dir, caplog):
    idx0 = 4_637_344_608_000.0
    minutes = capture_dir / "m.jsonl"
    minutes.write_text("\n".join(json.dumps({"t3_shadow": {
        "t6_anchor_input_index": idx0 + k * 60 * RATE,
        "t6_anchor_utc_ns": T0_NS + k * 60 * 10**9 + (2_000_000 if k else 0)}})
        for k in (0, 1)) + "\n")
    (capture_dir / "index.txt").write_text(_line("a.npy", in_idx0=idx0))
    assert replay(capture_dir, [minutes], GROUPS, fs=FS, channel_centre_hz=4.463e6) == []
    assert "disagree" in caplog.text


def test_other_frequencies_rates_and_missing_files_are_left_out(capture_dir):
    (capture_dir / "index.txt").write_text(
        _line("a.npy", f=13_475_000) + _line("missing.npy") + "a stray line\n"
        + _line("b.npy"))
    rows = replay(capture_dir, [], GROUPS, fs=FS, channel_centre_hz=4.463e6)
    assert {r["engineering"]["capture"] for r in rows} == {"b.npy"}
    assert replay(capture_dir, [], GROUPS, fs=96_000, channel_centre_hz=4.463e6) == []


def test_replay_subcommand_prints_one_json_row_per_arrival(capture_dir, capsys):
    from chirp_recorder.cli import main
    (capture_dir / "index.txt").write_text(_line("a.npy"))
    assert main(["replay", str(capture_dir), "--gate-us", "1945.6"]) == 0
    lines = capsys.readouterr().out.splitlines()
    rows = [json.loads(s) for s in lines]
    assert rows and {r["engineering"]["label_source"] for r in rows} == {"index"}
    assert any(abs(r["delay_ns"] - (T0_NS + round(TAU_S * 1e9)) % 10**9) < 50 for r in rows)
