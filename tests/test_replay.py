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
from chirp_recorder.core.compare import clusters
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
    # An index label never claims T6, whatever tier its line names: on the golden
    # night those labels scatter 200-420 ms against radiod's counter while
    # claiming 1-5 us.  Null block; the line's tier and sigma stay as engineering.
    for r in (a, b):
        assert r["u_epoch_ns"] is None and r["origin"] == ""
        assert r["engineering"]["delay_basis"] == "index"
    assert a["engineering"]["label_tier"] == "T6" and a["engineering"]["label_sigma_ns"] == "1000"
    assert b["engineering"]["label_tier"] == "T5"
    assert b["engineering"]["label_sigma_ns"] == "10206461"
    six = [dict(r, time=f"2026-11-20T0{i}:00:00Z") for i in range(6) for r in rows]
    assert clusters(six, producer="chirp-recorder") == []


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
    assert main(["replay", str(capture_dir), "--gate-us", "1945.6", "--label", "counter"]) == 0
    rows = [json.loads(s) for s in capsys.readouterr().out.splitlines()]
    assert rows and {r["engineering"]["label_source"] for r in rows} == {"counter"}


# --- fix round 1: captures placed by radiod's input counter ------------------
def test_counter_label_places_x0_by_the_input_counter_on_the_runs_epoch(capture_dir):
    """utc(x[0]) = in_idx0 / in_rate + the run's input epoch (GPS seconds,
    the median over the run's lines) + GPS-to-UTC.  The per-line t0 and its
    per-line epoch jitter never enter, so captures of one run stay consistent
    to the counter's own precision."""
    from ka9q.rtp_recorder import GPS_LEAP_SECONDS, GPS_UTC_OFFSET
    from chirp_recorder.core.replay import counter_labels
    idx_a, idx_b = 4_637_344_608_000.0, 4_637_344_608_000.0 + 1800 * RATE
    lines = (_line("a.npy", in_idx0=idx_a)
             + _line("b.npy", in_idx0=idx_b, t0="2026-11-20T06:00:01.703230+00:00")
             .replace(f'"in_epoch":{EPOCH}', f'"in_epoch":{EPOCH + 0.004}'))
    (capture_dir / "index.txt").write_text(lines)
    labels = counter_labels(capture_dir, fs=FS, channel_centre_hz=4.463e6)
    epoch_ns = round((EPOCH + 0.002) * 1e9)                 # median of the two lines
    off = (GPS_UTC_OFFSET - GPS_LEAP_SECONDS) * 10**9
    assert abs(labels["a.npy"] - (int(idx_a) * 10**9 // RATE + epoch_ns + off)) < 1_000
    assert labels["b.npy"] - labels["a.npy"] == 1800 * 10**9      # exactly the counter's
    rows = replay(capture_dir, [], GROUPS, fs=FS, channel_centre_hz=4.463e6, label="counter")
    a, b = _arrival(rows, "a.npy"), _arrival(rows, "b.npy")
    assert a["engineering"]["label_source"] == "counter" and a["u_epoch_ns"] is None
    assert a["engineering"]["delay_basis"] == "counter"
    assert a["delay_ns"] == b["delay_ns"]        # same synthetic capture, same phase


def test_relabel_rows_refolds_delays_onto_new_labels(capture_dir):
    from chirp_recorder.core.replay import relabel_rows
    (capture_dir / "index.txt").write_text(_line("a.npy"))
    rows = replay(capture_dir, [], GROUPS, fs=FS, channel_centre_hz=4.463e6)
    moved = relabel_rows(rows, {"a.npy": T0_NS + 7_000_123}, source="counter")
    (r0,), (r1,) = rows, moved
    assert r1["delay_ns"] == (r0["delay_ns"] + 7_000_123) % 10**9
    assert r1["engineering"]["label_source"] == "counter"
    assert r1["engineering"]["label_utc_ns"] == str(T0_NS + 7_000_123)
    assert r0["engineering"]["label_source"] == "index"          # the input stays


def test_an_unknown_label_mode_is_refused(capture_dir):
    (capture_dir / "index.txt").write_text(_line("a.npy"))
    with pytest.raises(ValueError, match="label"):
        replay(capture_dir, [], GROUPS, fs=FS, channel_centre_hz=4.463e6, label="host")
