"""The physics stopgap against the truth: synthetic transmitters at known
paths and known sweep phases read their group range, at any CPI phase.

Until 2026-11-22 these tests compared bytes with codar-sounder's
_TransmitterPipeline (pre-flight C3).  That reference aims its replica at
-offset from an unreferenced buffer start and models half a sweep, so
chirp-recorder's physics no longer matches it, by design."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from chirp_recorder import __version__
from chirp_recorder.config import ConfigError
from chirp_recorder.core.writer import RowWriter
from chirp_recorder.physics import PhysicsOut, PhysicsStopgap
from chirp_recorder.physics.spots import make_spots_writer


def _stop(plan_and_txs, settings, out, **kw):
    return PhysicsStopgap(*plan_and_txs, settings=settings, output_dir=out,
                          reporter_id="AC0G/B4", radiod="b4", host_call="AC0G",
                          host_grid="EM38", **kw)


def _records(root: Path):
    f = next(root.rglob("*.jsonl"))
    return [json.loads(line) for line in f.read_text().splitlines()]


def _plan(rate):
    from chirp_recorder.core.catalogue import Transmitter
    from chirp_recorder.core.channels import ChannelPlan, groups_for
    from conftest import PHYS_TX
    tx = Transmitter(id="TX1", kind="codar", centre_hz=4_463_000.0, rate_hz_per_s=rate,
                     period_s=1.0, sweep_bw_hz=25_734.0, lat=PHYS_TX[0], lon=PHYS_TX[1])
    plan = ChannelPlan("4.4630MHz", "radiod.local", 4_463_000.0, 64_000, groups_for([tx]))
    return plan, {"TX1": tx}


@pytest.mark.parametrize("rate", [-25_733.913, 25_733.913])
def test_a_path_reads_its_group_range_at_every_cpi_phase(
        tmp_path, monkeypatch, physics_settings, split_into_intervals, rate):
    """A TX sweeping from the UTC second over a 420 km path reads 420 km
    whether the CPI opens 0, 0.25, 0.6 or 0.75 s after the second, and its
    SNR holds within 1 dB across those phases (a run's phase is arbitrary).
    The per-sweep MAD pre-filter is off: on stationary synthetic input it
    zeroes 0-3 clean sweeps per CPI, whatever the phase, and each zeroed
    sweep lifts snr_db by about 1 dB (round 1 report, concern)."""
    import chirp_recorder.physics.dechirp as dc
    monkeypatch.setattr(dc, "SWEEP_MAD_REJECTION_K", float("inf"))
    from conftest import phys_cpi
    snrs = []
    for k, phase_s in enumerate((0.0, 0.25, 0.6, 0.75)):
        start_ns = 1_800_000_000 * 10**9 + int(round(phase_s * 1e9))
        x, start = phys_cpi(rate=rate, start_ns=start_ns, noise_db=-10.0)
        out = tmp_path / f"p{k}"
        stop = _stop(_plan(rate), physics_settings, out)
        spots = []
        for iv, lab in split_into_intervals(x, start):
            spots += stop.feed(iv, lab).spots
        assert spots, f"no physics row at CPI phase {phase_s}"
        rec = _records(out)[0]
        assert rec["group_range_km"] == pytest.approx(420.0, abs=12.0)
        assert rec["client"] == "chirp-recorder"
        snrs.append(rec["snr_db"])
    assert max(snrs) - min(snrs) < 1.0, snrs
    p = next((tmp_path / "p0").rglob("*.jsonl")).relative_to(tmp_path / "p0")
    assert p.parts[:2] == ("physics", "TX1") and len(p.parts) == 5


def test_spots_rows_read_the_path_and_name_chirp_recorder(
        tmp_path, monkeypatch, fake_sink, synth_cpi, plan_and_txs,
        physics_settings, split_into_intervals):
    import hs_uploader.sink as sink
    x, start = synth_cpi()
    # make_spots_writer builds the sink writer; the fake records its producer
    made = {}

    def from_env(**kw):
        made.update(kw)
        return fake_sink("new", producer=kw.get("producer"))
    monkeypatch.setattr(sink.Writer, "from_env", staticmethod(from_env))
    spots_writer = make_spots_writer()
    assert (made["table"], made["mode"], made["schema_version"], made["batch_rows"],
            made["producer"]) == ("spots", "codar", 1, 200, "chirp-recorder")
    stop = _stop(plan_and_txs, physics_settings, tmp_path / "newdir")
    w = RowWriter(output_dir=tmp_path / "new", sink_writers=(None, None, spots_writer),
                  phase_store=None)
    for iv, lab in split_into_intervals(x, start):
        w.put(None, [], [], spots=stop.feed(iv, lab).spots)       # the worker's path (C12)
    w.close()
    new = fake_sink.rows("new")
    assert new
    assert new[0]["group_range_km"] == pytest.approx(420.0, abs=12.0)
    assert new[0]["station_id"] == "TX1"
    assert {r["producer"] for r in new} == {"chirp-recorder"}
    assert {r["processing_version"] for r in new} == {f"chirp-recorder/{__version__}"}


def test_a_gap_discards_the_partial_cpi(
        tmp_path, plan_and_txs, physics_settings, intervals_with_gap,
        contiguous_intervals):
    stop = _stop(plan_and_txs, physics_settings, tmp_path)
    assert all(stop.feed(iv, lab).paths == [] for iv, lab in intervals_with_gap)
    # positive control (pre-flight C23): two contiguous intervals make one CPI
    outs = [stop.feed(iv, lab) for iv, lab in contiguous_intervals]
    assert outs[0].paths == [] and len(outs[1].paths) == 1
    assert outs[1].spots


def test_a_new_input_epoch_breaks_contiguity(
        tmp_path, plan_and_txs, physics_settings, contiguous_intervals):
    from dataclasses import replace
    stop = _stop(plan_and_txs, physics_settings, tmp_path)
    (iv0, l0), (iv1, l1) = contiguous_intervals
    stop.feed(iv0, l0)
    out = stop.feed(replace(iv1, input_epoch_gps_s=iv1.input_epoch_gps_s + 500.0), l1)
    assert out == PhysicsOut(paths=[], spots=[])


def test_coherent_seconds_must_be_whole_intervals(tmp_path, plan_and_txs, physics_settings):
    with pytest.raises(ConfigError, match="coherent_seconds"):
        _stop(plan_and_txs, dict(physics_settings, coherent_seconds=45), tmp_path)


def test_a_physics_exception_logs_one_warning_per_cpi(
        tmp_path, monkeypatch, caplog, plan_and_txs, physics_settings,
        contiguous_intervals):
    import chirp_recorder.physics.pipeline as pl

    def boom(*a, **k):
        raise RuntimeError("synthetic physics fault")
    monkeypatch.setattr(pl, "range_profile", boom)
    stop = _stop(plan_and_txs, physics_settings, tmp_path)
    with caplog.at_level(logging.WARNING):
        outs = [stop.feed(iv, lab) for iv, lab in contiguous_intervals]
    assert outs == [PhysicsOut([], []), PhysicsOut([], [])]
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1 and "synthetic physics fault" in warns[0].getMessage()


def test_a_transmitter_off_the_channel_centre_is_mixed_down(
        tmp_path, synth_cpi, plan_and_txs, physics_settings, split_into_intervals):
    """A channel centred 5 kHz below the transmitter sees it 5 kHz high; the
    stopgap mixes it back and finds the same peak as on the centred channel."""
    import dataclasses

    import numpy as np
    x, start = synth_cpi()
    plan, txs = plan_and_txs
    base = _stop((plan, txs), physics_settings, tmp_path / "base")
    for iv, lab in split_into_intervals(x, start):
        base.feed(iv, lab)
    shift = 5_000.0
    xs = (x * np.exp(2j * np.pi * shift * np.arange(x.size) / plan.sample_rate_hz)
          ).astype(np.complex64)
    plan2 = dataclasses.replace(plan, centre_hz=plan.centre_hz - shift)
    off = _stop((plan2, txs), physics_settings, tmp_path / "off")
    for iv, lab in split_into_intervals(xs, start):
        off.feed(iv, lab)
    a, b = _records(tmp_path / "base"), _records(tmp_path / "off")
    assert a and len(a) == len(b)
    for ra, rb in zip(a, b):
        assert ra["group_range_km"] == rb["group_range_km"]
        assert ra["snr_db"] == pytest.approx(rb["snr_db"], abs=0.05)


def _warns(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_a_precise_position_runs_without_a_warning(
        tmp_path, caplog, plan_and_txs, physics_settings, contiguous_intervals):
    with caplog.at_level(logging.WARNING):
        stop = _stop(plan_and_txs, physics_settings, tmp_path)
        outs = [stop.feed(iv, lab) for iv, lab in contiguous_intervals]
    assert _warns(caplog) == [] and outs[1].spots


def test_a_six_character_grid_runs_with_one_warning(
        tmp_path, caplog, plan_and_txs, physics_settings, contiguous_intervals):
    s = dict(physics_settings, position_source="grid_centre")
    with caplog.at_level(logging.WARNING):
        stop = PhysicsStopgap(*plan_and_txs, settings=s, output_dir=tmp_path,
                              reporter_id="AC0G/B4", radiod="b4", host_call="AC0G",
                              host_grid="EM38ww")
        outs = [stop.feed(iv, lab) for iv, lab in contiguous_intervals]
    w = _warns(caplog)
    assert len(w) == 1 and "AC0G/B4" in w[0] and "virtual height" in w[0]
    assert outs[1].spots


@pytest.mark.parametrize("src", ["grid_centre", "none"])
def test_a_four_character_grid_or_no_position_runs_no_physics(
        tmp_path, caplog, plan_and_txs, physics_settings, contiguous_intervals, src):
    s = {k: v for k, v in physics_settings.items() if not k.startswith("receiver_")}
    s["position_source"] = src
    with caplog.at_level(logging.WARNING):
        stop = _stop(plan_and_txs, s, tmp_path)
        outs = [stop.feed(iv, lab) for iv, lab in contiguous_intervals]
    assert len(_warns(caplog)) == 1
    assert outs == [PhysicsOut([], []), PhysicsOut([], [])]
    assert not list(tmp_path.rglob("*.jsonl"))


def test_settings_without_a_position_are_refused(tmp_path, plan_and_txs, physics_settings):
    s = {k: v for k, v in physics_settings.items() if not k.startswith("receiver_")}
    with pytest.raises(ConfigError, match="receiver_lat"):
        _stop(plan_and_txs, s, tmp_path)
    with pytest.raises(ConfigError, match="position_source"):
        _stop(plan_and_txs, {k: v for k, v in s.items() if k != "position_source"}, tmp_path)


def test_an_offset_read_off_the_rows_aims_physics_on_a_golden_capture(
        tmp_path, plan_and_txs, physics_settings, split_into_intervals):
    """Real data: one K3LR capture (Task 17).  The rows path measures the
    strongest arrival's phase after the second; an offset of that phase less
    a chosen 420 km path, the rule an operator follows, must put physics'
    peak at 420 km.  The capture holds 30 s, so physics runs a 30 s CPI.
    Absent captures fail, naming tests/golden/README.md."""
    import datetime as dt
    import os

    import numpy as np

    from chirp_recorder.core.detect import detect_group
    golden = Path(os.environ.get("CHIRP_GOLDEN_DIR",
                                 "/home/mjh/hamsci/tasks/chirp-registry/golden")) / "K3LR"
    caps = sorted(golden.glob("*.npy"))
    if not caps:
        pytest.fail(f"no K3LR golden captures in {golden}; see tests/golden/README.md",
                    pytrace=False)
    x = np.load(caps[0])
    assert x.size == 30 * 64_000
    start = dt.datetime(2026, 11, 20, 0, 30, 1, tzinfo=dt.timezone.utc)  # on the second
    plan, txs = plan_and_txs
    r = detect_group(x, fs=64_000, channel_centre_hz=plan.centre_hz, group=plan.groups[0])
    tau_s = max(r.arrivals, key=lambda a: a.level_db).tau_ns / 1e9
    off = int(round((tau_s - 420.0 / 299_792.458) * 64_000)) % 64_000
    stop = _stop(plan_and_txs, dict(physics_settings, coherent_seconds=30,
                                    tdma_offset_samples={"TX1": off}), tmp_path)
    spots = []
    for iv, lab in split_into_intervals(x, start):
        spots += stop.feed(iv, lab).spots
    assert spots
    assert spots[0]["group_range_km"] == pytest.approx(420.0, abs=12.0)


# --- the TDMA offset names the transmitter's sweep start after the UTC second --

@pytest.mark.parametrize("rate", [-25_733.913, 25_733.913])
@pytest.mark.parametrize("cpi_phase_s", [0.25, 0.6])
@pytest.mark.parametrize("tx_s", [0.2, 0.7])
def test_a_configured_offset_reads_the_path_whatever_the_cpi_phase(
        tmp_path, physics_settings, split_into_intervals, rate, cpi_phase_s, tx_s):
    """A transmitter configured at tx_s reads its true group range whether
    the CPI opens 0.25 s or 0.6 s after the second (two daemon runs)."""
    from conftest import phys_cpi
    start_ns = 1_800_000_000 * 10**9 + int(round(cpi_phase_s * 1e9))
    x, start = phys_cpi(rate=rate, tx_s=tx_s, start_ns=start_ns)
    stop = _stop(_plan(rate), dict(
        physics_settings, tdma_offset_samples={"TX1": int(round(tx_s * 64_000))}), tmp_path)
    spots = []
    for iv, lab in split_into_intervals(x, start):
        spots += stop.feed(iv, lab).spots
    assert spots, "no physics row: the replica missed the transmitter"
    assert spots[0]["group_range_km"] == pytest.approx(420.0, abs=15.0)
