"""Rows for one group.  The `result` fixture holds a 1 s period group, so the
delay fold is modulo 1e9 ns."""
import pytest
from hamsci_dsp import sigmondb_rows as sr
from hamsci_dsp.chirp.waveform import Waveform

from chirp_recorder.core.channels import Group
from chirp_recorder.core.detect import Arrival, GroupResult
from chirp_recorder.core.rows import rows_for_group
from chirp_recorder.core.timing import Label

KW = dict(input_epoch_gps_s=1477111949.31, interval_s=30.0)
CALL = dict(reporter_id="X", radiod="r", channel="c", duration_s=30.0,
            detector_version="v", position_source="precise")


@pytest.fixture
def result():
    wf = Waveform(4.463e6, -25_733.913, 1.0, None, 1945.6e-6, 0.5)
    arr = Arrival(tau_ns=123_456_789, delay_sigma_ns=40.5, level_db=22.0, margin_db=9.0,
                  half_band_db=1.5, clean_tag="A", doppler_hz=0.2, phase_rate=0.1,
                  carrier_phase_rad=0.3, s4=0.1, sigma_phi_rad=0.05, overlap_flag=True,
                  tau_uncorrected_ns=123_456_000, doppler_ambiguous=False)
    return GroupResult(Group(wf, ("A", "B")), 30, -3.0, -9.0, None, [arr],
                       {"rung": 2, "image": 1, "null": 4})


@pytest.fixture
def t6_label():
    return Label(utc_ns=1_800_000_000_987_654_321, delay_basis="t6", t_level="T6",
                 provenance=sr.provenance("T6", "epoch-7", u_epoch_ns=230.0,
                                          a_level_provenance="measured"),
                 engineering={"delay_basis": "t6", "t_level": "T6"})


@pytest.fixture
def authority_label():
    return Label(utc_ns=1_800_000_000_987_654_321, delay_basis="authority", t_level="T3",
                 provenance=sr.provenance(None),
                 engineering={"delay_basis": "authority", "t_level": "T3",
                              "authority_sigma_ns": 2500, "chain_delay_ns": 18700})


def test_columns_exactly_the_schema(result, t6_label):
    cap, arrs = rows_for_group(result, label=t6_label, **{**CALL, "detector_version": "chirp-recorder/0.1.0"}, **KW)
    assert tuple(cap) == sr.CAPTURE_COLUMNS and all(tuple(a) == sr.ARRIVAL_COLUMNS for a in arrs)
    assert cap["engineering"]["gate_phase_us"] == "0" and "gate_period_us" in cap["engineering"]
    assert cap["centre_hz"] == 4.463e6 and cap["expected"] == {"A": 1.0, "B": 1.0}
    assert cap["n_expected"] == 2 and cap["n_found"] == 1 and cap["n_sweeps"] == 30
    assert cap["time"] == sr.utc_iso_ms(t6_label.utc_ns / 1e9)
    assert (cap["engineering"]["rejected_rung"], cap["engineering"]["rejected_image"],
            cap["engineering"]["rejected_null"]) == ("2", "1", "4")
    assert "rejected_rung" not in arrs[0]["engineering"]
    assert arrs[0]["group_key"] == sr.group_key(4.463e6, -25_733.913, 1.0)
    assert arrs[0]["freq_hz"] == 4.463e6 and arrs[0]["detector_version"] == "chirp-recorder/0.1.0"


def test_delay_is_phase_from_utc_epoch(result, t6_label):
    _, arrs = rows_for_group(result, label=t6_label, **CALL, **KW)
    a0 = result.arrivals[0]
    assert arrs[0]["delay_ns"] == (t6_label.utc_ns + a0.tau_ns) % 1_000_000_000
    assert isinstance(arrs[0]["delay_ns"], int) and arrs[0]["tx_id"] == ""


def test_sigmas_stay_apart(result, t6_label):
    _, arrs = rows_for_group(result, label=t6_label, **CALL, **KW)
    assert arrs[0]["delay_sigma_ns"] == result.arrivals[0].delay_sigma_ns
    assert arrs[0]["u_epoch_ns"] == t6_label.provenance["u_epoch_ns"]


def test_authority_rows_carry_null_block(result, authority_label):
    cap, arrs = rows_for_group(result, label=authority_label, **CALL, **KW)
    assert arrs[0]["u_epoch_ns"] is None and arrs[0]["engineering"]["delay_basis"] == "authority"
    assert "authority_sigma_ns" in cap["engineering"]
    assert cap["engineering"]["chain_delay_ns"] == "18700"


def test_engineering_carries_arrival_fields_and_position_source(result, t6_label):
    cap, arrs = rows_for_group(result, label=t6_label, **{**CALL, "position_source": "grid_centre"}, **KW)
    e = arrs[0]["engineering"]
    assert e["tau_uncorrected_ns"] == "123456000" and e["overlap_flag"] == "1"
    assert e["doppler_ambiguous"] == "0"
    assert e["position_source"] == cap["engineering"]["position_source"] == "grid_centre"
    assert e["radiod"] == "r" and e["channel"] == "c" and e["interval_s"] == "30"
    assert e["sweep_repetition_hz"] == "1" and e["input_epoch_gps_s"] == "1477111949.31"
    assert all(isinstance(v, str) for v in e.values())
    assert arrs[0]["delay_ns"] == (t6_label.utc_ns + result.arrivals[0].tau_ns) % 10**9


def test_gate_period_fitted_or_catalogued(result, t6_label):
    cap, _ = rows_for_group(result, label=t6_label, **CALL, **KW)
    assert float(cap["engineering"]["gate_period_us"]) == pytest.approx(1945.6)


def test_radiod_basis_label_writes_empty_strings_not_None(result):
    lab = Label(utc_ns=1_800_000_000_987_654_321, delay_basis="radiod", t_level=None,
                provenance=sr.provenance(None),
                engineering={"delay_basis": "radiod", "t_level": None,
                             "authority_sigma_ns": None, "chain_delay_ns": None})
    cap, arrs = rows_for_group(result, label=lab, **CALL, **KW)
    for e in (cap["engineering"], arrs[0]["engineering"]):
        assert e["t_level"] == "" and e["authority_sigma_ns"] == "" and e["chain_delay_ns"] == ""
        assert "None" not in e.values()


def test_doppler_quality_travels_in_engineering(result, t6_label):
    """Task 18 fix round 1: the gate's inputs and verdicts ride with each row."""
    from dataclasses import replace
    a = replace(result.arrivals[0], doppler_hz=None, phase_rate=None, doppler_coherence=0.12,
                doppler_sigma_hz=0.011, doppler_low_coherence=True, doppler_clipped=False,
                doppler_raw_hz=-0.223)
    _, (row,) = rows_for_group(replace(result, arrivals=[a]), label=t6_label, **CALL, **KW)
    e = row["engineering"]
    assert row["doppler_hz"] is None
    assert (e["doppler_coherence"], e["doppler_sigma_hz"], e["doppler_low_coherence"],
            e["doppler_clipped"], e["doppler_raw_hz"]) == ("0.12", "0.011", "1", "0", "-0.223")
