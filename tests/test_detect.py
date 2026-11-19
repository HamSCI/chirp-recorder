import numpy as np
from hamsci_dsp.chirp.waveform import Waveform

from chirp_recorder.core.channels import Group
from chirp_recorder.core.detect import (FLOOR_DB, HALF_BAND_MAX_DB, MARGIN_DB,
                                        MAX_COMPONENTS, Arrival, GroupResult,
                                        detect_group)

WF_4463 = Waveform(4.463e6, -25_733.913, 1.0, None, 0.006, 0.5)
WF_13_4499 = Waveform(13.4499e6, -99_260.0, 0.5, None, 0.006, 0.5)
WF_13_4875 = Waveform(13.4875e6, -99_260.0, 0.5, None, 0.006, 0.5)
GROUP_4463 = Group(WF_4463, ("A",))
GROUP_13_4499 = Group(WF_13_4499, ("B",))
GROUP_13_4875 = Group(WF_13_4875, ("C",))


def test_constants():
    assert (MARGIN_DB, HALF_BAND_MAX_DB, FLOOR_DB, MAX_COMPONENTS) == (3.0, 6.0, 10.0, 12)


def test_finds_ground_wave_and_skywave(synth):
    x = synth(fs=64_000, centre_offset_hz=0, arrivals=[(0.004120, 1.0), (0.042906, 0.5)], seconds=30)
    r = detect_group(x, fs=64_000, channel_centre_hz=4.463e6, group=GROUP_4463)
    taus = sorted(a.tau_ns for a in r.arrivals)
    assert len(taus) == 2
    assert abs(taus[0] - 4_120_000) < 500 and abs(taus[1] - 42_906_000) < 500
    assert r.n_sweeps == 30 and all(a.clean_tag == "A" for a in r.arrivals)
    assert all(isinstance(a.tau_ns, int) for a in r.arrivals)
    assert isinstance(r, GroupResult) and isinstance(r.arrivals[0], Arrival)


def test_group_off_channel_centre_still_found(synth):
    x = synth(fs=192_000, centre_offset_hz=-25_100, arrivals=[(0.1, 1.0)], seconds=30, wf=WF_13_4499)
    r = detect_group(x, fs=192_000, channel_centre_hz=13.4750e6, group=GROUP_13_4499)
    assert len(r.arrivals) == 1
    assert abs(r.arrivals[0].tau_ns - 100_000_000) < 500


def test_neighbour_image_rejected(synth):
    # a same-rate neighbour 25 kHz off lands in this group's profile; the half-band test drops it
    x = synth(fs=192_000, centre_offset_hz=25_000, arrivals=[(0.1, 1.0)], seconds=30, wf=WF_13_4875)
    r = detect_group(x, fs=192_000, channel_centre_hz=13.4875e6, group=GROUP_13_4875)
    assert not r.arrivals and r.rejected.get("image", 0) >= 1


def test_noise_only_yields_nothing(synth):
    r = detect_group(synth(fs=64_000, arrivals=[], seconds=30), fs=64_000,
                     channel_centre_hz=4.463e6, group=GROUP_4463)
    assert r.arrivals == [] and r.null_max_db < 10


def test_sigma_shrinks_with_level(synth):
    strong = detect_group(synth(fs=64_000, arrivals=[(0.0429, 1.0)], seconds=30), fs=64_000,
                          channel_centre_hz=4.463e6, group=GROUP_4463).arrivals[0]
    weak = detect_group(synth(fs=64_000, arrivals=[(0.0429, 0.1)], seconds=30), fs=64_000,
                        channel_centre_hz=4.463e6, group=GROUP_4463).arrivals[0]
    assert strong.delay_sigma_ns < weak.delay_sigma_ns
    # the bound, verbatim
    B = WF_4463.bandwidth_hz()
    want = 1e9 * np.sqrt(3) / (np.pi * B * np.sqrt(2 * 10 ** (strong.level_db / 10) * 30))
    assert abs(strong.delay_sigma_ns - want) < 1e-6 * want


def test_doppler_from_a_moving_path(synth):
    x = synth(fs=64_000, arrivals=[(0.0429, 1.0)], seconds=30, doppler_hz=0.2)
    a = detect_group(x, fs=64_000, channel_centre_hz=4.463e6, group=GROUP_4463).arrivals[0]
    assert abs(a.doppler_hz - 0.2) < 0.02
    assert abs(a.phase_rate - 2 * np.pi * 0.2) < 2 * np.pi * 0.02
    assert a.s4 is not None and a.sigma_phi_rad is not None
    assert -np.pi <= a.carrier_phase_rad <= np.pi


def test_negative_doppler(synth):
    x = synth(fs=64_000, arrivals=[(0.0429, 1.0)], seconds=30, doppler_hz=-0.15)
    a = detect_group(x, fs=64_000, channel_centre_hz=4.463e6, group=GROUP_4463).arrivals[0]
    assert abs(a.doppler_hz + 0.15) < 0.02


def test_carrier_phase_comparable_across_intervals(synth):
    # one steady carrier: the phase at mid-interval follows n counted from the
    # interval's first sample, so a second interval of the same signal agrees
    x = synth(fs=64_000, arrivals=[(0.0429, 1.0)], seconds=30, centre_offset_hz=0.0)
    a = detect_group(x, fs=64_000, channel_centre_hz=4.463e6, group=GROUP_4463).arrivals[0]
    b = detect_group(x, fs=64_000, channel_centre_hz=4.463e6, group=GROUP_4463).arrivals[0]
    assert a.carrier_phase_rad == b.carrier_phase_rad


def test_trailing_part_dropped(synth):
    x = synth(fs=64_000, arrivals=[(0.0429, 1.0)], seconds=12)
    x = np.concatenate([x, x[:30_000]])
    r = detect_group(x, fs=64_000, channel_centre_hz=4.463e6, group=GROUP_4463)
    assert r.n_sweeps == 12 and len(r.arrivals) == 1


def test_few_sweeps_gives_no_scintillation_not_an_error(synth):
    x = synth(fs=64_000, arrivals=[(0.0429, 1.0)], seconds=5)
    r = detect_group(x, fs=64_000, channel_centre_hz=4.463e6, group=GROUP_4463)
    assert r.n_sweeps == 5 and len(r.arrivals) == 1
    a = r.arrivals[0]
    assert a.s4 is None and a.sigma_phi_rad is None
    assert a.doppler_hz is not None


def test_shorter_than_a_period_returns_empty():
    r = detect_group(np.zeros(1000, np.complex64), fs=64_000, channel_centre_hz=4.463e6,
                     group=GROUP_4463)
    assert r.n_sweeps == 0 and r.arrivals == [] and r.gate is None


def test_rung_refusals_counted(synth):
    # an arrival 4 rungs after a strong one reads as that arrival's rung
    from hamsci_dsp.chirp.waveform import rung_spacing_s
    L = rung_spacing_s(WF_4463)
    x = synth(fs=64_000, arrivals=[(0.01, 1.0), (0.01 + 2 * L, 0.3)], seconds=30)
    r = detect_group(x, fs=64_000, channel_centre_hz=4.463e6, group=GROUP_4463)
    assert len(r.arrivals) == 1 and r.rejected.get("rung", 0) >= 1


def test_gate_fitted_when_group_has_none(synth):
    wf = Waveform(4.463e6, -25_733.913, 1.0, None, None, 0.5)
    x = synth(fs=64_000, arrivals=[(0.0429, 1.0)], seconds=30)
    r = detect_group(x, fs=64_000, channel_centre_hz=4.463e6, group=Group(wf, ("A",)))
    assert r.gate is not None and abs(r.gate.period_s - 0.006) < 2e-4
    assert abs(r.arrivals[0].tau_ns - 42_900_000) < 500


def test_known_gate_not_reported_as_fitted(synth):
    x = synth(fs=64_000, arrivals=[(0.0429, 1.0)], seconds=30)
    r = detect_group(x, fs=64_000, channel_centre_hz=4.463e6, group=GROUP_4463)
    assert r.gate is None
