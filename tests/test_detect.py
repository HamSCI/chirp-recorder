import numpy as np
import pytest
from hamsci_dsp.chirp.waveform import Waveform

from chirp_recorder.core.channels import Group
from chirp_recorder.core.detect import (FLOOR_DB, HALF_BAND_MAX_DB, MARGIN_DB,
                                        MAX_COMPONENTS, Arrival, GroupResult,
                                        detect_group)

WF_4463 = Waveform(4.463e6, -25_733.913, 1.0, None, 1945.6e-6, 0.5)
WF_13_4499 = Waveform(13.4499e6, -99_260.0, 0.5, None, 972.8e-6, 0.5)
WF_13_4875 = Waveform(13.4875e6, -99_260.0, 0.5, None, 972.8e-6, 0.5)
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


def _carrier(x, shift_hz_unused=None):
    return detect_group(x, fs=64_000, channel_centre_hz=4.463e6, group=GROUP_4463).arrivals[0]


def test_carrier_phase_of_a_steady_carrier_agrees_across_two_intervals(synth):
    # two different 30 s intervals of one 60 s record; the signal sits at the
    # mix frequency (shift 0), so a restarted mix changes nothing
    x = synth(fs=64_000, arrivals=[(0.0429, 1.0)], seconds=60)
    half = x.size // 2
    a, b = _carrier(x[:half]), _carrier(x[half:])
    d = (a.carrier_phase_rad - b.carrier_phase_rad + np.pi) % (2 * np.pi) - np.pi
    assert abs(d) < 0.05


def test_carrier_phase_across_intervals_needs_whole_cycles_of_shift(synth):
    # Documented limit: the mix restarts at each interval's first sample, so
    # two intervals agree only when shift_hz * interval_s is a whole number of
    # cycles.  Here the signal sits 0.01 Hz off the group centre: 0.3 cycle
    # over 30 s, so the second interval's phase differs by 2*pi*0.3.
    x = synth(fs=64_000, arrivals=[(0.0429, 1.0)], seconds=60, centre_offset_hz=0.01)
    half = x.size // 2
    grp = Group(WF_4463, ("A",))
    a, b = (detect_group(x[s], fs=64_000, channel_centre_hz=4.463e6 - 0.01, group=grp).arrivals[0]
            for s in (slice(0, half), slice(half, None)))
    d = (b.carrier_phase_rad - a.carrier_phase_rad + np.pi) % (2 * np.pi) - np.pi
    want = (2 * np.pi * 0.01 * 30 + np.pi) % (2 * np.pi) - np.pi
    assert abs(d - want) < 0.1 or abs(abs(d - want) - 2 * np.pi) < 0.1


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


def _two(synth, t_sky, amp=0.5, wf=WF_4463):
    x = synth(fs=64_000, arrivals=[(0.00412, 1.0), (t_sky, amp)], seconds=30, wf=wf)
    return detect_group(x, fs=64_000, channel_centre_hz=4.463e6, group=Group(wf, ("A",)))


def test_sky_wave_on_an_odd_rung_is_refused_and_counted(synth):
    # Limitation (tag_ladder, hamsci-dsp): a real arrival on a ladder rung
    # cannot be told from the rung.  The refusal counts under "rung" and the
    # arrival is never kept with a wrong delay.
    from hamsci_dsp.chirp.waveform import rung_spacing_s
    t = 0.00412 + 3 * rung_spacing_s(WF_4463)
    r = _two(synth, t)
    assert r.rejected.get("rung", 0) >= 1
    assert all(abs(a.tau_ns - round(t * 1e9)) > 1_000_000 for a in r.arrivals)
    assert [a.tau_ns for a in r.arrivals if abs(a.tau_ns - 4_120_000) < 500]


def test_coherent_overlap_near_rung_five_is_flagged(synth):
    # 5 rungs past the ground wave is beyond tag_ladder's range (4), so the
    # sky wave stays "A".  Its delay carries a bias that delay_sigma_ns does
    # not cover; overlap_flag says so.
    from hamsci_dsp.chirp.waveform import rung_spacing_s
    r = _two(synth, 0.00412 + 5 * rung_spacing_s(WF_4463))
    sky = max(r.arrivals, key=lambda a: a.tau_ns)
    ground = min(r.arrivals, key=lambda a: a.tau_ns)
    assert sky.overlap_flag is True and ground.overlap_flag is False
    assert r.overlap_flagged == 1


def test_isolated_arrivals_carry_no_overlap_flag(synth):
    r = _two(synth, 0.042906)
    assert len(r.arrivals) == 2 and not any(a.overlap_flag for a in r.arrivals)
    assert r.overlap_flagged == 0


def test_neighbours_inside_two_cells_flag_each_other(synth):
    # 2 cells = 2 / B = 78 us at 25.7 kHz of sweep; two arrivals 60 us apart
    r = _two(synth, 0.00412 + 60e-6, amp=0.7)
    assert r.arrivals and all(a.overlap_flag for a in r.arrivals)


@pytest.mark.xfail(reason="coherent ground + sky 5 rungs apart: CLEAN fits the mean power "
                          "profile, the cross term does not average away, and the sky delay "
                          "errs about 1 us (hamsci-dsp; flagged by overlap_flag)", strict=False)
def test_coherent_overlap_near_rung_five_timing(synth):
    from hamsci_dsp.chirp.waveform import rung_spacing_s
    t = 0.00412 + 5 * rung_spacing_s(WF_4463)
    sky = max(_two(synth, t).arrivals, key=lambda a: a.tau_ns)
    assert abs(sky.tau_ns - round(t * 1e9)) < 500


def test_arrival_defaults():
    a = Arrival(5, 1.0, 20.0, 5.0, 0.1, "A", None, None, None, None, None)
    assert a.overlap_flag is False and a.tau_uncorrected_ns == 5


# --- range-Doppler coupling ---------------------------------------------------
# A Doppler shift f turns k*(t - tau) into k*(t - (tau - f/k)) + 0: the matched
# filter peaks at tau - f/k.  detect corrects by +f/k (k signed).  The test
# true delay is chosen so the delay sits on a whole sample: CLEAN's
# parabolic refine has no bias there (it biases a delay by up to about 1.5 us
# between samples).
import pytest  # noqa: E402
from dataclasses import replace as _replace  # noqa: E402


@pytest.mark.parametrize("rate", [-25_733.913, 25_733.913])
@pytest.mark.parametrize("f", [0.2, -0.2])
def test_doppler_range_coupling_is_corrected(synth, rate, f):
    wf = _replace(WF_4463, rate_hz_per_s=rate)
    shift = -f / rate                     # the peak moves by this many seconds
    tau = 2752 / 64_000                     # on a whole sample
    x = synth(fs=64_000, arrivals=[(tau, 1.0)], seconds=30, wf=wf, doppler_hz=f)
    a = detect_group(x, fs=64_000, channel_centre_hz=4.463e6,
                     group=Group(wf, ("A",))).arrivals[0]
    assert abs(a.doppler_hz - f) < 0.02
    assert abs(a.tau_ns - round(tau * 1e9)) < 500
    # uncorrected is off by about f/k: CLEAN's template ignores Doppler, so it
    # lands within 2 us of the exact coupling
    assert abs((a.tau_uncorrected_ns - round(tau * 1e9)) - shift * 1e9) < 2_000
    assert not a.overlap_flag


def test_doppler_correction_at_an_arbitrary_fraction_is_within_two_us(synth):
    # CLEAN's parabolic peak refine biases a delay by up to about 1.5 us,
    # depending on its position between samples (hamsci-dsp); the correction
    # inherits it.
    x = synth(fs=64_000, arrivals=[(0.0429, 1.0)], seconds=30, doppler_hz=0.2)
    a = detect_group(x, fs=64_000, channel_centre_hz=4.463e6, group=GROUP_4463).arrivals[0]
    assert abs(a.tau_ns - 42_900_000) < 2_000
    assert a.tau_uncorrected_ns - 42_900_000 > 5_000


def test_step_doppler_does_not_couple_into_the_raw_delay(synth):
    x = synth(fs=64_000, arrivals=[(0.043, 1.0)], seconds=30, doppler_hz=0.2, doppler_mode="step")
    a = detect_group(x, fs=64_000, channel_centre_hz=4.463e6, group=GROUP_4463).arrivals[0]
    assert abs(a.tau_uncorrected_ns - 43_000_000) < 500
    # a per-sweep phase step is not a frequency shift: the raw delay is unbiased,
    # so only the uncorrected value is meaningful in this mode


def test_gate_fitted_when_group_has_none(synth):
    wf = Waveform(4.463e6, -25_733.913, 1.0, None, None, 0.5)
    x = synth(fs=64_000, arrivals=[(0.0429, 1.0)], seconds=30)
    r = detect_group(x, fs=64_000, channel_centre_hz=4.463e6, group=Group(wf, ("A",)))
    assert r.gate is not None and abs(r.gate.period_s - 1945.6e-6) < 2e-5
    assert abs(r.arrivals[0].tau_ns - 42_900_000) < 500


def test_known_gate_not_reported_as_fitted(synth):
    x = synth(fs=64_000, arrivals=[(0.0429, 1.0)], seconds=30)
    r = detect_group(x, fs=64_000, channel_centre_hz=4.463e6, group=GROUP_4463)
    assert r.gate is None
