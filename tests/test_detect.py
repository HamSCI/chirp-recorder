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
    # the bound, verbatim, plus the Doppler correction's sigma_f/|k| (fix round 1)
    B = WF_4463.bandwidth_hz()
    want = 1e9 * np.sqrt(3) / (np.pi * B * np.sqrt(2 * 10 ** (strong.level_db / 10) * 30))
    want = np.hypot(want, strong.doppler_sigma_hz / abs(WF_4463.rate_hz_per_s) * 1e9)
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
    # the coherent pair biases both: the ground wave sees the sky's cross term too
    assert sky.overlap_flag is True and ground.overlap_flag is True
    assert r.overlap_flagged == 2


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


def test_doppler_correction_at_an_arbitrary_fraction_is_within_500_ns(synth):
    # CLEAN's parabolic refine biases a delay by up to ~190 ns at 64 kS/s,
    # depending on its position between samples; delay_sigma_ns carries it.
    x = synth(fs=64_000, arrivals=[(0.0429, 1.0)], seconds=30, doppler_hz=0.2)
    a = detect_group(x, fs=64_000, channel_centre_hz=4.463e6, group=GROUP_4463).arrivals[0]
    assert abs(a.tau_ns - 42_900_000) < 500
    assert a.tau_uncorrected_ns - 42_900_000 > 5_000


@pytest.mark.parametrize("seed", [7, 8, 9])
def test_weaker_arrival_three_cells_off_is_timed_or_flagged(synth, seed):
    # ground 1.0 + sky 0.3, 150 us (3.9 cells) apart: a coherent neighbour
    # biases the sky delay by microseconds; it must carry overlap_flag
    t = 0.00412 + 150e-6
    x = synth(fs=64_000, arrivals=[(0.00412, 1.0), (t, 0.3)], seconds=30, seed=seed)
    r = detect_group(x, fs=64_000, channel_centre_hz=4.463e6, group=GROUP_4463)
    sky = [a for a in r.arrivals if a.tau_ns > 4_200_000]
    assert sky and all(a.overlap_flag or abs(a.tau_ns - round(t * 1e9)) < 500 for a in sky)


def test_doppler_beyond_the_alias_limit_is_not_corrected(synth):
    # 0.55 Hz reads as -0.45 Hz on a 1 s slow-time grid; the correction would
    # err by one cell (38.9 us), so detect skips it and says so
    x = synth(fs=64_000, arrivals=[(0.043, 1.0)], seconds=30, doppler_hz=0.55)
    a = detect_group(x, fs=64_000, channel_centre_hz=4.463e6, group=GROUP_4463).arrivals[0]
    assert a.doppler_ambiguous is True
    # no coupling correction (that would move it by 22 us); only the refine
    assert abs(a.tau_ns - a.tau_uncorrected_ns) < 500
    assert abs(a.tau_ns - 43_000_000) > 10_000


def test_ordinary_doppler_is_not_ambiguous(synth):
    x = synth(fs=64_000, arrivals=[(0.043, 1.0)], seconds=30, doppler_hz=0.2)
    a = detect_group(x, fs=64_000, channel_centre_hz=4.463e6, group=GROUP_4463).arrivals[0]
    assert a.doppler_ambiguous is False


def test_rederiving_the_delay_stays_small_in_memory():
    # at 192 kS/s, 0.5 s period, 30 s: the interval is 92 MB as complex128;
    # the Doppler-removed profile must not hold more than 50 MB beyond it
    import tracemalloc

    from chirp_recorder.core.detect import _derotated_profile
    rng = np.random.default_rng(1)
    sweeps = (rng.standard_normal((60, 96_000)) + 1j * rng.standard_normal((60, 96_000)))
    tracemalloc.start()
    _derotated_profile(sweeps, WF_13_4875, 192_000, 0.2)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 50e6, peak


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


# --- refine bias and differential Doppler (fix round 3) ------------------------
@pytest.mark.parametrize("wf,fs", [
    (WF_4463, 48_000), (WF_4463, 64_000), (WF_4463, 96_000), (WF_4463, 192_000),
    (WF_13_4875, 64_000), (WF_13_4875, 96_000), (WF_13_4875, 192_000)])
def test_refine_bias_below_50_ns_at_every_fs_over_b(wf, fs):
    # fs/B = 1.29 (13 MHz at 64 kS/s) .. 7.46 (4.463 MHz at 192 kS/s)
    from hamsci_dsp.chirp.waveform import gated

    from chirp_recorder.core.detect import _refined_tau
    worst = 0.0
    for fr in np.arange(0.0, 1.0, 0.1):
        tau = (int(0.043 * fs) + fr) / fs
        sweeps = np.tile(gated(wf, fs, tau), (2, 1))
        worst = max(worst, abs(_refined_tau(sweeps, wf, fs, tau, 0.0) - tau) * 1e9)
    assert worst < 50, (wf.bandwidth_hz(), fs, worst)


@pytest.mark.parametrize("sep_s", [1e-3, 2e-3, 3e-3])
@pytest.mark.parametrize("seed", [11, 12])
def test_pair_with_differential_doppler_is_not_flagged(synth, sep_s, seed):
    # a sky wave 15 dB down with 0.1 Hz differential Doppler: the cross term
    # averages out over 30 s, so the predicted bias stays under delay_sigma
    t = 0.00412 + sep_s
    x = synth(fs=64_000, arrivals=[(0.00412, 1.0, 0.0), (t, 10 ** (-15 / 20), 0.1)],
              seconds=30, seed=seed)
    r = detect_group(x, fs=64_000, channel_centre_hz=4.463e6, group=GROUP_4463)
    assert len(r.arrivals) == 2 and not any(a.overlap_flag for a in r.arrivals)


def test_coherent_pair_twenty_db_down_one_ms_away_is_flagged(synth):
    t = 0.00412 + 1e-3
    x = synth(fs=64_000, arrivals=[(0.00412, 1.0), (t, 0.1)], seconds=30, seed=11)
    r = detect_group(x, fs=64_000, channel_centre_hz=4.463e6, group=GROUP_4463)
    sky = [a for a in r.arrivals if a.tau_ns > 4_800_000]
    assert sky and all(a.overlap_flag for a in sky)


# --- Task 18 fix round 1: Doppler quality gate, physics bound, isolated refine --
from hamsci_dsp.chirp.waveform import gated as _gated  # noqa: E402

K_4463 = -25_733.913


def _incoherent(wf, fs, arrivals, m=30, seed=3):
    """Sweeps whose arrivals carry a fresh random carrier phase every sweep
    (no coherent Doppler at all), as (tau_s, amplitude, coherent?) triples."""
    rng = np.random.default_rng(seed)
    sw = []
    for _ in range(m):
        s = np.zeros(int(wf.period_s * fs), complex)
        for tau, amp, coherent in arrivals:
            ph = 1.0 if coherent else np.exp(2j * np.pi * rng.random())
            s += amp * ph * _gated(wf, fs, tau)
        sw.append(s)
    x = np.concatenate(sw)
    x += 0.01 * (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size))
    return x.astype(np.complex64)


def test_an_incoherent_arrival_publishes_no_doppler_and_no_correction():
    """Slow-time coherence under DOPPLER_MIN_COHERENCE: the fitted Doppler has
    nothing to do with one path, so the delay stays uncorrected, the row
    carries no Doppler, and the arrival says why."""
    from chirp_recorder.core.detect import DOPPLER_MIN_COHERENCE
    x = _incoherent(WF_4463, 64_000, [(0.0429, 1.0, False)])
    (a,) = detect_group(x, fs=64_000, channel_centre_hz=4.463e6, group=GROUP_4463).arrivals
    assert a.doppler_low_coherence is True and a.doppler_coherence < DOPPLER_MIN_COHERENCE
    assert a.doppler_hz is None and a.phase_rate is None
    assert abs(a.tau_ns - 42_900_000) < 300
    assert abs(a.tau_ns - a.tau_uncorrected_ns) < 300


def test_a_coherent_doppler_is_kept_and_corrected(synth):
    x = synth(fs=64_000, arrivals=[(0.0429, 1.0)], seconds=30, doppler_hz=0.2)
    (a,) = detect_group(x, fs=64_000, channel_centre_hz=4.463e6, group=GROUP_4463).arrivals
    assert a.doppler_low_coherence is False and a.doppler_coherence > 0.9
    assert abs(a.doppler_hz - 0.2) < 0.02 and abs(a.tau_ns - 42_900_000) < 500


def test_delay_sigma_carries_the_doppler_sigma_over_k(synth):
    x = synth(fs=64_000, arrivals=[(0.0429, 0.3)], seconds=30, doppler_hz=0.2, noise=3.0)
    (a,) = detect_group(x, fs=64_000, channel_centre_hz=4.463e6, group=GROUP_4463).arrivals
    assert a.doppler_sigma_hz is not None and a.doppler_sigma_hz > 0
    snr = 10 ** (a.level_db / 10)
    stat = 1e9 * np.sqrt(3) / (np.pi * WF_4463.bandwidth_hz() * np.sqrt(2 * snr * 30))
    want = np.hypot(stat, a.doppler_sigma_hz / abs(K_4463) * 1e9)
    assert a.delay_sigma_ns == pytest.approx(want, rel=1e-6)


def test_the_correction_is_clipped_to_the_physics_bound():
    """|g| <= 1: the peak of a Doppler-shifted arrival moves by at most |f/k|
    from its Doppler-free delay, whatever the row split.  The bound runs from
    tau0, the same isolated refine with no Doppler removed, so the estimators'
    own gap cancels.  Past it the correction landed on another path: publish
    tau0 (a delay an estimator measured), never tau0 +- bound, and say so."""
    from chirp_recorder.core.detect import _clip_correction
    tau0, f, P = 0.042, -0.223, 1.0
    bound = abs(f / K_4463)                                   # 8.67 us
    assert _clip_correction(tau0 + 44.7e-6, tau0, f, K_4463, P) == (tau0, True)
    tau, clipped = _clip_correction(tau0 - 3e-6, tau0, f, K_4463, P)
    assert not clipped and tau == pytest.approx(tau0 - 3e-6, abs=1e-12)
    assert _clip_correction(tau0 + bound + 0.5e-6, tau0, f, K_4463, P, margin_s=1e-6) \
        == (tau0 + bound + 0.5e-6, False)
    assert _clip_correction(0.9999995 - 20e-6, 0.0000005, f, K_4463, P) == (0.0000005, True)


def test_the_clip_margin_is_three_stat_sigmas_plus_100_ns():
    from chirp_recorder.core.detect import _clip_margin_s
    assert _clip_margin_s(40.0) == pytest.approx(220e-9)
    assert _clip_margin_s(1000.0) == pytest.approx(3.1e-6)


def test_a_clipped_correction_publishes_tau0_and_keeps_its_doppler(synth, monkeypatch):
    import chirp_recorder.core.detect as D
    real = D._doppler_free_tau
    monkeypatch.setattr(D, "_doppler_free_tau",
                        lambda sw, wf, fs, tau, dop: (real(sw, wf, fs, tau, dop) + 50e-6) % 1.0)
    x = synth(fs=64_000, arrivals=[(0.0429, 1.0)], seconds=30, doppler_hz=0.2)
    (a,) = D.detect_group(x, fs=64_000, channel_centre_hz=4.463e6, group=GROUP_4463).arrivals
    assert a.doppler_clipped is True and abs(a.doppler_hz - 0.2) < 0.02
    assert a.tau_ns == a.tau_zero_doppler_ns
    snr = 10 ** (a.level_db / 10)
    stat = 1e9 * np.sqrt(3) / (np.pi * WF_4463.bandwidth_hz() * np.sqrt(2 * snr * 30))
    # the published delay keeps g * f/k of coupling, |g| <= 1
    assert a.delay_sigma_ns == pytest.approx(np.hypot(stat, abs(a.doppler_hz / K_4463) * 1e9))


def test_a_gated_off_row_publishes_tau0_with_the_coupling_in_its_sigma():
    """Below the coherence gate the delay keeps up to |f/k| of coupling, and
    the incoherent fit still sizes f fairly: sigma = hypot(stat, |f_raw/k|).
    The delay comes from the same isolated refine as every other row."""
    x = _incoherent(WF_4463, 64_000, [(0.0429, 1.0, False)])
    (a,) = detect_group(x, fs=64_000, channel_centre_hz=4.463e6, group=GROUP_4463).arrivals
    assert a.doppler_low_coherence and a.doppler_hz is None and a.doppler_raw_hz is not None
    assert a.tau_ns == a.tau_zero_doppler_ns
    snr = 10 ** (a.level_db / 10)
    stat = 1e9 * np.sqrt(3) / (np.pi * WF_4463.bandwidth_hz() * np.sqrt(2 * snr * 30))
    assert a.delay_sigma_ns == pytest.approx(np.hypot(stat, abs(a.doppler_raw_hz / K_4463) * 1e9))


def test_every_row_carries_tau0(synth):
    x = synth(fs=64_000, arrivals=[(0.0429, 1.0)], seconds=30, doppler_hz=0.2)
    (a,) = detect_group(x, fs=64_000, channel_centre_hz=4.463e6, group=GROUP_4463).arrivals
    assert isinstance(a.tau_zero_doppler_ns, int)
    # tau0 keeps the coupling the correction removes (g ~ 0.95 at this split)
    assert abs(a.tau_zero_doppler_ns - a.tau_uncorrected_ns) < 500


@pytest.mark.parametrize("sep", [6.3, 7.3, 9.3])
def test_the_refine_reads_the_arrival_with_its_neighbours_removed(sep):
    """A strong neighbour's skirt drags a raw zero-padded refine by 1.5-6 us
    (sep 6.3-9.3 samples, neighbour 14 dB up).  The weak arrival's phase turns
    whole cycles over the sweeps, so the cross term averages to zero and only
    the skirt remains.  Refining with CLEAN's other components removed
    recovers the weak arrival's delay."""
    from chirp_recorder.core.detect import _refined_tau
    fs, t1 = 64_000, 0.00412
    t2 = t1 + sep / fs
    g1, g2 = _gated(WF_4463, fs, t1), _gated(WF_4463, fs, t2)
    sw = np.array([g1 + 0.2 * np.exp(2j * np.pi * k / 60) * g2 for k in range(60)])
    from hamsci_dsp.chirp.clean import matched_profile
    peak1 = float(matched_profile(g1[None, :], WF_4463, fs).max())   # its CLEAN amplitude
    assert abs(_refined_tau(sw, WF_4463, fs, t2, 0.0) - t2) * 1e9 > 1_000   # the old reading
    got = _refined_tau(sw, WF_4463, fs, t2, 0.0, others=((t1, peak1),))
    assert abs(got - t2) * 1e9 < 200


def test_a_weak_neighbour_is_timed_with_the_strong_one_removed(synth):
    """Weak path 14 dB down, 7.3 samples (114 us) from a strong one, 2/60 Hz
    differential Doppler over 60 sweeps: the cross term averages out, and
    the delay must come from CLEAN's fitted arrival, not the strong skirt."""
    t2 = 0.00412 + 7.3 / 64_000
    x = synth(fs=64_000, arrivals=[(0.00412, 1.0, 0.0), (t2, 0.2, 2 / 60)], seconds=60,
              noise=0.01)
    r = detect_group(x, fs=64_000, channel_centre_hz=4.463e6, group=GROUP_4463)
    weak = [a for a in r.arrivals if a.tau_uncorrected_ns > 4_160_000]
    assert weak, [a.tau_ns for a in r.arrivals]
    assert abs(weak[0].tau_ns - round((0.00412 + 7.3 / 64_000) * 1e9)) < 300



def test_an_ambiguous_row_carries_the_coupling_in_its_sigma(synth):
    """R2-I1: an ambiguous Doppler (|f| > 0.4/P) skips the correction, so the
    delay keeps g f/k of coupling, and the aliased value hides its size: the
    bound never falls below the alias limit, max(|f_raw|, 0.4/P) / |k|."""
    x = synth(fs=64_000, arrivals=[(0.043, 1.0)], seconds=30, doppler_hz=0.55)
    (a,) = detect_group(x, fs=64_000, channel_centre_hz=4.463e6, group=GROUP_4463).arrivals
    assert a.doppler_ambiguous is True and a.tau_ns == a.tau_zero_doppler_ns
    snr = 10 ** (a.level_db / 10)
    stat = 1e9 * np.sqrt(3) / (np.pi * WF_4463.bandwidth_hz() * np.sqrt(2 * snr * 30))
    coupling = max(abs(a.doppler_raw_hz), 0.4 / WF_4463.period_s) / abs(K_4463) * 1e9
    assert a.delay_sigma_ns == pytest.approx(np.hypot(stat, coupling))
    assert a.delay_sigma_ns > 15_000
