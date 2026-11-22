"""Detection on one interval for one transmitter group (the raw path).

Six steps: mix the group's centre to zero; cut whole sweep periods from
sample 0; compress with the plain-replica matched filter; CLEAN and tag the
ladder; screen with the half-band test and the null margin; measure delay,
delay sigma, Doppler, carrier phase and scintillation per kept arrival.

The shared pieces (matched filter, CLEAN, gate fit, screens, scintillation)
come from hamsci-dsp.  Pure function: no clock reads, no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Optional

import numpy as np
from hamsci_dsp.chirp.clean import clean, matched_profile
from hamsci_dsp.chirp.screen import half_band_db, null_levels
from hamsci_dsp.chirp.clean import tag_ladder
from hamsci_dsp.chirp.waveform import GateFit, fit_gate, gated, replica, samples_per_period
from hamsci_dsp.propagation.scintillation import HF_OBLIQUE, compute_scintillation

from chirp_recorder.core.channels import Group

MARGIN_DB = 3.0
HALF_BAND_MAX_DB = 6.0
FLOOR_DB = 10.0
MAX_COMPONENTS = 12
OVERLAP_CELLS = 2.0     # an overlap lies within this many resolution cells ...
BIAS_CELLS = 1.0        # predicted bias = BIAS_CELLS / B * cross amplitude * coherence; weaker-arrival K: median 0.6, max 1.5 (scripts/calibrate_overlap.py)
DOPPLER_AMBIGUOUS = 0.4  # |doppler| over this / period: slow-time sampling may alias; no correction
REFINE_WINDOW = 64       # samples of the complex matched-filter output around a peak
REFINE_UPSAMPLE = 16     # zero-padding factor of that window
_CHUNK_SAMPLES = 400_000     # samples per block when the interval is re-compressed
RESIDUE_DB = 12.0       # an "m" residue this far under an arrival is CLEAN's own leftover
# Slow-time coherence |sum v e^{-i phi_hat t}| / sum |v| under which the fitted
# Doppler describes no single path (spread night sky wave: 0.03-0.25 on the
# golden captures).  Provisional: Task 18 fix round 1 chose 0.5 from the
# golden night's spread of values, which runs continuously from 0.06 to 1.
DOPPLER_MIN_COHERENCE = 0.5
# A Doppler-shifted arrival's raw peak sits g * (-f/k) from its Doppler-free
# delay, with |g| <= 1 set by where the row split cuts the chirp.  So the
# correction, measured from tau0 (the same isolated refine with no Doppler
# removed), never exceeds |f/k|; past that bound, plus CLIP_SIGMAS
# statistical sigmas and CLIP_FLOOR_NS, it has jumped to another path.
CLIP_SIGMAS = 3.0
CLIP_FLOOR_NS = 100.0
OVERLAP_MAX_RUNG = 6    # ... of another component, or of this rung of a stronger one


@dataclass(frozen=True)
class Arrival:
    tau_ns: int
    delay_sigma_ns: float
    level_db: float
    margin_db: float
    half_band_db: float
    clean_tag: str
    doppler_hz: Optional[float]
    phase_rate: Optional[float]
    carrier_phase_rad: Optional[float]
    s4: Optional[float]
    sigma_phi_rad: Optional[float]
    # --- added after review (fix round 1); defaults keep older callers valid ---
    overlap_flag: bool = False   # see _overlaps: resolved overlaps only; sub-cell multipath biases the delay unflagged
    tau_uncorrected_ns: Optional[int] = None    # None reads as "equal to tau_ns"
    doppler_ambiguous: bool = False   # |doppler| > 0.4/P: delay not corrected
    slow: Optional[tuple] = field(default=None, compare=False, repr=False)   # per-sweep complex series at the arrival (PhaseStore)
    # --- added in Task 18 fix round 1 ---
    doppler_coherence: Optional[float] = None    # slow-time coherence of the fitted Doppler
    doppler_sigma_hz: Optional[float] = None     # the slope fit's own sigma
    doppler_low_coherence: bool = False          # coherence under the gate: no Doppler, no correction
    doppler_clipped: bool = False                # correction clipped to |f/k|
    doppler_raw_hz: Optional[float] = None       # the fitted Doppler, published or not
    tau_zero_doppler_ns: Optional[int] = None    # tau0: isolated refine, no Doppler removed

    def __post_init__(self):
        if self.tau_uncorrected_ns is None:
            object.__setattr__(self, "tau_uncorrected_ns", self.tau_ns)


@dataclass(frozen=True)
class GroupResult:
    group: Group
    n_sweeps: int
    null_max_db: float
    null_median_db: float
    gate: Optional[GateFit]
    arrivals: list
    rejected: dict
    overlap_flagged: int = 0     # kept arrivals whose delay sigma understates the error


def _slow_time(sweeps: np.ndarray, wf, fs: float, tau_s: float) -> np.ndarray:
    """Complex matched-filter value per sweep at the arrival's lag.

    The lag is the strongest bin of the mean profile within two samples of
    ``tau_s``, so a fractional-sample delay still reads its peak.
    """
    n = sweeps.shape[1]
    rfc = np.conj(np.fft.fft(replica(wf, fs)))
    R = np.fft.ifft(np.fft.fft(sweeps, axis=1) * rfc[None, :], axis=1)
    i = int(round(tau_s * fs))
    near = (np.arange(i - 2, i + 3)) % n
    best = near[int(np.argmax(np.mean(np.abs(R[:, near]) ** 2, axis=0)))]
    return R[:, best]


def _doppler(v: np.ndarray, period_s: float):
    """phase_rate (rad/s), doppler_hz, carrier phase at mid-interval, the
    slow-time coherence and the Doppler's sigma (Hz) from the slope fit."""
    m = v.size
    if m < 2:
        return None, None, None, None, None
    ph = np.unwrap(np.angle(v))
    t = np.arange(m) * period_s
    # Least-squares slope over every sweep, not first minus last.  Slow-time
    # sampling aliases |doppler| above 1 / (2 * period).
    sigma_hz = None
    if m >= 4:
        coef, cov = np.polyfit(t, ph, 1, cov=True)
        rate = float(coef[0])
        sigma_hz = float(np.sqrt(max(cov[0, 0], 0.0)) / (2 * np.pi))
    else:
        rate = float(np.polyfit(t, ph, 1)[0])
    t_mid = m * period_s / 2
    z = np.sum(v * np.exp(-1j * rate * (t - t_mid)))
    tot = float(np.sum(np.abs(v)))
    coherence = float(abs(z) / tot) if tot > 0 else 0.0
    return rate, rate / (2 * np.pi), float(np.angle(z)), coherence, sigma_hz


def _derotated_profile(sweeps, wf, fs, dop_hz):
    """Mean matched profile of the interval with ``dop_hz`` removed.

    Works block by block so the temporaries stay a few MB, whatever the rate.
    The phase uses n counted from the interval's first sample.
    """
    m, n = sweeps.shape
    step = max(1, _CHUNK_SAMPLES // n)
    acc = np.zeros(n)
    for a in range(0, m, step):
        blk = sweeps[a:a + step]
        idx = np.arange(a * n, (a + blk.shape[0]) * n)
        rot = np.exp(-2j * np.pi * dop_hz * idx / fs).reshape(blk.shape)
        acc += matched_profile(blk * rot, wf, fs) * blk.shape[0]
    return acc / m


@lru_cache(maxsize=256)
def _template(wf, fs, tau_s):
    """One noiseless gated sweep at tau_s and its plain profile's peak power."""
    g = gated(wf, fs, tau_s)
    return g, float(matched_profile(g[None, :], wf, fs).max())


def _fine_power(sweeps, wf, fs, cols, dop_hz):
    """Mean power over the sweeps of the complex matched-filter output on
    ``cols``, zero-padded REFINE_UPSAMPLE times (band-limited interpolation)."""
    m, n = sweeps.shape
    W, up = cols.size, REFINE_UPSAMPLE
    rfc = np.conj(np.fft.fft(replica(wf, fs)))
    win = np.empty((m, W), complex)
    step = max(1, _CHUNK_SAMPLES // n)
    for a in range(0, m, step):
        blk = sweeps[a:a + step]
        if dop_hz:
            idx = np.arange(a * n, (a + blk.shape[0]) * n)
            blk = blk * np.exp(-2j * np.pi * dop_hz * idx / fs).reshape(blk.shape)
        win[a:a + step] = np.fft.ifft(np.fft.fft(blk, axis=1) * rfc[None, :], axis=1)[:, cols]
    F = np.fft.fft(win, axis=1)
    Fp = np.zeros((m, W * up), complex)
    Fp[:, :W // 2] = F[:, :W // 2]
    Fp[:, -W // 2:] = F[:, W // 2:]
    return np.mean(np.abs(np.fft.ifft(Fp, axis=1)) ** 2, axis=0)


def _refined_tau(sweeps, wf, fs, tau_s, dop_hz, others=()):
    """Delay at the peak near ``tau_s`` of the complex matched-filter output.

    CLEAN refines its peak with a 3-point parabola in log power, which biases
    a delay by up to ~190 ns at 64 kS/s (and 1.8 us at fs/B = 1.3), depending
    on where it falls between samples.  Here the complex output of each sweep,
    with ``dop_hz`` removed, is cut to REFINE_WINDOW samples around the peak,
    zero-padded REFINE_UPSAMPLE times by FFT (the output is band-limited, so
    this interpolates it), averaged in power over the sweeps, and the same
    log-parabola is applied on the fine grid.  Measured bias: under 4 ns
    across fs/B = 1.29 .. 7.46 (tests).  Blocks keep the temporaries small.

    ``others`` holds CLEAN's other components as (tau_s, amplitude), the
    amplitude being the peak power CLEAN's joint fit gave each.  Their
    noiseless responses, interpolated the same way, come off the fine power
    first, so a neighbour's skirt or rung does not drag this peak (Task 18
    fix round 1, I2: the delay comes from CLEAN's fitted arrival, not the raw
    peak).  CLEAN fits power, so a coherent cross term stays (overlap_flag).
    """
    m, n = sweeps.shape
    W, up = REFINE_WINDOW, REFINE_UPSAMPLE
    i0 = int(round(tau_s * fs))
    cols = (np.arange(i0 - W // 2, i0 + W // 2)) % n
    p = _fine_power(sweeps, wf, fs, cols, dop_hz)
    for tau_o, amp in others:
        g, peak = _template(wf, fs, float(tau_o))
        if amp > 0 and peak > 0:
            p = p - (amp / peak) * _fine_power(g[None, :], wf, fs, cols, 0.0)
    c0 = (W // 2) * up                 # fine index of the sample at tau_s
    lo = c0 - 3 * up                   # look within 3 samples only: a neighbour's
    j = lo + int(np.argmax(p[lo:c0 + 3 * up + 1]))   # peak in the window is not ours
    p = np.maximum(p, p[j] * 1e-6)
    la, lb, lc = np.log(p[j - 1]), np.log(p[j]), np.log(p[j + 1])
    den = la - 2 * lb + lc
    jj = j + (0.5 * (la - lc) / den if den < 0 else 0.0)
    return (((i0 - W // 2) + jj / up) / fs) % wf.period_s


def _amplitudes(profile, comps, wf, fs) -> list:
    """CLEAN's joint fit of every component's peak power (profile units)."""
    if not comps:
        return []
    sig = np.asarray(profile, float) - float(np.median(profile))
    A = np.stack([_template_response(wf, fs, c.tau_s) for c in comps], axis=1)
    return [float(a) for a in np.linalg.lstsq(A, sig, rcond=None)[0]]


@lru_cache(maxsize=256)
def _template_response(wf, fs, tau_s):
    g, peak = _template(wf, fs, tau_s)
    r = matched_profile(g[None, :], wf, fs) / peak
    r.setflags(write=False)
    return r


def _others(comps, amps, c):
    """The components to remove before refining ``c``: every other one except
    c's own main-lobe residues and rungs (tagged with c as parent), which hold
    part of c's own power."""
    return tuple((o.tau_s, a) for o, a in zip(comps, amps)
                 if o is not c and a > 0 and getattr(o, "parent_tau_s", None) != c.tau_s)


def _clip_margin_s(sigma_stat_ns: float) -> float:
    """The clip's allowance for noise: CLIP_SIGMAS * sigma_stat + CLIP_FLOOR_NS."""
    return (CLIP_SIGMAS * float(sigma_stat_ns) + CLIP_FLOOR_NS) * 1e-9


def _clip_correction(tau_s, tau0_s, dop_hz, rate_hz_per_s, period_s, margin_s=0.0):
    """(tau, clipped).  ``tau_s`` stands when it lies within |f/k| + margin of
    ``tau0_s``, the physics bound on range-Doppler coupling (|g| <= 1).
    Otherwise the correction jumped to another path, and tau0 itself comes
    back: a delay an estimator measured, never tau0 +- bound."""
    bound = abs(dop_hz / rate_hz_per_s) + margin_s
    d = ((tau_s - tau0_s + period_s / 2) % period_s) - period_s / 2
    if abs(d) <= bound:
        return tau_s, False
    return tau0_s, True


def _doppler_free_tau(sweeps, wf, fs, tau_raw_s, dop_hz):
    """Delay of one arrival with its Doppler removed, and whether it was clipped.

    The raw peak sits g * (-f/k) from the Doppler-free delay, where g in
    [-1, 1] depends on where the row cut from sample 0 splits the chirp (a
    capture starts at any phase of the sweep; Task 18 review, section 1).  So
    remove the measured Doppler from the interval and run CLEAN again, which
    keeps its joint fit of any neighbours; take the component nearest
    tau_raw within |f/k| plus one sample (none: tau_raw itself), and refine it
    with the other components removed (_refined_tau).  The caller clips the
    result against tau0 (_clip_correction).
    """
    P = wf.period_s
    prof = _derotated_profile(sweeps, wf, fs, dop_hz)
    comps, _ = clean(prof, wf, fs, floor_db=FLOOR_DB, max_comp=MAX_COMPONENTS)
    comps = tag_ladder(comps, wf)

    def dist(a, b):
        return abs(((a - b + P / 2) % P) - P / 2)

    radius = abs(dop_hz / wf.rate_hz_per_s) + 1.0 / fs
    near = [c for c in comps if dist(c.tau_s, tau_raw_s) <= radius]
    best = min(near, key=lambda c: dist(c.tau_s, tau_raw_s), default=None)
    coarse = best.tau_s if best is not None else tau_raw_s
    others = _others(comps, _amplitudes(prof, comps, wf, fs), best) if best is not None else ()
    return _refined_tau(sweeps, wf, fs, coarse, dop_hz, others=others)


@lru_cache(maxsize=8)
def _sidelobes(wf, fs):
    """Power response of one noiseless arrival, relative to its peak, by lag."""
    p = matched_profile(gated(wf, fs, 0.0)[None, :], wf, fs)
    p = p / p.max()
    p.setflags(write=False)
    return p


def _cross_amplitude(c, o, wf, fs, dop_c=None, dop_o=None, m=None) -> float:
    """Amplitude of ``o``'s response at ``c``'s delay, relative to ``c``'s own.

    ``o`` reaches ``c`` through its sidelobes and gating rungs, read from the
    real waveform's response.  The profile CLEAN fits holds a cross term
    2*Re(a*b'), and a coherent one does not average away; its size relative
    to ``c`` is this ratio.  Read the response's largest value within half a
    cell of the lag, both sides.

    Differential Doppler averages the cross term over the interval: the two
    paths' relative phase turns by 2*pi*df*T (T = m * period) and the term
    shrinks as sinc, so the ratio scales by min(1, 1 / (pi * |df| * T)).  Without
    both Doppler estimates, or ``m``, the factor is 1 (coherent: worst case).
    """
    S = _sidelobes(wf, fs)
    n = S.size
    d = int(round(abs(((c.tau_s - o.tau_s + wf.period_s / 2) % wf.period_s) - wf.period_s / 2) * fs))
    w = max(1, int(round(0.5 * fs / wf.bandwidth_hz())))
    ids = np.concatenate([np.arange(d - w, d + w + 1), np.arange(-d - w, -d + w + 1)]) % n
    ratio = 10 ** ((o.level_db - c.level_db) / 20)
    coh = 1.0
    if dop_c is not None and dop_o is not None and m:
        df = abs(dop_c - dop_o)
        if df > 0:
            coh = min(1.0, 1.0 / (np.pi * df * m * wf.period_s))
    return float(np.sqrt(S[ids].max()) * ratio * coh)


def _overlaps(c, comps, wf, fs, sigma_ns, dops, m) -> bool:
    """True when this arrival's delay may carry a bias that delay_sigma_ns omits.

    Resolved overlaps only.  Three tests; any one flags:
    1. another component within OVERLAP_CELLS resolution cells (cell = 1 / B,
       B = wf.bandwidth_hz(); an "m" residue counts only within RESIDUE_DB of
       this arrival);
    2. a ladder rung (m = 1..OVERLAP_MAX_RUNG) of a stronger component within
       that distance;
    3. the predicted bias exceeds this arrival's delay_sigma_ns, from another "A" component (rungs and residues are derived from a stronger
       component, and Doppler distortion makes CLEAN leave some).  Predicted
       bias = BIAS_CELLS / B * _cross_amplitude, which carries the level
       difference, the real waveform's sidelobes and rungs, and the pair's
       Doppler coherence.  The radius so grows with the level difference
       and shrinks with differential Doppler.  BIAS_CELLS comes from
       scripts/calibrate_overlap.py.
    One cell is the physics limit: a neighbour inside one cell cannot be
    resolved, and sub-cell multipath biases the delay unflagged.
    """
    P = wf.period_s
    cell = OVERLAP_CELLS / wf.bandwidth_hz()
    L = (1.0 / (abs(wf.rate_hz_per_s) * wf.gate_period_s)
         if wf.gate_period_s else None)

    def dist(a, b):
        return abs(((a - b + P / 2) % P) - P / 2)

    for o in comps:
        if o is c:
            continue
        real = o.tag != "m" or o.level_db > c.level_db - RESIDUE_DB
        if real and dist(c.tau_s, o.tau_s) < cell:
            return True
        if L is not None and o.level_db > c.level_db:
            for rung in range(1, OVERLAP_MAX_RUNG + 1):
                if min(dist(c.tau_s, o.tau_s + rung * L), dist(c.tau_s, o.tau_s - rung * L)) < cell:
                    return True
        if o.tag == "A":
            bias_ns = BIAS_CELLS / wf.bandwidth_hz() * 1e9 * _cross_amplitude(
                c, o, wf, fs, dops.get(id(c)), dops.get(id(o)), m)
            if bias_ns > sigma_ns:
                return True
    return False


def detect_group(samples: np.ndarray, *, fs: int, channel_centre_hz: float,
                 group: Group) -> GroupResult:
    wf = group.waveform
    n = samples_per_period(wf, fs)
    m = len(samples) // n
    if m == 0:
        return GroupResult(group, 0, float("nan"), float("nan"), None, [], {})
    x = np.asarray(samples)[: m * n].astype(np.complex128)
    shift = wf.centre_hz - channel_centre_hz
    x = x * np.exp(-2j * np.pi * shift * np.arange(m * n) / fs)   # n from the first sample
    sweeps = x.reshape(m, n)

    profile = matched_profile(sweeps, wf, fs)
    gate = None
    if wf.gate_period_s is None:
        gate = fit_gate(profile, wf, fs)
        if gate is not None:
            wf = replace(wf, gate_period_s=gate.period_s)

    nulls = null_levels(sweeps, wf, fs)
    null_max = float(max(nulls))
    null_med = float(np.median(nulls))

    comps, _ = clean(profile, wf, fs, floor_db=FLOOR_DB, max_comp=MAX_COMPONENTS)
    comps = tag_ladder(comps, wf)
    amps = _amplitudes(profile, comps, wf, fs)

    rejected: dict = {}
    arrivals = []
    kept = []
    dops: dict = {}
    B = wf.bandwidth_hz()
    for c in comps:
        if c.tag != "A":
            rejected["rung"] = rejected.get("rung", 0) + 1
            continue
        hb = half_band_db(sweeps, wf, fs, c.tau_s)
        if hb > HALF_BAND_MAX_DB:
            rejected["image"] = rejected.get("image", 0) + 1
            continue
        margin = c.level_db - null_max
        if margin < MARGIN_DB:
            rejected["null"] = rejected.get("null", 0) + 1
            continue
        snr = 10 ** (c.level_db / 10)
        sigma_stat = 1e9 * np.sqrt(3) / (np.pi * B * np.sqrt(2 * snr * m))
        sigma = float(sigma_stat)   # the refine's own bias measures < 4 ns: no term
        v = _slow_time(sweeps, wf, fs, c.tau_s)
        rate, dop_raw, cph, coh, dop_sigma = _doppler(v, wf.period_s)
        low = dop_raw is not None and coh < DOPPLER_MIN_COHERENCE
        dop = None if low else dop_raw
        sc = compute_scintillation(v, sample_rate_hz=1.0 / wf.period_s,
                                   convention=HF_OBLIQUE)
        known = sc.confidence > 0
        tau_raw_ns = int(round(c.tau_s * 1e9))
        dops[id(c)] = dop
        ambiguous = dop is not None and abs(dop) > DOPPLER_AMBIGUOUS / wf.period_s
        clipped = False
        # tau0: CLEAN's arrival, refined with the other components removed and
        # no Doppler taken out.  Every row's delay comes from this estimator
        # or from the same refine with the Doppler removed.
        tau0_s = _refined_tau(sweeps, wf, fs, c.tau_s, 0.0, others=_others(comps, amps, c))
        tau0_ns = int(round(tau0_s * 1e9))
        coupling_ns = abs((dop_raw or 0.0) / wf.rate_hz_per_s) * 1e9
        if dop is not None and not ambiguous:
            # Range-Doppler coupling, corrected (see _doppler_free_tau), and
            # clipped against tau0.
            tau_s, clipped = _clip_correction(
                _doppler_free_tau(sweeps, wf, fs, c.tau_s, dop), tau0_s, dop,
                wf.rate_hz_per_s, wf.period_s, margin_s=_clip_margin_s(sigma_stat))
            tau_ns = int(round(tau_s * 1e9))
            if clipped:
                # tau0 keeps g * f/k of coupling, |g| <= 1
                sigma = float(np.hypot(sigma, coupling_ns))
            elif dop_sigma is not None:
                # the correction moves the delay by up to |f/k|; its error by sigma_f/|k|
                sigma = float(np.hypot(sigma, dop_sigma / abs(wf.rate_hz_per_s) * 1e9))
        else:
            # No usable Doppler: tau0.  Gated off (incoherent), the delay
            # still holds up to |f/k| of coupling, and the incoherent fit
            # sizes f fairly: that bound joins the sigma.  Ambiguous (may
            # alias), the true |f| is at least the alias limit 0.4/P, so the
            # bound never falls below it.
            tau_ns = tau0_ns
            if ambiguous:
                f_bound = max(abs(dop_raw), DOPPLER_AMBIGUOUS / wf.period_s)
                sigma = float(np.hypot(sigma, f_bound / abs(wf.rate_hz_per_s) * 1e9))
            elif low:
                sigma = float(np.hypot(sigma, coupling_ns))
        kept.append(c)
        arrivals.append(Arrival(
            tau_ns=tau_ns, tau_uncorrected_ns=tau_raw_ns,
            doppler_ambiguous=bool(ambiguous), delay_sigma_ns=float(sigma),
            level_db=float(c.level_db), margin_db=float(margin), half_band_db=float(hb),
            clean_tag=c.tag, doppler_hz=dop, phase_rate=None if low else rate,
            carrier_phase_rad=cph,
            s4=float(sc.s4_index) if known else None,
            sigma_phi_rad=float(sc.sigma_phi_rad) if known else None,
            slow=tuple(np.asarray(v).tolist()),
            doppler_coherence=coh, doppler_sigma_hz=dop_sigma,
            doppler_low_coherence=bool(low), doppler_clipped=bool(clipped),
            doppler_raw_hz=dop_raw, tau_zero_doppler_ns=tau0_ns))
    arrivals = [replace(a, overlap_flag=_overlaps(c, comps, wf, fs, a.delay_sigma_ns, dops, m))
                for a, c in zip(arrivals, kept)]
    return GroupResult(group, m, null_max, null_med, gate, arrivals, rejected,
                       sum(a.overlap_flag for a in arrivals))
