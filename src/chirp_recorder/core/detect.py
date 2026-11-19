"""Detection on one interval for one transmitter group (the raw path).

Six steps: mix the group's centre to zero; cut whole sweep periods from
sample 0; compress with the plain-replica matched filter; CLEAN and tag the
ladder; screen with the half-band test and the null margin; measure delay,
delay sigma, Doppler, carrier phase and scintillation per kept arrival.

The shared pieces (matched filter, CLEAN, gate fit, screens, scintillation)
come from hamsci-dsp.  Pure function: no clock reads, no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
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
    """phase_rate (rad/s), doppler_hz and carrier phase at mid-interval."""
    m = v.size
    if m < 2:
        return None, None, None
    ph = np.unwrap(np.angle(v))
    t = np.arange(m) * period_s
    # Least-squares slope over every sweep, not first minus last.  Slow-time
    # sampling aliases |doppler| above 1 / (2 * period).
    rate = float(np.polyfit(t, ph, 1)[0])
    t_mid = m * period_s / 2
    z = np.sum(v * np.exp(-1j * rate * (t - t_mid)))
    return rate, rate / (2 * np.pi), float(np.angle(z))


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


def _refined_tau(sweeps, wf, fs, tau_s, dop_hz):
    """Delay at the peak near ``tau_s`` of the complex matched-filter output.

    CLEAN refines its peak with a 3-point parabola in log power, which biases
    a delay by up to ~190 ns at 64 kS/s (and 1.8 us at fs/B = 1.3), depending
    on where it falls between samples.  Here the complex output of each sweep,
    with ``dop_hz`` removed, is cut to REFINE_WINDOW samples around the peak,
    zero-padded REFINE_UPSAMPLE times by FFT (the output is band-limited, so
    this interpolates it), averaged in power over the sweeps, and the same
    log-parabola is applied on the fine grid.  Measured bias: under 4 ns
    across fs/B = 1.29 .. 7.46 (tests).  Blocks keep the temporaries small.
    """
    m, n = sweeps.shape
    W, up = REFINE_WINDOW, REFINE_UPSAMPLE
    i0 = int(round(tau_s * fs))
    cols = (np.arange(i0 - W // 2, i0 + W // 2)) % n
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
    p = np.mean(np.abs(np.fft.ifft(Fp, axis=1)) ** 2, axis=0)
    c0 = (W // 2) * up                 # fine index of the sample at tau_s
    lo = c0 - 3 * up                   # look within 3 samples only: a neighbour's
    j = lo + int(np.argmax(p[lo:c0 + 3 * up + 1]))   # peak in the window is not ours
    la, lb, lc = np.log(p[j - 1]), np.log(p[j]), np.log(p[j + 1])
    den = la - 2 * lb + lc
    jj = j + (0.5 * (la - lc) / den if den < 0 else 0.0)
    return (((i0 - W // 2) + jj / up) / fs) % wf.period_s


def _doppler_free_tau(sweeps, wf, fs, tau_raw_s, dop_hz):
    """Delay of one arrival with its Doppler removed.

    Range-Doppler coupling: a Doppler shift f turns k*(t - tau) into
    k*(t - (tau - f/k)), so the matched filter peaks at tau - f/k, k the
    waveform's signed rate (the sign flips with sweep direction).  CLEAN's
    template assumes f = 0; 0.2 Hz over a 1 s sweep distorts the response
    enough that CLEAN, then +f/k, leaves about 1.4 us.  So remove the measured
    Doppler from the interval and run CLEAN again, which keeps its joint fit
    of any neighbours; take the component nearest tau_raw + f/k (none within
    3 cells: tau_raw + f/k itself), then refine it (_refined_tau).
    """
    expect = (tau_raw_s + dop_hz / wf.rate_hz_per_s) % wf.period_s
    comps, _ = clean(_derotated_profile(sweeps, wf, fs, dop_hz), wf, fs,
                     floor_db=FLOOR_DB, max_comp=MAX_COMPONENTS)
    P = wf.period_s
    best = min(comps, key=lambda c: abs(((c.tau_s - expect + P / 2) % P) - P / 2),
               default=None)
    coarse = expect
    if best is not None and abs(((best.tau_s - expect + P / 2) % P) - P / 2) < 3 / wf.bandwidth_hz():
        coarse = best.tau_s
    return _refined_tau(sweeps, wf, fs, coarse, dop_hz)


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
        rate, dop, cph = _doppler(v, wf.period_s)
        sc = compute_scintillation(v, sample_rate_hz=1.0 / wf.period_s,
                                   convention=HF_OBLIQUE)
        known = sc.confidence > 0
        tau_raw_ns = int(round(c.tau_s * 1e9))
        tau_ns = tau_raw_ns
        dops[id(c)] = dop
        ambiguous = dop is not None and abs(dop) > DOPPLER_AMBIGUOUS / wf.period_s
        if dop is not None and not ambiguous:
            # Range-Doppler coupling, corrected (see _doppler_free_tau).  No
            # Doppler estimate, or one that may alias (|f| > 0.4 / period):
            # no correction.
            tau_ns = int(round(_doppler_free_tau(sweeps, wf, fs, c.tau_s, dop) * 1e9))
        else:
            tau_ns = int(round(_refined_tau(sweeps, wf, fs, c.tau_s, 0.0) * 1e9))
        kept.append(c)
        arrivals.append(Arrival(
            tau_ns=tau_ns, tau_uncorrected_ns=tau_raw_ns,
            doppler_ambiguous=bool(ambiguous), delay_sigma_ns=float(sigma),
            level_db=float(c.level_db), margin_db=float(margin), half_band_db=float(hb),
            clean_tag=c.tag, doppler_hz=dop, phase_rate=rate, carrier_phase_rad=cph,
            s4=float(sc.s4_index) if known else None,
            sigma_phi_rad=float(sc.sigma_phi_rad) if known else None))
    arrivals = [replace(a, overlap_flag=_overlaps(c, comps, wf, fs, a.delay_sigma_ns, dops, m))
                for a, c in zip(arrivals, kept)]
    return GroupResult(group, m, null_max, null_med, gate, arrivals, rejected,
                       sum(a.overlap_flag for a in arrivals))
