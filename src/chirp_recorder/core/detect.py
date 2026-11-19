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
CROSS_MAX = 0.01        # cross-term amplitude, relative to the weaker arrival; errors > 500 ns began at 0.015 (sweep, report)
DOPPLER_AMBIGUOUS = 0.4  # |doppler| over this / period: slow-time sampling may alias; no correction
REFINE_BIAS_NS_64K = 190.0   # sub-sample bias of CLEAN's log-power parabola at 64 kS/s (re-review curve)
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


def _doppler_free_tau(sweeps, wf, fs, tau_raw_s, dop_hz):
    """Delay of one arrival with its Doppler removed.

    Range-Doppler coupling: a Doppler shift f turns k*(t - tau) into
    k*(t - (tau - f/k)), so the matched filter peaks at tau - f/k, k the
    waveform's signed rate (the sign flips with sweep direction).  CLEAN's
    template assumes f = 0; 0.2 Hz over a 1 s sweep distorts the response
    enough that CLEAN, then +f/k, leaves about 1.4 us.  So remove the measured
    Doppler from the interval and run CLEAN again, which keeps its joint fit
    of any neighbours; take the component nearest tau_raw + f/k.  When none
    lies within 3 cells, fall back to tau_raw + f/k.
    """
    expect = (tau_raw_s + dop_hz / wf.rate_hz_per_s) % wf.period_s
    comps, _ = clean(_derotated_profile(sweeps, wf, fs, dop_hz), wf, fs,
                     floor_db=FLOOR_DB, max_comp=MAX_COMPONENTS)
    P = wf.period_s
    best = min(comps, key=lambda c: abs(((c.tau_s - expect + P / 2) % P) - P / 2),
               default=None)
    if best is not None and abs(((best.tau_s - expect + P / 2) % P) - P / 2) < 3 / wf.bandwidth_hz():
        return best.tau_s
    return expect


@lru_cache(maxsize=8)
def _sidelobes(wf, fs):
    """Power response of one noiseless arrival, relative to its peak, by lag."""
    p = matched_profile(gated(wf, fs, 0.0)[None, :], wf, fs)
    p = p / p.max()
    p.setflags(write=False)
    return p


def _cross_amplitude(c, o, wf, fs) -> float:
    """Amplitude of ``o``'s response at ``c``'s delay, relative to ``c``'s own.

    ``o`` reaches ``c`` through its sidelobes and gating rungs, read from the
    real waveform's response.  The profile CLEAN fits holds a cross term
    2*Re(a*b'), and a coherent one does not average away; its size relative
    to ``c`` is this ratio.  Read the response's largest value within half a
    cell of the lag, both sides.
    """
    S = _sidelobes(wf, fs)
    n = S.size
    d = int(round(abs(((c.tau_s - o.tau_s + wf.period_s / 2) % wf.period_s) - wf.period_s / 2) * fs))
    w = max(1, int(round(0.5 * fs / wf.bandwidth_hz())))
    ids = np.concatenate([np.arange(d - w, d + w + 1), np.arange(-d - w, -d + w + 1)]) % n
    ratio = 10 ** ((o.level_db - c.level_db) / 20)
    return float(np.sqrt(S[ids].max()) * ratio)


def _overlaps(c, comps, wf, fs) -> bool:
    """True when this arrival's delay may carry a bias that delay_sigma_ns omits.

    Resolved overlaps only.  Three tests; any one flags:
    1. another component within OVERLAP_CELLS resolution cells (cell = 1 / B,
       B = wf.bandwidth_hz(); an "m" residue counts only within RESIDUE_DB of
       this arrival);
    2. a ladder rung (m = 1..OVERLAP_MAX_RUNG) of a stronger component within
       that distance;
    3. another "A" component (rungs and residues are derived from a stronger
       component, and Doppler distortion makes CLEAN leave some) whose response at this delay, relative to this
       arrival's own (_cross_amplitude: sidelobes and rungs of the real
       waveform, scaled by the level difference), exceeds CROSS_MAX.  The
       radius so grows with the level difference: a neighbour 20 dB stronger
       reaches ~10x further in amplitude.  CROSS_MAX comes from measured
       delay errors of coherent pairs (report, fix round 2).
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
            for m in range(1, OVERLAP_MAX_RUNG + 1):
                if min(dist(c.tau_s, o.tau_s + m * L), dist(c.tau_s, o.tau_s - m * L)) < cell:
                    return True
        if o.tag == "A" and _cross_amplitude(c, o, wf, fs) > CROSS_MAX:
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
        # plus CLEAN's sub-sample bias, in quadrature: REFINE_BIAS_NS_64K at
        # 64 kS/s, scaled by 64000 / fs (the re-review's measured curve)
        sigma = float(np.hypot(sigma_stat, REFINE_BIAS_NS_64K * 64_000 / fs))
        v = _slow_time(sweeps, wf, fs, c.tau_s)
        rate, dop, cph = _doppler(v, wf.period_s)
        sc = compute_scintillation(v, sample_rate_hz=1.0 / wf.period_s,
                                   convention=HF_OBLIQUE)
        known = sc.confidence > 0
        tau_raw_ns = int(round(c.tau_s * 1e9))
        tau_ns = tau_raw_ns
        ambiguous = dop is not None and abs(dop) > DOPPLER_AMBIGUOUS / wf.period_s
        if dop is not None and not ambiguous:
            # Range-Doppler coupling, corrected (see _doppler_free_tau).  No
            # Doppler estimate, or one that may alias (|f| > 0.4 / period):
            # no correction.
            tau_ns = int(round(_doppler_free_tau(sweeps, wf, fs, c.tau_s, dop) * 1e9))
        arrivals.append(Arrival(
            tau_ns=tau_ns, tau_uncorrected_ns=tau_raw_ns,
            overlap_flag=_overlaps(c, comps, wf, fs), doppler_ambiguous=bool(ambiguous), delay_sigma_ns=float(sigma),
            level_db=float(c.level_db), margin_db=float(margin), half_band_db=float(hb),
            clean_tag=c.tag, doppler_hz=dop, phase_rate=rate, carrier_phase_rad=cph,
            s4=float(sc.s4_index) if known else None,
            sigma_phi_rad=float(sc.sigma_phi_rad) if known else None))
    return GroupResult(group, m, null_max, null_med, gate, arrivals, rejected,
                       sum(a.overlap_flag for a in arrivals))
