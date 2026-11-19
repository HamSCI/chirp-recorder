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
from typing import Optional

import numpy as np
from hamsci_dsp.chirp.clean import clean, matched_profile
from hamsci_dsp.chirp.screen import half_band_db, null_levels
from hamsci_dsp.chirp.clean import tag_ladder
from hamsci_dsp.chirp.waveform import GateFit, fit_gate, replica, samples_per_period
from hamsci_dsp.propagation.scintillation import HF_OBLIQUE, compute_scintillation

from chirp_recorder.core.channels import Group

MARGIN_DB = 3.0
HALF_BAND_MAX_DB = 6.0
FLOOR_DB = 10.0
MAX_COMPONENTS = 12
OVERLAP_CELLS = 2.0     # an overlap lies within this many resolution cells ...
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
    overlap_flag: bool = False
    tau_uncorrected_ns: Optional[int] = None    # None reads as "equal to tau_ns"

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


def _doppler_free_tau(sweeps, wf, fs, tau_raw_s, dop_hz):
    """Delay of one arrival with its Doppler removed.

    Range-Doppler coupling: a Doppler shift f turns k*(t - tau) into
    k*(t - (tau - f/k)), so the matched filter peaks at tau - f/k, k the
    waveform's signed rate (the sign flips with sweep direction).  CLEAN's
    template assumes f = 0, and 0.2 Hz over a 1 s sweep distorts the response
    enough that CLEAN, then +f/k, leaves about 1.4 us.  So derotate the
    interval by the measured Doppler, compress again, and read the peak
    near tau_raw + f/k, refined by a parabola in log power as CLEAN does.
    """
    m, n = sweeps.shape
    rot = np.exp(-2j * np.pi * dop_hz * np.arange(m * n) / fs).reshape(m, n)
    p = matched_profile(sweeps * rot, wf, fs)
    i0 = int(round((tau_raw_s + dop_hz / wf.rate_hz_per_s) * fs))
    idx = np.arange(i0 - 2, i0 + 3) % n
    i = int(idx[int(np.argmax(p[idx]))])
    a, b, c = np.log(p[(i - 1) % n]), np.log(p[i]), np.log(p[(i + 1) % n])
    den = a - 2 * b + c
    frac = i + (0.5 * (a - c) / den if den < 0 else 0.0)
    return (frac / fs) % wf.period_s


def _overlaps(c, comps, wf) -> bool:
    """True when another component lies within OVERLAP_CELLS resolution cells.

    The cell is 1 / B, B = wf.bandwidth_hz().  Also true when a ladder rung
    (m = 1..OVERLAP_MAX_RUNG, either side) of a stronger component lies
    within that distance: a coherent neighbour pulls the CLEAN peak, and
    delay_sigma_ns (one isolated arrival) does not cover that.
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
        if dist(c.tau_s, o.tau_s) < cell and (o.tag != "m" or o.level_db > c.level_db - RESIDUE_DB):
            return True
        if L is not None and o.level_db > c.level_db:
            for m in range(1, OVERLAP_MAX_RUNG + 1):
                if min(dist(c.tau_s, o.tau_s + m * L), dist(c.tau_s, o.tau_s - m * L)) < cell:
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
        sigma = 1e9 * np.sqrt(3) / (np.pi * B * np.sqrt(2 * snr * m))
        v = _slow_time(sweeps, wf, fs, c.tau_s)
        rate, dop, cph = _doppler(v, wf.period_s)
        sc = compute_scintillation(v, sample_rate_hz=1.0 / wf.period_s,
                                   convention=HF_OBLIQUE)
        known = sc.confidence > 0
        tau_raw_ns = int(round(c.tau_s * 1e9))
        tau_ns = tau_raw_ns
        if dop is not None:
            # Range-Doppler coupling, corrected (see _doppler_free_tau).  With
            # no Doppler estimate, no correction.
            tau_ns = int(round(_doppler_free_tau(sweeps, wf, fs, c.tau_s, dop) * 1e9))
        arrivals.append(Arrival(
            tau_ns=tau_ns, tau_uncorrected_ns=tau_raw_ns,
            overlap_flag=_overlaps(c, comps, wf), delay_sigma_ns=float(sigma),
            level_db=float(c.level_db), margin_db=float(margin), half_band_db=float(hb),
            clean_tag=c.tag, doppler_hz=dop, phase_rate=rate, carrier_phase_rad=cph,
            s4=float(sc.s4_index) if known else None,
            sigma_phi_rad=float(sc.sigma_phi_rad) if known else None))
    return GroupResult(group, m, null_max, null_med, gate, arrivals, rejected,
                       sum(a.overlap_flag for a in arrivals))
