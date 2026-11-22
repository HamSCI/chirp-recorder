"""FMCW dechirping engine — Kaeppler 2022 §2.1.

A coastal CODAR transmitter radiates a linear-frequency-modulated
continuous-wave (FMCW) signal: the carrier sweeps linearly across a
narrow band (e.g. 25 kHz at 4.5 MHz) once per sweep-repetition period
(typically 1 s), with a known signed sweep rate ``κ`` (Hz/s).  At a
distant receiver, multiple delayed copies of that chirp arrive — direct
ground-wave plus one or more sky-wave (ionospherically-reflected)
modes.

Dechirping recovers each propagation mode's group-range delay by
mixing the received signal with a phase-coherent replica of the
transmitted chirp.  The mixer output is a "beat" frequency
proportional to the path delay::

       f_beat = κ · τ           ⇒    τ = f_beat / κ
       group_range = c · τ

So a 100 Hz beat tone at κ = -25.7 kHz/s indicates a target at
group-range ≈ c · (100 / 25700) s ≈ 1167 km.

This module implements Kaeppler's frequency-domain formulation:
  1. Build a windowed replica of one sweep.
  2. Reshape received IQ into an M × N matrix (M sweeps, N samples each).
  3. Per-sweep dechirp by complex-multiplying the conjugate of the
     replica into each row.
  4. FFT in fast-time → range profile per sweep.
  5. FFT in slow-time → range-Doppler matrix.

The implementation is deliberately stdlib-numpy (no scipy) so the
client stays inside sigmond's "stdlib + ka9q-python + numpy" envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# Speed of light, km/s.  Matches core/invert.py — keep them in sync.
C_KM_PER_S = 299_792.458

# Per-sweep MAD outlier rejection threshold (v0.6.1).  After
# computing the range_spectrum, sweeps whose total power deviates
# more than this many MADs from the median are zeroed out before the
# slow-time FFT.  Catches sferic-like impulses, discrete-tone RFI
# bursts, and longer-duration disturbances at the row level — broader
# coverage than the v0.5.1 per-peak MAD filter applied downstream in
# scintillation.  K=4 mirrors the scintillation constant.
SWEEP_MAD_REJECTION_K = 4.0


@dataclass(frozen=True)
class DechirpResult:
    """One coherent processing interval's dechirped range-Doppler output.

    Fields:
        range_doppler: complex M×N array — Doppler bins × range bins.
        range_spectrum: complex M×N array — *pre-slow-time-FFT* range
            spectrum (matched-filter output per sweep, per range bin).
            One column ``range_spectrum[:, raw_bin]`` is the slow-time
            complex amplitude time series at that range bin's
            propagation mode — the natural input for scintillation
            indices (see :mod:`hamsci_dsp.propagation.scintillation`).
            Stored as complex64 to halve the per-CPI memory cost vs.
            numpy's default complex128 FFT output; precision is still
            ~24 bits, far above what amplitude/phase statistics need.
        range_axis_km: real N-vector — group range (km) per FFT bin.
        doppler_axis_hz: real M-vector — Doppler frequency (Hz) per slow-time bin.
        sweep_rate_hz_per_s: the sweep rate used (echoed for downstream consumers).
        sample_rate_hz: the sample rate used (likewise).
        n_sweeps_rejected: count of sweeps zeroed out by the per-sweep
            MAD pre-filter (v0.6.1).  A high count flags a CPI with
            heavy RFI / sferic contamination.
        bad_sweep_mask: 1D bool array of length M.  ``True`` at sweep
            positions that the pre-filter zeroed out.  Propagated to
            scintillation as ``pre_rejected_mask`` so the downstream
            per-peak MAD operates only on the clean samples — avoids
            the case where embedded zeros pollute the MAD scale and
            mask their own outlier status.
    """
    range_doppler: np.ndarray
    range_spectrum: np.ndarray
    range_axis_km: np.ndarray
    doppler_axis_hz: np.ndarray
    sweep_rate_hz_per_s: float
    sample_rate_hz: float
    n_sweeps_rejected: int = 0
    bad_sweep_mask: Optional[np.ndarray] = None


def make_replica(
    n_samples: int,
    sample_rate_hz: float,
    sweep_rate_hz_per_s: float,
    *,
    window: bool = True,
    phase_offset_samples: float = 0,
) -> np.ndarray:
    """Generate one FMCW chirp replica at complex baseband.

    A CODAR sweep starts at one band edge, crosses the channel centre half a
    period P later and ends at the other edge, with linearly-varying
    instantaneous frequency ``f(t) = κ·(t - P/2)``, t counted from the sweep
    start.  Phase = ∫f dt, so the complex baseband signal is::

        s(t) = exp(j · 2π · ½ · κ · (t - P/2)²)

    the model hamsci_dsp's chirp (and so the sigmondb rows) uses.  The
    replica sweeps the transmitter's whole band, +B/2 to -B/2 for a
    down-chirp.  (codar-sounder's replica swept 0 to -B, so only the half
    after the centre crossing matched.)

    A Hann window is applied by default to suppress sidelobe artifacts
    when Fourier-transforming a finite-support function (Kaeppler Eq. 6).

    Args:
        n_samples: samples per sweep (= sample_rate_hz / sweep_repetition_hz).
        sample_rate_hz: IQ sample rate (Hz).
        sweep_rate_hz_per_s: signed sweep rate (Hz/s).  Sign matters: a
            CODAR down-chirp has ``κ < 0``; up-chirps have ``κ > 0``.
        window: apply a Hann window to suppress FFT sidelobes.
        phase_offset_samples: the buffer sample at the replica's sweep
            onset (its start frequency, one band edge), modulo ``n_samples``.  A transmitter whose sweep
            starts at that sample dechirps to its path delay.  It may be
            fractional.  Default 0.

    The offset convention, one for the whole physics path: a configured
    ``tdma_offset_samples`` is the phase of the transmitter's sweep onset
    after the UTC second, in samples at the channel rate, and the replica
    starts there.  Every sweep time counts from the onset
    (docs/SWEEP-TIMING.md).  ``TransmitterPipeline`` passes this function the configured
    phase less the CPI's own phase after the second
    (:func:`utc_phase_samples`).  The sigmondb rows count from the same
    onset.  codar-sounder's ``make_replica`` put its zero at
    ``-offset`` instead; the two agree only at 0 and ``n_samples/2``.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0; got {n_samples}")
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be > 0; got {sample_rate_hz}")

    # Time into the sweep, modulo one period so the replica wraps
    # continuously: the sweep starts at sample +phase_offset_samples, so
    # sample n is (n - offset) mod N samples into it.
    offset = float(phase_offset_samples) % n_samples
    t_into_sweep = ((np.arange(n_samples) - offset) % n_samples) / sample_rate_hz
    half = n_samples / sample_rate_hz / 2
    phase = np.pi * sweep_rate_hz_per_s * (t_into_sweep - half) ** 2
    replica = np.exp(1j * phase)
    if window:
        replica = replica * np.hanning(n_samples)
    return replica.astype(np.complex64)


def utc_phase_samples(utc_ns: int, sample_rate_hz: float, n_per_sweep: int) -> float:
    """Where ``utc_ns`` falls within the sweep period, in samples.

    The period is counted from the UTC epoch, so for a 1 s period this is
    the phase after the UTC second (the reference sigmondb rows use too).
    Integer nanoseconds keep the large epoch term exact."""
    period_ns = n_per_sweep * 10**9 / sample_rate_hz
    if period_ns.is_integer():
        return (int(utc_ns) % int(period_ns)) * sample_rate_hz / 1e9
    return (int(utc_ns) / 1e9 * sample_rate_hz) % n_per_sweep


def dechirp(
    rx_samples: np.ndarray,
    *,
    sample_rate_hz: float,
    sweep_rate_hz_per_s: float,
    sweep_repetition_hz: float,
    apply_window: bool = True,
    phase_offset_samples: float = 0,
) -> DechirpResult:
    """Dechirp a CPI of received IQ samples and return the range-Doppler matrix.

    Args:
        rx_samples: 1-D complex IQ from the radiod channel.  Must be at
            least M·N samples long (M = SRF·CPI seconds, N =
            sample_rate / SRF); excess samples at the tail are
            discarded.
        sample_rate_hz: IQ sample rate.  Must equal the rate of the
            radiod channel that produced ``rx_samples``.
        sweep_rate_hz_per_s: signed CODAR sweep rate.  This selects
            *which* CODAR transmitter is being dechirped — multiple
            transmitters in the same band (TDMA-offset) are
            distinguished by their sweep timing/rate.
        sweep_repetition_hz: number of sweeps per second.  Determines
            both the sweep period (and thus N) and the Doppler
            unambiguous bandwidth.
        apply_window: pass a Hann window through to the replica
            generator.
        phase_offset_samples: the buffer sample at this TX's sweep onset
            (see ``make_replica``).  Selects which TX in a shared
            band to dechirp.  The rows start at the first whole sample at
            or after it, so a nonzero offset costs up to one sweep at the
            head of the CPI.  Default 0.
    """
    if rx_samples.dtype.kind != "c":
        raise ValueError(
            f"rx_samples must be complex; got dtype={rx_samples.dtype}"
        )
    if sweep_repetition_hz <= 0:
        raise ValueError(
            f"sweep_repetition_hz must be > 0; got {sweep_repetition_hz}"
        )

    n_samples = int(round(sample_rate_hz / sweep_repetition_hz))
    if n_samples <= 1:
        raise ValueError(
            f"sample_rate_hz / sweep_repetition_hz = {n_samples} must be > 1"
        )
    # Cut the rows at the transmitter's sweep start: skip to the first whole
    # sample at or after it, and let the replica carry the fraction left.
    # A row cut mid-sweep would put the sweep's wrap (and a phase step in
    # the beat tone) under the window's middle, and the path's gain would
    # then swing with the run's CPI phase.
    offset = float(phase_offset_samples) % n_samples
    skip = int(np.ceil(offset))
    if skip >= n_samples:
        skip -= n_samples
    n_sweeps = (len(rx_samples) - skip) // n_samples
    if n_sweeps < 1:
        raise ValueError(
            f"need at least one full sweep ({n_samples} samples) after the "
            f"sweep start at sample {skip}; got {len(rx_samples)}"
        )

    # Reshape into M × N: rows are sweeps (slow time), columns are samples
    # within one sweep (fast time).
    rx_matrix = rx_samples[skip: skip + n_sweeps * n_samples].reshape(n_sweeps, n_samples)

    # Build the replica and dechirp each sweep by multiplying with the
    # complex-conjugate.  The result has a beat tone for each propagation
    # mode at frequency = sweep_rate · path_delay.
    replica = make_replica(
        n_samples, sample_rate_hz, sweep_rate_hz_per_s,
        window=apply_window,
        phase_offset_samples=offset - skip,
    )
    dechirped = rx_matrix * np.conj(replica)

    # FFT along the fast-time axis → range spectrum per sweep.
    # Downcast to complex64 immediately: numpy's FFT returns complex128
    # even for complex64 input, and we don't need the extra precision
    # downstream (amplitude/phase statistics are well within complex64
    # range).  Halves the per-CPI memory footprint of `range_spectrum`.
    range_spectrum = np.fft.fft(dechirped, axis=1).astype(np.complex64)

    # ─── Per-sweep MAD rejection (v0.6.1) ────────────────────────────
    # Live verification on bee1-rx888 found that ~10-80% of CPIs have
    # 1-2 sweeps contaminated by sferic-like impulses, discrete-tone
    # RFI bursts (e.g., shortwave broadcast carriers at ±20 kHz from
    # the CODAR center), or longer-duration disturbances.  Each
    # bad sweep contributes elevated power across many range bins,
    # corrupting both range_profile (peak detection) and per-peak
    # slow-time vectors (scintillation).  Zeroing the bad sweeps
    # here removes their contribution from both downstream products.
    # The MeanAD fallback handles MAD=0 degeneracy (uniform input,
    # rare in real signal but common in synthetic test fixtures).
    sweep_total_power = (np.abs(range_spectrum) ** 2).sum(axis=1)
    sweep_deviations = np.abs(sweep_total_power - np.median(sweep_total_power))
    mad = float(np.median(sweep_deviations))
    if mad > 0.0:
        scale = mad
    else:
        scale = 1.2533 * float(np.mean(sweep_deviations))
    if scale > 0.0:
        bad_sweep_mask = sweep_deviations > SWEEP_MAD_REJECTION_K * scale
    else:
        bad_sweep_mask = np.zeros(n_sweeps, dtype=bool)
    n_sweeps_rejected = int(bad_sweep_mask.sum())
    if n_sweeps_rejected > 0:
        # In-place zero: range_spectrum is a fresh array we own.
        # Zeroing keeps the range_doppler / range_profile clean
        # (sferic-contaminated sweeps don't contribute to the
        # incoherent sum); the bad_sweep_mask propagates the same
        # information to scintillation so the per-peak MAD treats
        # those positions as already-rejected instead of trying to
        # detect them anew via MAD-on-intensity.
        range_spectrum[bad_sweep_mask, :] = 0

    # FFT along the slow-time axis → Doppler resolution per range bin.
    # Computed AFTER the sweep rejection so dropped sweeps don't
    # contribute to range_doppler (which range_profile sums across).
    range_doppler = np.fft.fftshift(np.fft.fft(range_spectrum, axis=0), axes=0)

    # Convert FFT bin → group range.  rx · conj(replica) leaves a tone at
    # beat_freq = -κ · τ, so τ = -beat_freq / κ.  A down-chirp (κ < 0) puts
    # positive delays at positive beat frequencies, an up-chirp (κ > 0) at
    # negative ones; the signed κ lands both on positive group range.
    bin_freqs_hz = np.fft.fftfreq(n_samples, d=1.0 / sample_rate_hz)
    delays_s = -bin_freqs_hz / sweep_rate_hz_per_s
    range_axis_km = C_KM_PER_S * delays_s

    doppler_axis_hz = np.fft.fftshift(
        np.fft.fftfreq(n_sweeps, d=1.0 / sweep_repetition_hz)
    )

    return DechirpResult(
        range_doppler=range_doppler,
        range_spectrum=range_spectrum,
        range_axis_km=range_axis_km,
        doppler_axis_hz=doppler_axis_hz,
        sweep_rate_hz_per_s=sweep_rate_hz_per_s,
        sample_rate_hz=sample_rate_hz,
        n_sweeps_rejected=n_sweeps_rejected,
        bad_sweep_mask=bad_sweep_mask,
    )


def range_profile(result: DechirpResult) -> np.ndarray:
    """Sum |range_doppler|² across Doppler bins → 1-D power-vs-range vector.

    For F-region detection we don't need Doppler resolution — the dominant
    energy at each range integrates across all Doppler bins.  Returns
    real-valued power in arbitrary units (consumer normalises if needed).
    """
    return np.abs(result.range_doppler).sum(axis=0).astype(np.float32)


def positive_range_window(
    result: DechirpResult, profile: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return only the positive-range half of the range axis + profile.

    The FFT output covers ±half the unambiguous range; for sky-wave
    propagation we care only about positive group ranges (the path
    delay is always positive).  This crops the negative half cleanly.
    """
    pos_mask = result.range_axis_km >= 0
    # Sort by ascending range so plotting/peak-finding is well-behaved.
    order = np.argsort(result.range_axis_km[pos_mask])
    return (
        result.range_axis_km[pos_mask][order],
        profile[pos_mask][order],
    )


def positive_to_raw_index_map(result: DechirpResult) -> np.ndarray:
    """Return the index map from positive-sorted bin → raw FFT bin.

    ``find_f_region_peaks`` returns each peak's ``bin_index`` as an
    index into the positive-sorted profile that came out of
    :func:`positive_range_window`.  Slicing into
    ``result.range_spectrum`` or ``result.range_doppler`` requires the
    *raw* FFT-bin index (the original axis of those tensors).

    This helper builds the lookup once per CPI::

        raw_indices = positive_to_raw_index_map(result)
        raw_bin = raw_indices[detection.bin_index]
        slow_time = result.range_spectrum[:, raw_bin]

    Returns a 1-D ``int64`` array whose length equals the number of
    positive range bins (typically half of N).
    """
    pos_mask = result.range_axis_km >= 0
    order = np.argsort(result.range_axis_km[pos_mask])
    return np.where(pos_mask)[0][order]


def raw_bin_from_positive(
    result: DechirpResult, positive_bin_index: int,
) -> int:
    """Convert a positive-sorted bin index to its raw FFT-bin index.

    Convenience scalar wrapper around :func:`positive_to_raw_index_map`
    for one-off lookups.  When converting many indices for the same
    ``DechirpResult`` (e.g. one per detected peak within a CPI), call
    ``positive_to_raw_index_map(result)`` once and index into the
    returned array — that avoids recomputing the mapping per peak.

    Precondition: ``0 <= positive_bin_index < (number of positive
    range bins)``.  Raises ``IndexError`` otherwise.
    """
    raw_indices = positive_to_raw_index_map(result)
    return int(raw_indices[positive_bin_index])
