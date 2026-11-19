"""The codar.spots row builder and sink writer, moved from codar-sounder's
core/daemon.py (`_ch_row_for`, `_make_sink_writer`).

The stopgap owns no sink connection.  `build_spots_row` makes the row; the
channel worker hands it to RowWriter.put(..., spots=...), and RowWriter's
writer thread inserts it through the writer `make_spots_writer` builds
(pre-flight C12).
"""
from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)


def build_spots_row(
    *, timestamp, host_call: str, host_grid: str, radiod_id: str,
    reporter_id: Optional[str], processing_version: str, station_id: str,
    center_freq_hz: int, sweep_rate_hz_per_s: float, coherent_seconds: float,
    detection, fix, scintillation, peak_index: int, peak_count: int,
    dechirp_sweeps_rejected: int = 0,
) -> dict:
    """One row for the codar.spots table.  Numeric fields keep full precision
    (no rounding); the analytics side benefits from the extra digits."""
    return {
        "time":               timestamp,
        "host_call":          host_call,
        "host_grid":          host_grid,
        "radiod_id":          radiod_id,
        "instance":           radiod_id,  # legacy; removed Phase 9
        "reporter_id":        reporter_id or radiod_id,
        "processing_version": processing_version,
        "station_id":         station_id,
        "oblique_freq_hz":    int(center_freq_hz),
        "sweep_rate_hz_per_s": float(sweep_rate_hz_per_s),
        "coherent_seconds":   float(coherent_seconds),
        "peak_index":         int(peak_index),
        "peak_count":         int(peak_count),
        "mode_layer":         fix.mode_layer,
        "snr_db":             float(detection.snr_db),
        "group_range_km":     float(fix.group_range_km),
        "ground_distance_km": float(fix.ground_distance_km),
        "virtual_height_km":  float(fix.virtual_height_km),
        "virtual_height_uncertainty_km":
            float(fix.virtual_height_uncertainty_km),
        "equivalent_vertical_freq_mhz":
            float(fix.equivalent_vertical_freq_mhz),
        "equivalent_vertical_freq_uncertainty_mhz":
            float(fix.equivalent_vertical_freq_uncertainty_mhz),
        "takeoff_zenith_deg": float(fix.takeoff_zenith_deg),
        # Hop count chosen by invert()'s multi-hop selector (v0.7).
        "n_hops":             int(fix.n_hops),
        # ITU-R P.531 scintillation (additive v0.5).
        "s4_index":           float(scintillation.s4_index),
        "s4_severity":        scintillation.s4_severity,
        "sigma_phi_rad":      float(scintillation.sigma_phi_rad),
        "sigma_phi_severity": scintillation.sigma_phi_severity,
        "scintillation_event": bool(scintillation.scintillation_event),
        "scintillation_confidence": float(scintillation.confidence),
        "scintillation_samples": int(scintillation.n_samples),
        "scintillation_outliers_rejected":
            int(scintillation.n_outliers_rejected),
        "mode_doppler_hz":    float(scintillation.mode_doppler_hz),
        # v0.6 diagnostics: linear-detrend sigma_phi + underfit ratio.
        "sigma_phi_linear_rad": float(scintillation.sigma_phi_linear_rad),
        "sigma_phi_quadratic_rad":
            float(scintillation.sigma_phi_quadratic_rad),
        "sigma_phi_underfit_ratio":
            float(scintillation.sigma_phi_underfit_ratio),
        # v0.6.1: per-CPI count of sweeps zeroed by dechirp's pre-filter.
        "dechirp_sweeps_rejected": int(dechirp_sweeps_rejected),
    }


def make_spots_writer() -> Optional[object]:
    """The local sink writer for codar.spots, or None.

    None when neither hs_uploader nor sigmond.hamsci_sink imports, or the
    writer fails to start (JSONL stays unaffected).  Only
    core.writer.make_sink_writers calls it."""
    try:
        try:
            from hs_uploader.sink import Writer  # type: ignore[import-not-found]
        except ImportError:
            from sigmond.hamsci_sink import Writer  # type: ignore[import-not-found]
        return Writer.from_env(table="spots", mode="codar", schema_version=1,
                               batch_rows=200, producer="chirp-recorder")
    except ImportError:
        log.debug("no sink writer importable; codar.spots off")
    except Exception as exc:  # noqa: BLE001
        log.warning("spots writer init failed (%s); JSONL unaffected", exc)
    return None
