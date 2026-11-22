"""One transmitter's dechirp -> trace -> invert -> write pipeline.

Moved from codar-sounder's core/daemon.py (`_TransmitterPipeline`), otherwise unchanged
in behaviour.  Three differences, all about plumbing:
  - no sink writer and no sigmondb hook (the stopgap owns no sink; the rows
    come back to the caller, pre-flight C12);
  - dechirp and trace failures raise instead of logging, so the stopgap can
    log one warning per CPI;
  - JSONL lands under <output_dir>/physics/<tx>/YYYY/MM/DD.jsonl, and the
    record's "client" names chirp-recorder.
Two differences in behaviour (2026-11-22): tdma_offset_samples names the
transmitter's sweep start after the UTC second, and process_cpi places the
replica from the CPI's own UTC label.  codar-sounder aimed the replica at
-offset from an unreferenced buffer start (see dechirp.make_replica).  And
the range axis takes the sweep rate's sign, so an up-chirp reads positive range.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from hamsci_dsp.geometry import great_circle_km
from hamsci_dsp.propagation.scintillation import compute_scintillation

from chirp_recorder.physics.dechirp import (
    dechirp,
    positive_range_window,
    positive_to_raw_index_map,
    range_profile,
    utc_phase_samples,
)
from chirp_recorder.physics.invert import group_range_resolution_km, invert
from chirp_recorder.physics.output import JsonlWriter
from chirp_recorder.physics.spots import build_spots_row
from chirp_recorder.physics.trace import (
    DEFAULT_MAX_PEAKS,
    GroundClutterMask,
    find_f_region_peaks,
)

log = logging.getLogger(__name__)

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class PhysicsJsonlWriter(JsonlWriter):
    """codar-sounder's record schema, laid out <root>/<tx>/YYYY/MM/DD.jsonl."""

    CLIENT = "chirp-recorder"

    def _path_for(self, ts: datetime) -> Path:
        d = ts.astimezone(timezone.utc)
        return (self.output_root / self.station_id
                / f"{d.year:04d}" / f"{d.month:02d}" / f"{d.day:02d}.jsonl")


class TransmitterPipeline:
    """One CODAR transmitter's end-to-end processing pipeline."""

    def __init__(
        self,
        *,
        receiver_lat: float,
        receiver_lon: float,
        radiod_id: str,
        radiod_status_dns: str,
        tx_config: dict,
        output_dir: Path,
        sample_rate_hz: float,
        coherent_seconds: float,
        range_min_km: float,
        range_max_km: float,
        snr_threshold_db: float,
        clutter_window: int = 20,
        max_peaks_per_cpi: int = DEFAULT_MAX_PEAKS,
        host_call: str = "",
        host_grid: str = "",
        processing_version: str = "",
        reporter_id: Optional[str] = None,
    ):
        self.station_id = tx_config["id"]
        self.tx_config = tx_config
        self.center_freq_hz = int(tx_config["center_freq_hz"])
        self.sweep_rate_hz_per_s = float(tx_config["sweep_rate_hz_per_s"])
        self.sweep_bw_hz = float(tx_config["sweep_bw_hz"])
        self.sweep_repetition_hz = float(tx_config["sweep_repetition_hz"])
        # The TX's sweep-start phase after the UTC second, in samples at the
        # channel rate (the convention in dechirp.make_replica); process_cpi
        # turns it into this CPI's buffer offset.
        self.tdma_offset_samples = int(tx_config.get("tdma_offset_samples", 0))
        # Sweep length (s); tx_config's sweep_s is None for a full-period
        # sweep.  The sweep crosses the channel centre (the replica's zero)
        # half a sweep after it starts.
        self.sweep_s = float(tx_config.get("sweep_s") or 1.0 / self.sweep_repetition_hz)
        self.ground_distance_km = great_circle_km(
            receiver_lat, receiver_lon,
            float(tx_config["tx_lat_deg"]), float(tx_config["tx_lon_deg"]),
        )
        self.group_range_uncertainty_km = group_range_resolution_km(self.sweep_bw_hz)
        self.radiod_id = radiod_id
        self.radiod_status_dns = radiod_status_dns
        self.sample_rate_hz = sample_rate_hz
        self.coherent_seconds = coherent_seconds
        self.range_min_km = range_min_km
        self.range_max_km = range_max_km
        self.snr_threshold_db = snr_threshold_db
        self.max_peaks_per_cpi = max_peaks_per_cpi
        self.host_call = host_call
        self.host_grid = host_grid
        self.processing_version = processing_version
        self.reporter_id = reporter_id
        self.clutter_mask = GroundClutterMask(window=clutter_window)
        self.writer = PhysicsJsonlWriter(
            output_root=output_dir, radiod_id=radiod_id, station_id=self.station_id)
        log.info(
            "physics pipeline: tx=%s freq=%d Hz kappa=%.1f Hz/s BW=%.0f Hz "
            "TX-RX dist=%.1f km tdma_offset=%d samples",
            self.station_id, self.center_freq_hz, self.sweep_rate_hz_per_s,
            self.sweep_bw_hz, self.ground_distance_km, self.tdma_offset_samples)

    def replica_start_samples(self, cpi_start_ns: int) -> float:
        """The CPI sample where this TX's sweep crosses the channel centre
        (make_replica's phase_offset_samples): its configured sweep-start
        phase after the UTC second, plus half a sweep, less the CPI's own
        phase after the second."""
        n = int(round(self.sample_rate_hz / self.sweep_repetition_hz))
        return (self.tdma_offset_samples + self.sweep_s * self.sample_rate_hz / 2
                - utc_phase_samples(cpi_start_ns, self.sample_rate_hz, n)) % n

    def process_cpi(self, rx_samples, cpi_start_utc: datetime, *,
                    cpi_start_ns: Optional[int] = None):
        """Run one CPI through dechirp -> trace -> invert -> write.

        ``cpi_start_ns`` is the UTC label of the first sample in integer
        nanoseconds; without it the datetime (microseconds) stands in.

        Returns (path of the last JSONL write or None, codar.spots rows, one
        per recorded peak).  A dechirp or trace failure raises."""
        if cpi_start_ns is None:
            cpi_start_ns = (cpi_start_utc - _EPOCH) // timedelta(microseconds=1) * 1000
        result = dechirp(
            rx_samples,
            sample_rate_hz=self.sample_rate_hz,
            sweep_rate_hz_per_s=self.sweep_rate_hz_per_s,
            sweep_repetition_hz=self.sweep_repetition_hz,
            phase_offset_samples=self.replica_start_samples(cpi_start_ns),
        )
        ranges_km, profile = positive_range_window(result, range_profile(result))
        # The ground-clutter mask runs here and nowhere else: no arrival row
        # passes through it (pre-flight S7).
        detections = find_f_region_peaks(
            profile, ranges_km,
            range_min_km=self.range_min_km,
            range_max_km=self.range_max_km,
            snr_threshold_db=self.snr_threshold_db,
            clutter_mask=self.clutter_mask,
            max_peaks=self.max_peaks_per_cpi,
        )
        rows: list[dict] = []
        if not detections:
            log.debug("[%s] no F-region peak above %.1f dB this CPI",
                      self.station_id, self.snr_threshold_db)
            return None, rows

        raw_indices = positive_to_raw_index_map(result)
        if result.n_sweeps_rejected > 0:
            log.info("[%s] dechirp rejected %d/%d sweep(s) (RFI/sferic pre-filter)",
                     self.station_id, result.n_sweeps_rejected,
                     result.range_spectrum.shape[0])

        ts = cpi_start_utc
        last_path: Optional[Path] = None
        peak_count = len(detections)
        for peak_index, detection in enumerate(detections):
            try:
                fix = invert(
                    group_range_km=detection.group_range_km,
                    ground_distance_km=self.ground_distance_km,
                    oblique_freq_mhz=self.center_freq_hz / 1e6,
                    group_range_uncertainty_km=self.group_range_uncertainty_km,
                )
            except ValueError as exc:
                # group_range < ground_distance: short-range clutter the mask
                # missed.  Skip this peak, keep the rest.
                log.warning(
                    "[%s] inversion rejected peak %d/%d at %.0f km "
                    "(TX-RX %.0f km): %s",
                    self.station_id, peak_index, peak_count,
                    detection.group_range_km, self.ground_distance_km, exc)
                continue

            raw_bin = int(raw_indices[detection.bin_index])
            slow_time = result.range_spectrum[:, raw_bin]
            scint = compute_scintillation(
                slow_time, sample_rate_hz=self.sweep_repetition_hz,
                pre_rejected_mask=result.bad_sweep_mask,
            )
            last_path = self.writer.write(
                timestamp=ts, fix=fix, detection=detection, scintillation=scint,
                radiod_status_dns=self.radiod_status_dns,
                oblique_freq_hz=self.center_freq_hz,
                coherent_seconds=self.coherent_seconds,
                sweep_rate_hz_per_s=self.sweep_rate_hz_per_s,
                peak_index=peak_index, peak_count=peak_count,
                dechirp_sweeps_rejected=result.n_sweeps_rejected,
            )
            rows.append(self._ch_row_for(
                timestamp=ts, detection=detection, fix=fix, scintillation=scint,
                peak_index=peak_index, peak_count=peak_count,
                dechirp_sweeps_rejected=result.n_sweeps_rejected))
            log.info(
                "[%s] peak %d/%d %s(%dF) P=%.0f km h'=%.0f+-%.0f km fv=%.3f MHz "
                "SNR=%.1f dB S4=%.2f(%s) sigma_phi=%.2f(%s)%s",
                self.station_id, peak_index, peak_count, fix.mode_layer,
                fix.n_hops, fix.group_range_km, fix.virtual_height_km,
                fix.virtual_height_uncertainty_km,
                fix.equivalent_vertical_freq_mhz, detection.snr_db,
                scint.s4_index, scint.s4_severity,
                scint.sigma_phi_rad, scint.sigma_phi_severity,
                " EVENT" if scint.scintillation_event else "")
        return last_path, rows

    def _ch_row_for(self, *, timestamp, detection, fix, scintillation,
                    peak_index: int, peak_count: int,
                    dechirp_sweeps_rejected: int = 0) -> dict:
        return build_spots_row(
            timestamp=timestamp, host_call=self.host_call,
            host_grid=self.host_grid, radiod_id=self.radiod_id,
            reporter_id=self.reporter_id,
            processing_version=self.processing_version,
            station_id=self.station_id, center_freq_hz=self.center_freq_hz,
            sweep_rate_hz_per_s=self.sweep_rate_hz_per_s,
            coherent_seconds=self.coherent_seconds, detection=detection,
            fix=fix, scintillation=scintillation, peak_index=peak_index,
            peak_count=peak_count,
            dechirp_sweeps_rejected=dechirp_sweeps_rejected)

    def close(self) -> None:
        self.writer.close()
