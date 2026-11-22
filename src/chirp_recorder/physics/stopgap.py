"""PhysicsStopgap: feeds codar-sounder's per-transmitter pipeline from
chirp-recorder's own intervals.

`feed` joins contiguous intervals into one CPI of `coherent_seconds`, mixes
the CPI from the channel centre to each transmitter's centre, and runs the
moved pipeline.  It returns the JSONL paths it wrote and the codar.spots rows;
the channel worker hands the rows to RowWriter.put (pre-flight C12).  A
physics failure logs one warning per CPI and never stops the caller.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from chirp_recorder import __version__
from chirp_recorder.config import ConfigError
from chirp_recorder.core.catalogue import Transmitter
from chirp_recorder.core.channels import ChannelPlan
from chirp_recorder.core.source import Interval
from chirp_recorder.core.timing import Label
from chirp_recorder.physics.pipeline import TransmitterPipeline

log = logging.getLogger(__name__)

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
#: Intervals count as contiguous within half an input sample.
_INDEX_TOL = 0.5


@dataclass(frozen=True)
class PhysicsOut:
    paths: list[Path] = field(default_factory=list)
    spots: list[dict] = field(default_factory=list)


def utc_datetime(utc_ns: int) -> datetime:
    """tz-aware UTC datetime to the microsecond (rounded)."""
    return _EPOCH + timedelta(microseconds=(int(utc_ns) + 500) // 1000)


class PhysicsStopgap:
    def __init__(self, plan: ChannelPlan, txs: dict[str, Transmitter], *,
                 settings: dict, output_dir: Path, reporter_id: str, radiod: str,
                 host_call: str, host_grid: str):
        self.plan = plan
        self.coherent_seconds = float(settings.get("coherent_seconds", 60))
        self.interval_s = settings.get("interval_s")
        self._need: Optional[int] = None
        if self.interval_s is not None:
            self._need = self._intervals_per_cpi(float(self.interval_s))
        self.enabled = True
        src = settings.get("position_source")
        if src not in ("precise", "grid_centre", "none"):
            raise ConfigError("physics settings need position_source "
                              "(precise, grid_centre or none)")
        grid_len = len(host_grid.strip())
        if src == "precise" or (src == "grid_centre" and grid_len == 6):
            if "receiver_lat" not in settings or "receiver_lon" not in settings:
                raise ConfigError("physics settings need receiver_lat and receiver_lon")
        if src == "grid_centre" and grid_len == 6:
            log.warning(
                "physics on %s runs from the centre of grid square %s, not a precise "
                "position: ground distance errs up to about 4 km, virtual height by "
                "about 3-6 km.  Set receiver_lat and receiver_lon to remove it.",
                reporter_id, host_grid)
        elif src != "precise":
            self.enabled = False
            log.warning(
                "physics off on %s: no precise position and no 6-character grid square "
                "(grid %r); a 4-character square would put virtual height wrong by "
                "tens of km.  Set receiver_lat and receiver_lon.", reporter_id, host_grid)
            self._need = None
            self._pipes = []
            self._ivs = []
            self._first_label = None
            return
        lat, lon = settings["receiver_lat"], settings["receiver_lon"]
        offsets = settings.get("tdma_offset_samples") or {}
        version = f"chirp-recorder/{__version__}"
        self._pipes: list[tuple[TransmitterPipeline, float]] = []
        for g in plan.groups:
            for tx_id in g.tx_ids:
                tx = txs[tx_id]
                cfg = {"id": tx.id, "center_freq_hz": int(round(tx.centre_hz)),
                       "sweep_rate_hz_per_s": tx.rate_hz_per_s,
                       "sweep_bw_hz": tx.sweep_bw_hz,
                       "sweep_repetition_hz": 1.0 / tx.period_s,
                       "tx_lat_deg": tx.lat, "tx_lon_deg": tx.lon,
                       "sweep_s": tx.waveform().sweep_s,
                       # sweep-start phase after the UTC second (dechirp.make_replica)
                       "tdma_offset_samples": int(offsets.get(tx.id, 0))}
                pipe = TransmitterPipeline(
                    receiver_lat=lat, receiver_lon=lon, radiod_id=radiod,
                    radiod_status_dns=plan.status, tx_config=cfg,
                    output_dir=Path(output_dir) / "physics",
                    sample_rate_hz=float(plan.sample_rate_hz),
                    coherent_seconds=self.coherent_seconds,
                    range_min_km=float(settings.get("range_min_km", 200)),
                    range_max_km=float(settings.get("range_max_km", 800)),
                    snr_threshold_db=float(settings.get("snr_threshold_db", 5.0)),
                    clutter_window=int(settings.get("clutter_window", 20)),
                    max_peaks_per_cpi=int(settings.get("max_peaks_per_cpi", 4)),
                    host_call=host_call, host_grid=host_grid,
                    processing_version=version, reporter_id=reporter_id)
                self._pipes.append((pipe, tx.centre_hz - plan.centre_hz))
        self._ivs: list[Interval] = []
        self._first_label: Optional[Label] = None

    def _intervals_per_cpi(self, interval_s: float) -> int:
        n = self.coherent_seconds / interval_s
        if interval_s <= 0 or n < 1 or abs(n - round(n)) > 1e-9:
            raise ConfigError(
                f"[physics] coherent_seconds = {self.coherent_seconds:g} must hold a "
                f"whole number of {interval_s:g} s intervals")
        return int(round(n))

    @staticmethod
    def _step(iv: Interval, rate: int) -> float:
        """Input samples one interval spans."""
        return len(iv.samples) * iv.input_samprate / rate

    def _contiguous(self, iv: Interval) -> bool:
        last = self._ivs[-1]
        return (iv.input_epoch_gps_s == last.input_epoch_gps_s
                and abs(iv.first_input_index
                        - (last.first_input_index + self._step(last, self.plan.sample_rate_hz)))
                <= _INDEX_TOL)

    def feed(self, iv: Interval, label: Label) -> PhysicsOut:
        if not self.enabled:
            return PhysicsOut()
        if self._need is None:
            self._need = self._intervals_per_cpi(
                len(iv.samples) / self.plan.sample_rate_hz)
        if self._ivs and not self._contiguous(iv):
            self._ivs = []                      # a break discards the partial CPI
        if not self._ivs:
            self._first_label = label
        self._ivs.append(iv)
        if len(self._ivs) < self._need:
            return PhysicsOut()
        samples = np.concatenate([i.samples for i in self._ivs])
        start_ns = int(self._first_label.utc_ns)
        self._ivs, self._first_label = [], None
        return self._process(samples, start_ns)

    def _process(self, samples: np.ndarray, start_ns: int) -> PhysicsOut:
        start = utc_datetime(start_ns)
        paths: list[Path] = []
        spots: list[dict] = []
        failures: list[str] = []
        n = np.arange(samples.size)
        fs = float(self.plan.sample_rate_hz)
        for pipe, shift_hz in self._pipes:
            try:
                x = samples
                if shift_hz:
                    x = (samples * np.exp(-2j * np.pi * shift_hz * n / fs)).astype(np.complex64)
                path, rows = pipe.process_cpi(x, start, cpi_start_ns=start_ns)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{pipe.station_id}: {exc}")
                continue
            if path is not None:
                paths.append(path)
            spots.extend(rows)
        if failures:
            log.warning("physics failed for CPI %s on %s (%s)", start.isoformat(),
                        self.plan.name, "; ".join(failures))
        return PhysicsOut(paths=paths, spots=spots)

    def close(self) -> None:
        for pipe, _ in self._pipes:
            pipe.close()
