"""The picker: rank channels by how many other T6 stations also hear them.

While offsets get established, a channel whose transmitters other T6 stations
hear teaches the solver more than a channel only this station hears.  Once
offsets settle, phase "spread" prefers the channels fewest others hear.

data/t6-stations.toml holds grid-square centres, never precise positions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from hamsci_dsp.geometry import great_circle_km

from chirp_recorder.core.catalogue import Transmitter, load_seed
from chirp_recorder.core.channels import ChannelPlan, form_channels

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

log = logging.getLogger(__name__)

T6_PATH = Path(__file__).resolve().parents[3] / "data" / "t6-stations.toml"

# (low Hz, high Hz, km).  Starting values (decision 4, 2026-11-19): K3LR heard
# 4.46 MHz ground wave at 550-880 km on 11-07; 300 and 150 km follow CODAR's
# published ocean ranges, doubled for night skywave overlap.  Evidence may
# move them; [picker] reach_km in the instance file overrides them.
DEFAULT_REACH_KM: tuple[tuple[float, float, float], ...] = (
    (3e6, 6e6, 900.0), (6e6, 15e6, 300.0), (15e6, 60e6, 150.0))

AUDIBLE_KM = 4000.0   # codar-sounder's audible range

PHASES = ("establish", "spread")


@dataclass(frozen=True)
class T6Station:
    reporter_id: str
    lat: float
    lon: float


@dataclass(frozen=True)
class ChannelChoice:
    plan: ChannelPlan
    shared_t6: int
    nearest_km: float
    reason: str


def load_t6_stations(path: Optional[Path] = None) -> list[T6Station]:
    p = Path(path) if path is not None else T6_PATH
    with open(p, "rb") as f:
        doc = tomllib.load(f)
    return [T6Station(rid, float(s["lat"]), float(s["lon"]))
            for rid, s in sorted((doc.get("stations") or {}).items())]


def reach_km(freq_hz: float, table=DEFAULT_REACH_KM) -> float:
    """Distance within which another T6 station hears a transmitter at freq_hz.
    A frequency outside the table takes the nearest edge row."""
    for lo, hi, km in table:
        if lo <= freq_hz < hi:
            return km
    rows = sorted(table)
    return rows[0][2] if freq_hz < rows[0][0] else rows[-1][2]


def _band(freq_hz: float, table) -> tuple[float, float]:
    for lo, hi, _ in table:
        if lo <= freq_hz < hi:
            return (lo, hi)
    rows = sorted(table)
    return rows[0][:2] if freq_hz < rows[0][0] else rows[-1][:2]


def _reason(n: int, heard: list[str]) -> str:
    if n == 0:
        return "no other T6 station hears this channel"
    noun = "station" if n == 1 else "stations"
    verb = "hears" if n == 1 else "hear"
    return f"{n} other T6 {noun} {verb} {', '.join(heard)}"


def rank(plans: list[ChannelPlan], txs: dict[str, Transmitter], *,
         receiver_lat: float, receiver_lon: float, self_reporter: str,
         t6: list[T6Station], phase: str = "establish",
         reach_table=DEFAULT_REACH_KM, reach_source: str = "default"
         ) -> list[ChannelChoice]:
    if phase not in PHASES:
        raise ValueError(f"picker phase {phase!r} must be one of {PHASES}")
    others = [s for s in t6 if s.reporter_id != self_reporter]
    logged: set[tuple[float, float]] = set()
    out: list[ChannelChoice] = []
    for plan in plans:
        reach = reach_km(plan.centre_hz, reach_table)
        band = _band(plan.centre_hz, reach_table)
        if band not in logged:
            logged.add(band)
            log.info("picker: T6 hearing reach %g km at %.3f MHz (%s)",
                     reach, plan.centre_hz / 1e6, reach_source)
        members = [txs[t] for g in plan.groups for t in g.tx_ids if t in txs]
        stations: set[str] = set()
        heard: list[str] = []
        for t in members:
            near = [s.reporter_id for s in others
                    if great_circle_km(t.lat, t.lon, s.lat, s.lon) <= reach]
            if near:
                stations.update(near)
                heard.append(t.id)
        nearest = min((great_circle_km(receiver_lat, receiver_lon, t.lat, t.lon)
                       for t in members), default=float("inf"))
        out.append(ChannelChoice(plan, len(stations), nearest,
                                 _reason(len(stations), heard)))
    sign = -1 if phase == "establish" else 1
    out.sort(key=lambda c: (sign * c.shared_t6, c.nearest_km))
    return out


class ChannelMenu(Protocol):
    def choices(self, *, receiver_lat: float, receiver_lon: float, reporter_id: str,
                status: str, hour_utc: int) -> list[ChannelChoice]: ...


class VendoredMenu(ChannelMenu):
    """The vendored seed, transmitters within AUDIBLE_KM, ranked.  The scout's
    menu later becomes a second implementation of ChannelMenu."""

    def __init__(self, seed: Optional[dict[str, Transmitter]] = None,
                 t6: Optional[list[T6Station]] = None, *, phase: str = "establish",
                 reach_table=DEFAULT_REACH_KM, reach_source: str = "default"):
        self.seed = seed if seed is not None else load_seed()
        self.t6 = t6 if t6 is not None else load_t6_stations()
        self.phase = phase
        self.reach_table = reach_table
        self.reach_source = reach_source

    def choices(self, *, receiver_lat: float, receiver_lon: float, reporter_id: str,
                status: str, hour_utc: int) -> list[ChannelChoice]:
        near = [t for t in self.seed.values()
                if great_circle_km(receiver_lat, receiver_lon, t.lat, t.lon) <= AUDIBLE_KM]
        return rank(form_channels(near, status), self.seed,
                    receiver_lat=receiver_lat, receiver_lon=receiver_lon,
                    self_reporter=reporter_id, t6=self.t6, phase=self.phase,
                    reach_table=self.reach_table, reach_source=self.reach_source)
