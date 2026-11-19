"""Per-instance config: /etc/chirp-recorder/<KEY>.toml.

KEY is the reporter id with '/' and '=' turned to '-' (AC0G/B1A -> AC0G-B1A).
chirp-recorder has no shared legacy config.  load_instance lists every
problem it finds in one ConfigError.

Position.  Timing uses the receiver's precise position (receiver_lat and
receiver_lon).  A grid-square centre serves display and ranking only: with
just a grid, position_source reads "grid_centre" and timing offsets stay
withheld until the operator sets a precise position.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from chirp_recorder._reporter_id import to_key
from chirp_recorder.core.catalogue import Transmitter, load_seed
from chirp_recorder.core.channels import (SAMPLE_RATES_HZ, ChannelPlan,
                                          channel_errors, groups_for)
from chirp_recorder.core.picker import DEFAULT_REACH_KM

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

log = logging.getLogger(__name__)

DEFAULT_ETC = Path("/etc/chirp-recorder")
DEFAULT_DATA = Path("/var/lib/chirp-recorder")


class ConfigError(Exception):
    """A config that cannot run.  The message lists every problem."""


@dataclass
class InstanceConfig:
    key: str
    reporter_id: str
    receiver_lat: float
    receiver_lon: float
    grid: str
    output_dir: Path
    interval_s: float
    channels: list[ChannelPlan]
    picker_phase: str
    max_channels: int
    raw: dict
    position_source: str    # "precise" | "grid_centre"
    reach_km: tuple = DEFAULT_REACH_KM
    reach_source: str = "default"    # "config" when [picker] reach_km was set


def grid_centre(grid: str) -> tuple[float, float]:
    """Centre (lat, lon) of a 4- or 6-character Maidenhead square.

    A 4-character square takes its centre as the 6-character one would
    ('MM' subsquare 'll' middle).  Raises ValueError on a bad grid."""
    g = grid.strip()
    if len(g) not in (4, 6):
        raise ValueError(f"grid square {grid!r} must have 4 or 6 characters")
    g = g[:2].upper() + g[2:4] + g[4:].lower()
    if not ("A" <= g[0] <= "R" and "A" <= g[1] <= "R" and g[2:4].isdigit()):
        raise ValueError(f"bad grid square {grid!r}")
    lon = (ord(g[0]) - 65) * 20 - 180 + int(g[2]) * 2
    lat = (ord(g[1]) - 65) * 10 - 90 + int(g[3])
    if len(g) == 6:
        if not ("a" <= g[4] <= "x" and "a" <= g[5] <= "x"):
            raise ValueError(f"bad grid square {grid!r}")
        lon += (ord(g[4]) - 97) * (2 / 24) + (1 / 24)
        lat += (ord(g[5]) - 97) * (1 / 24) + (1 / 48)
    else:
        lon += 1.0
        lat += 0.5
    return lat, lon


def grid6(lat: float, lon: float) -> str:
    """6-character Maidenhead square holding (lat, lon)."""
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"position {lat}, {lon} is out of range")
    lon_f = min(lon + 180, 360 - 1e-9)
    lat_f = min(lat + 90, 180 - 1e-9)
    f_lon, f_lat = int(lon_f // 20), int(lat_f // 10)
    s_lon, s_lat = int((lon_f % 20) // 2), int(lat_f % 10)
    u_lon = int(((lon_f % 2) * 12) // 1)
    u_lat = int(((lat_f % 1) * 24) // 1)
    return (chr(65 + f_lon) + chr(65 + f_lat) + str(s_lon) + str(s_lat)
            + chr(97 + u_lon) + chr(97 + u_lat))


def _number(v: Any, what: str, lo: float, hi: float, errs: list[str]) -> Optional[float]:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        errs.append(f"[station] {what} must be a number, not {v!r}")
        return None
    v = float(v)
    if not math.isfinite(v) or not (lo <= v <= hi):
        errs.append(f"[station] {what} = {v} must be finite and within [{lo:g}, {hi:g}]")
        return None
    return v


def load_instance(key: str, etc: Path = DEFAULT_ETC,
                  seed: Optional[dict[str, Transmitter]] = None) -> InstanceConfig:
    errs: list[str] = []
    path = Path(etc) / f"{key}.toml"
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except FileNotFoundError:
        raise ConfigError(f"{path}: no such config") from None
    except (tomllib.TOMLDecodeError, OSError) as e:
        raise ConfigError(f"{path}: {e}") from None

    # identity
    rid = (raw.get("instance") or {}).get("reporter_id")
    if not isinstance(rid, str) or not rid.strip():
        errs.append("[instance] reporter_id is missing")
        rid = ""
    else:
        try:
            if to_key(rid) != to_key(key):
                errs.append(f"[instance] reporter_id {rid!r} does not match instance key {key!r}")
        except ValueError as e:
            errs.append(f"[instance] reporter_id: {e}")

    # position
    st = raw.get("station") or {}
    if not isinstance(st, dict):
        errs.append("[station] must be a table")
        st = {}
    grid = st.get("grid_square", "")
    if not isinstance(grid, str):
        errs.append("[station] grid_square must be a string")
        grid = ""
    lat = lon = None
    source = ""
    has_lat, has_lon = "receiver_lat" in st, "receiver_lon" in st
    if has_lat != has_lon:
        errs.append("[station] receiver_lat and receiver_lon must be given together")
    elif has_lat:
        lat = _number(st["receiver_lat"], "receiver_lat", -90, 90, errs)
        lon = _number(st["receiver_lon"], "receiver_lon", -180, 180, errs)
        source = "precise"
    elif grid:
        try:
            lat, lon = grid_centre(grid)
            source = "grid_centre"
        except ValueError as e:
            errs.append(f"[station] grid_square: {e}")
    else:
        errs.append("[station] no position: give receiver_lat and receiver_lon, "
                    "or grid_square")

    # paths, processing, picker
    out_dir = Path((raw.get("paths") or {}).get("output_dir") or DEFAULT_DATA / key)
    interval_s = (raw.get("processing") or {}).get("interval_s", 30)
    if isinstance(interval_s, bool) or not isinstance(interval_s, (int, float)) \
            or not math.isfinite(interval_s) or interval_s <= 0:
        errs.append(f"[processing] interval_s = {interval_s!r} must be a positive number")
        interval_s = 30
    pk = raw.get("picker") or {}
    phase = pk.get("phase", "establish")
    max_channels = pk.get("max_channels", 2)
    if not isinstance(phase, str):
        errs.append("[picker] phase must be a string")
        phase = "establish"
    if isinstance(max_channels, bool) or not isinstance(max_channels, int) or max_channels < 1:
        errs.append(f"[picker] max_channels = {max_channels!r} must be a whole number >= 1")
        max_channels = 2

    reach_table = DEFAULT_REACH_KM
    reach_source = "default"
    if "reach_km" in pk:
        rk = pk["reach_km"]
        parsed = []
        ok = isinstance(rk, list) and bool(rk)
        for row in (rk if ok else []):
            if (not isinstance(row, list) or len(row) != 3 or not all(
                    isinstance(x, (int, float)) and not isinstance(x, bool)
                    and math.isfinite(x) for x in row)):
                ok = False
                break
            parsed.append(tuple(float(x) for x in row))
        if not ok:
            errs.append("[picker] reach_km must be a list of [low_hz, high_hz, km] rows")
        else:
            for lo, hi, km in parsed:
                if not (0 < lo < hi) or km <= 0:
                    errs.append(f"[picker] reach_km row [{lo:g}, {hi:g}, {km:g}]: "
                                "low edge must be positive and below the high edge, "
                                "and km positive")
            if not any("reach_km row" in e for e in errs):
                reach_table = tuple(parsed)
                reach_source = "config"

    # channels
    if seed is None:
        try:
            seed = load_seed()
        except (OSError, ValueError, KeyError, tomllib.TOMLDecodeError) as e:
            errs.append(f"transmitter seed: {e}")
            seed = {}
    channels: list[ChannelPlan] = []
    names: dict[str, int] = {}
    blocks = raw.get("channel") or []
    if not blocks:
        errs.append("no [[channel]] block")
    if not isinstance(blocks, list):
        errs.append("[[channel]] must be an array of tables")
        blocks = []
    for i, blk in enumerate(blocks):
        if not isinstance(blk, dict):
            errs.append(f"[[channel]] #{i + 1}: must be a table")
            continue
        name = blk.get("name")
        if not isinstance(name, str) or not name:
            errs.append(f"[[channel]] #{i + 1}: name is missing")
            name = f"#{i + 1}"
        label = f"[[channel]] {name}"
        names[name] = names.get(name, 0) + 1
        if names[name] == 2:
            errs.append(f"{label}: two channels carry the same name")
        status = blk.get("status")
        centre = blk.get("centre_hz")
        rate = blk.get("sample_rate_hz")
        ids = blk.get("transmitters")
        bad = False
        if not isinstance(status, str) or not status:
            errs.append(f"{label}: status is missing"); bad = True
        if isinstance(centre, bool) or not isinstance(centre, (int, float)) or centre <= 0:
            errs.append(f"{label}: centre_hz is missing or not positive"); bad = True
        if rate not in SAMPLE_RATES_HZ:
            errs.append(f"{label}: sample_rate_hz = {rate!r} must be one of {SAMPLE_RATES_HZ}")
            bad = True
        if not isinstance(ids, list) or not ids or not all(isinstance(x, str) for x in ids):
            errs.append(f"{label}: transmitters must be a non-empty list of ids"); bad = True
            ids = []
        unknown = [x for x in ids if x not in seed]
        for x in unknown:
            errs.append(f"{label}: transmitter {x} is not in the seed catalogue")
        if bad or unknown:
            continue
        plan = ChannelPlan(name, status, float(centre), int(rate),
                           groups_for([seed[x] for x in ids]))
        errs.extend(channel_errors(plan, seed))
        channels.append(plan)

    if errs:
        raise ConfigError(f"{path}:\n  " + "\n  ".join(errs))

    if source == "grid_centre":
        log.warning("instance %s: only grid_square %s given; using its centre for "
                    "display and ranking.  Timing offsets will be withheld until a "
                    "precise position (receiver_lat, receiver_lon) is set.", key, grid)
    return InstanceConfig(
        key=key, reporter_id=rid, receiver_lat=lat, receiver_lon=lon, grid=grid,
        output_dir=out_dir, interval_s=float(interval_s), channels=channels,
        picker_phase=phase, max_channels=max_channels, raw=raw,
        position_source=source, reach_km=reach_table, reach_source=reach_source)
