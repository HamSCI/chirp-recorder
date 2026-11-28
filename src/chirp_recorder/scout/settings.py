"""The scout's [scout] table in an instance config, checked.

scout_settings lists every bad key in one ConfigError, as load_instance
does.  Unknown keys count as errors so a typo cannot fall to a default.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from chirp_recorder.config import ConfigError, InstanceConfig

MAX_VIEW_BINS = 20_000

# 2 MHz windows over the CODAR allocations 4.438-4.488 and 5.250-5.275,
# 9.305-9.355, 13.450-13.550, 16.100-16.200, 24.450-24.600 and 26.200-26.350 MHz.
DEFAULT_TARGETS_MHZ = (4.86, 9.33, 13.50, 16.15, 25.40)
DEFAULT_HOSTS = ("https://db2.wspr.live", "https://wd1.wsprdaemon.org",
                 "https://wd2.wsprdaemon.org")


@dataclass(frozen=True)
class ScoutSettings:
    enabled: bool = False
    status: str = ""
    nyquist_hz: float = 64.8e6
    coarse_bin_hz: float = 4000.0
    coarse_poll_s: float = 0.4
    coarse_window_s: float = 3600.0
    fine_span_hz: float = 2.0e6
    fine_bin_hz: float = 150.0
    fine_poll_s: float = 0.15
    dwell_s: float = 45.0
    min_frames: int = 100
    targets_mhz: tuple = DEFAULT_TARGETS_MHZ
    public_hosts: tuple = DEFAULT_HOSTS
    refresh_s: float = 3600.0
    state_dir: Path = Path("scout")


_FLOATS = ("nyquist_hz", "coarse_bin_hz", "coarse_poll_s", "coarse_window_s",
           "fine_span_hz", "fine_bin_hz", "fine_poll_s", "dwell_s", "refresh_s")


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def scout_settings(cfg: InstanceConfig) -> ScoutSettings:
    """The [scout] table with defaults filled in.  Raises ConfigError."""
    tbl = cfg.raw.get("scout", {})
    errs: list[str] = []
    if not isinstance(tbl, dict):
        raise ConfigError("[scout] must be a table")
    known = set(_FLOATS) | {"enabled", "status", "min_frames", "targets_mhz",
                            "public_hosts", "state_dir"}
    for k in tbl:
        if k not in known:
            errs.append(f"[scout] unknown key {k!r}")
    base = ScoutSettings()
    vals: dict = {}

    enabled = tbl.get("enabled", False)
    if not isinstance(enabled, bool):
        errs.append(f"[scout] enabled = {enabled!r} must be true or false")
        enabled = False
    vals["enabled"] = enabled

    for k in _FLOATS:
        v = tbl.get(k, getattr(base, k))
        if not _is_num(v) or v <= 0:
            errs.append(f"[scout] {k} = {v!r} must be a positive number")
            v = getattr(base, k)
        vals[k] = float(v)
    v = tbl.get("min_frames", base.min_frames)
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        errs.append(f"[scout] min_frames = {v!r} must be a whole number >= 1")
        v = base.min_frames
    vals["min_frames"] = v

    status = tbl.get("status")
    if status is None:
        status = cfg.channels[0].status if cfg.channels else ""
    if not isinstance(status, str):
        errs.append("[scout] status must be a string")
        status = ""
    if enabled and not status:
        errs.append("[scout] status is missing and no [[channel]] supplies one")
    vals["status"] = status

    nyq, fspan = vals["nyquist_hz"], vals["fine_span_hz"]
    if vals["fine_span_hz"] / vals["fine_bin_hz"] > MAX_VIEW_BINS:
        errs.append(f"[scout] fine_bin_hz = {vals['fine_bin_hz']:g} gives "
                    f"{fspan / vals['fine_bin_hz']:.0f} fine bins; the limit is {MAX_VIEW_BINS}")
    if nyq / vals["coarse_bin_hz"] > MAX_VIEW_BINS:
        errs.append(f"[scout] coarse_bin_hz = {vals['coarse_bin_hz']:g} gives "
                    f"{nyq / vals['coarse_bin_hz']:.0f} coarse bins; the limit is {MAX_VIEW_BINS}")

    tg = tbl.get("targets_mhz", base.targets_mhz)
    if (not isinstance(tg, (list, tuple)) or not tg
            or not all(_is_num(t) for t in tg)):
        errs.append("[scout] targets_mhz must be a non-empty list of numbers")
        tg = base.targets_mhz
    else:
        for t in tg:
            lo, hi = t * 1e6 - fspan / 2, t * 1e6 + fspan / 2
            if lo < 0 or hi > nyq:
                errs.append(f"[scout] targets_mhz {t:g}: window {lo:g}..{hi:g} Hz "
                            f"leaves 0..{nyq:g} Hz")
    vals["targets_mhz"] = tuple(float(t) for t in tg)

    hosts = tbl.get("public_hosts", base.public_hosts)
    if (not isinstance(hosts, (list, tuple)) or not hosts
            or not all(isinstance(h, str) and h.startswith("https://") for h in hosts)):
        errs.append("[scout] public_hosts must be a non-empty list of https:// URLs")
        hosts = base.public_hosts
    vals["public_hosts"] = tuple(hosts)

    sd = tbl.get("state_dir")
    if sd is None:
        vals["state_dir"] = Path(cfg.output_dir) / "scout"
    elif isinstance(sd, str) and sd:
        vals["state_dir"] = Path(sd)
    else:
        errs.append("[scout] state_dir must be a path string")
        vals["state_dir"] = Path(cfg.output_dir) / "scout"

    if errs:
        raise ConfigError("[scout]:\n  " + "\n  ".join(errs))
    return ScoutSettings(**vals)
