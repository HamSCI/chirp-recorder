"""Sigmond client contract 0.8: the inventory and validate JSON builders.

  section 3   inventory --json   per-instance resource view
  section 12  validate --json    config validation

One instance per receiver.  Each instance entry lists every channel it opens,
with each channel's transmitter groups, the capacity admission against live
headroom, and the chain delay coordination.env names for each of its radiods.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from chirp_recorder import __version__
from chirp_recorder.config import DEFAULT_ETC, ConfigError, InstanceConfig, load_instance
from chirp_recorder.core import timing
from chirp_recorder.core.capacity import admit, load_costs, read_headroom
from chirp_recorder.core.channels import ChannelPlan
from chirp_recorder.core.daemon import applied_state_path
from chirp_recorder.version import GIT_INFO

CONTRACT_VERSION = "0.8"
DEPLOY_TOML = "/opt/git/sigmond/chirp-recorder/deploy.toml"
LOG_DIR = "/var/log/chirp-recorder"


def instance_keys(etc: Path) -> list[str]:
    """Keys of every per-instance file in etc, sorted."""
    return sorted(p.stem for p in Path(etc).glob("*.toml"))


def channel_dicts(channels: list[ChannelPlan]) -> list[dict]:
    return [{
        "name": c.name, "radiod": c.status, "centre_hz": c.centre_hz,
        "sample_rate_hz": c.sample_rate_hz,
        "groups": [{"group_key": g.waveform.group_key(), "tx_ids": list(g.tx_ids)}
                   for g in c.groups],
    } for c in channels]


def capacity_for(cfg: InstanceConfig) -> dict:
    """Admission.to_dict() (pre-flight C19) against live headroom."""
    try:
        costs = load_costs()
    except Exception:  # noqa: BLE001  an unreadable costs file reads as nothing measured
        costs = {}
    return admit(cfg.channels, read_headroom(), costs, __version__).to_dict()


def chain_delays(cfg: InstanceConfig, coordination_env: Optional[Path] = None) -> dict:
    """chain_delay_ns per radiod id, for each radiod coordination.env gives one."""
    env = coordination_env if coordination_env is not None else timing.COORDINATION_ENV
    out: dict[str, int] = {}
    for status in sorted({c.status for c in cfg.channels}):
        rid = timing.radiod_id_for_status(status, env)
        ns = timing.chain_delay_from_env(rid, env)
        if ns is not None:
            out[rid] = ns
    return out


def _read_applied(cfg: InstanceConfig) -> Optional[dict]:
    from hamsci_dsp.timing import read_applied_state
    return read_applied_state(applied_state_path(cfg.output_dir))


def _config_issue(key: str, exc: ConfigError) -> list[dict]:
    lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
    return [{"severity": "fail", "instance": key, "message": m}
            for m in (lines[1:] if len(lines) > 1 else lines)]


def _warnings(cfg: InstanceConfig) -> list[dict]:
    out = []
    if cfg.position_source != "precise":
        out.append({"severity": "warn", "instance": cfg.key,
                    "message": "no precise position: timing offsets stay withheld until "
                               "[station] receiver_lat and receiver_lon are set"})
    if not (cfg.raw.get("station") or {}).get("callsign"):
        out.append({"severity": "warn", "instance": cfg.key,
                    "message": "[station] callsign is empty"})
    return out


def build_inventory(keys: list[str], etc: Path = DEFAULT_ETC, *,
                    coordination_env: Optional[Path] = None) -> dict:
    instances: list[dict] = []
    issues: list[dict] = []
    log_paths: dict[str, Any] = {}
    config_paths: dict[str, str] = {}
    for key in keys:
        config_paths[key] = str(Path(etc) / f"{key}.toml")
        try:
            cfg = load_instance(key, etc)
        except ConfigError as exc:
            issues.extend(_config_issue(key, exc))
            instances.append({"instance": key, "reporter_id": None,
                              "uses_timing_calibration": True,
                              "provides_timing_calibration": False,
                              "ka9q_channels": 0, "channels": [], "frequencies_hz": []})
            continue
        issues.extend(_warnings(cfg))
        statuses = sorted({c.status for c in cfg.channels})
        instances.append({
            "instance": key,
            "reporter_id": cfg.reporter_id,
            "radiod_id": statuses[0],
            "radiod_status_dns": statuses[0],
            "radiod_status_all": statuses,
            "host": "localhost",
            "data_destination": None,
            "frequencies_hz": [int(c.centre_hz) for c in cfg.channels],
            "ka9q_channels": len(cfg.channels),
            "required_cores": [],
            "preferred_cores": "worker",
            "data_sinks": [{"kind": "file", "target": str(cfg.output_dir),
                            "schema_ref": None, "retention_days": 365, "mb_per_day": 5}],
            "uses_timing_calibration": True,
            "provides_timing_calibration": False,
            "timing_authority_applied": _read_applied(cfg),
            "position_source": cfg.position_source,
            "channels": channel_dicts(cfg.channels),
            "capacity": capacity_for(cfg),
            "chain_delay_ns": chain_delays(cfg, coordination_env),
        })
        log_paths[key] = {"process": f"{LOG_DIR}/{key}.log", "products": str(cfg.output_dir)}
    payload: dict[str, Any] = {
        "client": "chirp-recorder",
        "version": __version__,
        "contract_version": CONTRACT_VERSION,
        "config_paths": config_paths,
        "deploy_toml_path": DEPLOY_TOML,
    }
    if GIT_INFO:
        payload["git"] = GIT_INFO
    payload["log_paths"] = log_paths
    payload["log_level"] = logging.getLevelName(logging.getLogger().getEffectiveLevel())
    payload["instances"] = instances
    payload["deps"] = {"pypi": [{"name": "ka9q-python", "version": ">=3.14.0"},
                                {"name": "numpy", "version": ">=1.24.0"},
                                {"name": "hamsci-dsp", "version": ">=0.10.0"}]}
    payload["issues"] = issues
    return payload


def build_validate(keys: list[str], etc: Path = DEFAULT_ETC) -> dict:
    """validate --json.  With one instance in scope, `capacity` and `channels`
    sit at the top level; with several, each instance carries its own."""
    issues: list[dict] = []
    per: dict[str, dict] = {}
    for key in keys:
        try:
            cfg = load_instance(key, etc)
        except ConfigError as exc:
            found = _config_issue(key, exc)
            issues.extend(found)
            per[key] = {"ok": False, "channels": [], "capacity": None}
            continue
        found = _warnings(cfg)
        issues.extend(found)
        per[key] = {"ok": True, "channels": channel_dicts(cfg.channels),
                    "capacity": capacity_for(cfg)}
    if not keys:
        issues.append({"severity": "fail", "instance": "all",
                       "message": f"no instance config in {etc}"})
    payload: dict[str, Any] = {
        "ok": bool(keys) and not any(i["severity"] == "fail" for i in issues),
        "issues": issues,
        "instances": per,
    }
    if len(keys) == 1:
        payload["config_path"] = str(Path(etc) / f"{keys[0]}.toml")
        payload["channels"] = per[keys[0]]["channels"]
        payload["capacity"] = per[keys[0]]["capacity"]
    return payload
