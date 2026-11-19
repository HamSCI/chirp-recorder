"""Capacity before adding: admit only the channels this machine has room for.

read_headroom reads what the machine has free.  admit walks the plans in the
picker's order and keeps each one while the measured costs of the code version
fit inside LIMITS.  costs.toml holds the measured rows.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from chirp_recorder.core.channels import ChannelPlan

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

log = logging.getLogger(__name__)

COSTS_PATH = Path(__file__).resolve().parents[3] / "data" / "costs.toml"

LIMITS = {"cpu_psi_max": 10.0, "mem_psi_max": 5.0, "mem_floor_fraction": 0.15,
          "spare_cores": 1.0, "unit_mem_fraction": 0.8}


@dataclass(frozen=True)
class Headroom:
    online_cpus: int
    load1: float
    cpu_psi_some_avg60: float
    mem_psi_some_avg60: float
    mem_total: int
    mem_available: int
    unit_memory_max: Optional[int]


@dataclass(frozen=True)
class ChannelCost:
    cores: float
    mem_bytes: int
    version: str
    measured: bool


def _psi_avg60(path: Path) -> float:
    """The `some` line's avg60; a missing file reads 0.0 (logged at DEBUG)."""
    try:
        text = path.read_text()
    except OSError:
        log.debug("no pressure file %s; reading 0.0", path)
        return 0.0
    for line in text.splitlines():
        if line.startswith("some "):
            m = re.search(r"avg60=([0-9.]+)", line)
            if m:
                return float(m.group(1))
    log.debug("no 'some avg60' in %s; reading 0.0", path)
    return 0.0


def _meminfo_bytes(text: str, key: str) -> int:
    m = re.search(rf"^{key}:\s+(\d+)\s*kB", text, re.M)
    if not m:
        raise ValueError(f"meminfo has no {key}")
    return int(m.group(1)) * 1024


def _own_cgroup() -> Path:
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3 and parts[0] == "0":
                return Path("/sys/fs/cgroup") / parts[2].lstrip("/")
    except OSError:
        pass
    return Path("/sys/fs/cgroup")


def _unit_memory_max(cgroup: Path) -> Optional[int]:
    try:
        raw = (cgroup / "memory.max").read_text().strip()
    except OSError:
        return None
    return None if raw == "max" or not raw.isdigit() else int(raw)


def read_headroom(proc: Path = Path("/proc"), cgroup: Optional[Path] = None) -> Headroom:
    if cgroup is None:
        cgroup = _own_cgroup()
    meminfo = (proc / "meminfo").read_text()
    load1 = float((proc / "loadavg").read_text().split()[0])
    return Headroom(
        online_cpus=os.cpu_count() or 1,
        load1=load1,
        cpu_psi_some_avg60=_psi_avg60(proc / "pressure" / "cpu"),
        mem_psi_some_avg60=_psi_avg60(proc / "pressure" / "memory"),
        mem_total=_meminfo_bytes(meminfo, "MemTotal"),
        mem_available=_meminfo_bytes(meminfo, "MemAvailable"),
        unit_memory_max=_unit_memory_max(cgroup),
    )


def load_costs(path: Optional[Path] = None) -> dict[str, dict[int, tuple[float, int]]]:
    with open(path or COSTS_PATH, "rb") as fh:
        doc = tomllib.load(fh)
    out: dict[str, dict[int, tuple[float, int]]] = {}
    for version, body in doc.get("versions", {}).items():
        rates = {int(rate): (float(row["cores"]), int(row["mem_mib"]) << 20)
                 for rate, row in body.get("rates", {}).items()}
        if rates:
            out[version] = rates
    return out


def _vkey(version: str) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", version))


def cost_for(version: str, sample_rate_hz: int, costs: dict) -> Optional[ChannelCost]:
    """The version's own row when measured; else the newest older version's.

    Within a version, a rate with no row takes the smallest measured rate at
    or above it.  Anything but an exact version-and-rate hit is an estimate.
    """
    def pick(rates: dict[int, tuple[float, int]], ver: str, exact_ver: bool):
        if sample_rate_hz in rates:
            cores, mem = rates[sample_rate_hz]
            return ChannelCost(cores, mem, ver, exact_ver)
        above = [r for r in rates if r >= sample_rate_hz]
        if not above:
            return None
        cores, mem = rates[min(above)]
        return ChannelCost(cores, mem, ver, False)

    if version in costs and (c := pick(costs[version], version, True)):
        return c
    want = _vkey(version)
    older = sorted((v for v in costs if v != version and _vkey(v) < want),
                   key=_vkey, reverse=True)
    for v in older:
        if c := pick(costs[v], v, False):
            return ChannelCost(c.cores, c.mem_bytes, v, False)
    return None


@dataclass(frozen=True)
class Admission:
    admitted: list[ChannelPlan]
    refused: list[tuple[ChannelPlan, str]]
    estimated: bool

    def to_dict(self) -> dict:
        return {"admitted": [p.name for p in self.admitted],
                "refused": [[p.name, why] for p, why in self.refused],
                "estimated": self.estimated}


def admit(plans: list[ChannelPlan], head: Headroom, costs: dict, version: str, *,
          limits: dict = LIMITS, measure: bool = False) -> Admission:
    if measure:
        return Admission(list(plans), [], True)
    if head.cpu_psi_some_avg60 > limits["cpu_psi_max"]:
        why = f"cpu pressure {head.cpu_psi_some_avg60:.1f} %"
        return Admission([], [(p, why) for p in plans], False)
    if head.mem_psi_some_avg60 > limits["mem_psi_max"]:
        why = f"memory pressure {head.mem_psi_some_avg60:.1f} %"
        return Admission([], [(p, why) for p in plans], False)

    admitted: list[ChannelPlan] = []
    refused: list[tuple[ChannelPlan, str]] = []
    estimated = False
    cores = 0.0
    mem = 0
    for plan in plans:
        cost = cost_for(version, plan.sample_rate_hz, costs)
        if cost is None:
            if admitted:
                refused.append((plan, f"no measured cost for chirp-recorder {version}"))
            else:
                admitted.append(plan)   # the first plan opens, so it can be measured
            continue
        c, m = cores + cost.cores, mem + cost.mem_bytes
        if head.load1 + c > head.online_cpus - limits["spare_cores"]:
            refused.append((plan, f"cpu headroom: load {head.load1:.2f} + {c:.2f} "
                                  f"cores exceeds {head.online_cpus} CPUs less "
                                  f"{limits['spare_cores']:g} spare"))
        elif head.mem_available - m < limits["mem_floor_fraction"] * head.mem_total:
            refused.append((plan, "memory headroom: free memory would fall below "
                                  f"{limits['mem_floor_fraction']:.0%} of total"))
        elif (head.unit_memory_max is not None
              and m > limits["unit_mem_fraction"] * head.unit_memory_max):
            refused.append((plan, "unit memory cap: channels would exceed "
                                  f"{limits['unit_mem_fraction']:.0%} of memory.max"))
        else:
            admitted.append(plan)
            cores, mem = c, m
            estimated = estimated or not cost.measured
    return Admission(admitted, refused, estimated)
