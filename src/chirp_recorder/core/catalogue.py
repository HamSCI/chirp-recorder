"""The transmitter catalogue: what each CODAR site radiates and where it stands.

The seed (data/chirp-seed.toml) comes from scripts/rebuild-seed.py.  It holds
no offsets: a station never writes an identity or a delay it assumes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hamsci_dsp.chirp.waveform import Waveform

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

SEED_PATH = Path(__file__).resolve().parents[3] / "data" / "chirp-seed.toml"


@dataclass(frozen=True)
class Transmitter:
    id: str
    kind: str
    centre_hz: float
    rate_hz_per_s: float
    period_s: float
    sweep_bw_hz: float
    lat: float
    lon: float
    gate_period_s: Optional[float] = None

    def waveform(self) -> Waveform:
        # A sweep clearly shorter than its period names itself (sweep_s);
        # a full-period sweep leaves it None, as the group key expects.
        sweep_s = None
        if self.rate_hz_per_s:
            s = self.sweep_bw_hz / abs(self.rate_hz_per_s)
            if s < self.period_s * 0.99:
                sweep_s = s
        return Waveform(self.centre_hz, self.rate_hz_per_s, self.period_s,
                        sweep_s=sweep_s, gate_period_s=self.gate_period_s)


def load_seed(path: Optional[Path] = None) -> dict[str, Transmitter]:
    p = Path(path) if path is not None else SEED_PATH
    with open(p, "rb") as f:
        doc = tomllib.load(f)
    out: dict[str, Transmitter] = {}
    for tid, t in doc.get("transmitters", {}).items():
        out[tid] = Transmitter(
            id=tid, kind=t["kind"], centre_hz=float(t["centre_hz"]),
            rate_hz_per_s=float(t["rate_hz_per_s"]), period_s=float(t["period_s"]),
            sweep_bw_hz=float(t["sweep_bw_hz"]), lat=float(t["lat"]),
            lon=float(t["lon"]),
            gate_period_s=(float(t["gate_period_s"]) if "gate_period_s" in t else None))
    return out
