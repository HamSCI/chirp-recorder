#!/usr/bin/env python3
"""Crop the SIGMA fine-channel waterfall to the 4.46 MHz CODAR band.

Source: the scout-cost run (chirp-registry task), 410 frames x 13 334 bins
of float32 dB, bin k at 3.46 MHz + k x 150 Hz, frame arrival times in Unix
seconds.  Output: tests/data/scout_sigma_4460k.npz holding power_db
(float16), t_s (float64), f0_hz and bin_hz, 4.42-4.52 MHz.  It carries no
position.

    .venv/bin/python scripts/crop_scout_fixture.py \\
        ~/hamsci/tasks/chirp-registry/scout-cost/waterfall_fine.npy \\
        ~/hamsci/tasks/chirp-registry/scout-cost/waterfall_fine_t.npy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SRC_F0_HZ = 3.46e6
BIN_HZ = 150.0
LO_HZ, HI_HZ = 4.42e6, 4.52e6
OUT = Path(__file__).resolve().parents[1] / "tests" / "data" / "scout_sigma_4460k.npz"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("waterfall", type=Path)
    ap.add_argument("times", type=Path)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    w = np.load(a.waterfall, mmap_mode="r")
    t = np.load(a.times)
    k0 = int(round((LO_HZ - SRC_F0_HZ) / BIN_HZ))
    k1 = int(round((HI_HZ - SRC_F0_HZ) / BIN_HZ)) + 1
    np.savez_compressed(a.out, power_db=np.asarray(w[:, k0:k1], dtype=np.float16),
                        t_s=np.asarray(t, dtype=np.float64),
                        f0_hz=np.float64(SRC_F0_HZ + k0 * BIN_HZ), bin_hz=np.float64(BIN_HZ))
    print(f"{a.out}: {k1 - k0} bins x {w.shape[0]} frames, {a.out.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
