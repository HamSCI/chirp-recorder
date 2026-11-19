#!/usr/bin/env python3
"""Calibrate BIAS_CELLS in chirp_recorder.core.detect.

Runs detect_group on coherent pairs (a strong arrival and a weaker one with a
constant relative phase, the worst case) over a grid of separations and level
differences.  For each kept arrival it prints the delay error, the cross
amplitude detect computes for that pair, and K = |error| * B / cross: the
bias in resolution cells (1/B) per unit of cross amplitude.  BIAS_CELLS should
cover the large values of K; it also reports how many errors above the
arrival's delay_sigma_ns, and above 500 ns, the flag caught.

    uv run python scripts/calibrate_overlap.py [--seeds 3] [--seconds 8]

The default grid uses the 4.463 MHz group (1 s period, 1945.6 us gate) at
64 kS/s and delays on whole samples, so the sub-sample bias drops out.
"""
from __future__ import annotations

import argparse

import numpy as np
from hamsci_dsp.chirp.clean import Component
from hamsci_dsp.chirp.waveform import Waveform, gated

from chirp_recorder.core import detect
from chirp_recorder.core.channels import Group

WF = Waveform(4.463e6, -25_733.913, 1.0, None, 1945.6e-6, 0.5)
FS = 64_000


def synth(arrivals, seconds, seed, noise=3.0):
    rng = np.random.default_rng(seed)
    x = np.zeros(seconds * FS, complex)
    for tau, amp in arrivals:
        x += amp * np.exp(2j * np.pi * rng.random()) * np.tile(gated(WF, FS, tau), seconds)
    x += noise * (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size)) / np.sqrt(2)
    return x.astype(np.complex64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seconds", type=int, default=8)
    ap.add_argument("--offsets", type=int, nargs="*", default=[10, 13, 19, 26, 38, 64, 100])
    ap.add_argument("--db", type=float, nargs="*", default=[3, 6, 10, 15, 20])
    a = ap.parse_args()
    g, B, s0 = Group(WF, ("A",)), WF.bandwidth_hz(), 264 / FS
    rows = []
    print("offset_us  weaker_db  seed  which   err_ns  sigma_ns  cross   K_cells  flag")
    for d in a.offsets:
        for db in a.db:
            for seed in range(1, a.seeds + 1):
                t1 = s0 + d / FS
                r = detect.detect_group(synth([(s0, 1.0), (t1, 10 ** (-db / 20))], a.seconds, seed),
                                        fs=FS, channel_centre_hz=WF.centre_hz, group=g)
                if len(r.arrivals) != 2:
                    continue
                arr = sorted(r.arrivals, key=lambda x: x.level_db)
                weak, strong = arr
                for name, me, other, truth in (("weak", weak, strong, t1), ("strong", strong, weak, s0)):
                    cm = Component(me.tau_ns / 1e9, me.level_db, "A")
                    co = Component(other.tau_ns / 1e9, other.level_db, "A")
                    cross = detect._cross_amplitude(cm, co, WF, FS)       # coherent: factor 1
                    err = me.tau_ns - round(truth * 1e9)
                    k = abs(err) * 1e-9 * B / cross if cross > 0 else float("nan")
                    rows.append((abs(err), me.delay_sigma_ns, k, me.overlap_flag))
                    print(f"{d / FS * 1e6:9.0f} {db:10.0f} {seed:5d} {name:7s} {err:7d} "
                          f"{me.delay_sigma_ns:9.0f} {cross:7.3f} {k:8.2f}  {me.overlap_flag}")
    if not rows:
        return
    e, sg, k, fl = (np.array(c) for c in zip(*rows))
    print(f"\nBIAS_CELLS now {detect.BIAS_CELLS}; K_cells: median {np.nanmedian(k):.2f}, "
          f"90th pct {np.nanpercentile(k, 90):.2f}, max {np.nanmax(k):.2f}")
    for name, bad in (("above own sigma", e > sg), ("above 500 ns", e > 500)):
        print(f"errors {name}: {bad.sum()}, flagged {(bad & fl).sum()}")


if __name__ == "__main__":
    main()
