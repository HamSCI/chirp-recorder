#!/usr/bin/env python3
"""Write data/chirp-seed.toml from codar-sounder's data/codar-stations.toml.

    uv run scripts/rebuild-seed.py ../codar-sounder/data/codar-stations.toml > data/chirp-seed.toml

The input is the HFRNet live list (11-08 rebuild).  No offsets enter.

One measured waveform fact enters: the gate period of the Mid-Atlantic
4.463 MHz group (-25.73 kHz/s), 1945.6 us with the gate on at the sweep start
(the 11-03 result, codar-mf/clean.py).  Without it detect.py fits the gate per
capture, and on 5 of K3LR's 16 golden captures the fit locked on the spacing
of two arrivals instead (Task 18).  Other groups keep a fitted gate.
"""
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


MID_ATLANTIC_4463_GATE_S = 1.9456e-3


def gate_period_s(s):
    """The measured gate for the Mid-Atlantic 4.463 MHz group, else None."""
    if (round(float(s["freq_mhz"]) * 1e6) == 4_463_000
            and float(s["sweep_rate_hz_per_s"]) == -25733.913
            and 30.0 <= float(s["tx_lat_deg"]) <= 46.0
            and -80.0 <= float(s["tx_lon_deg"]) <= -66.0):
        return MID_ATLANTIC_4463_GATE_S
    return None


def main(argv):
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    with open(argv[1], "rb") as f:
        stations = tomllib.load(f)["stations"]
    print("# CODAR transmitter seed for chirp-recorder.  Regenerate with")
    print("# scripts/rebuild-seed.py from codar-sounder's data/codar-stations.toml")
    print("# (HFRNet live list, 2026-11-08).  Do not edit.  No offsets live here.")
    for sid in sorted(stations):
        s = stations[sid]
        print(f"\n[transmitters.{sid}]")
        print('kind = "codar"')
        print(f"centre_hz = {float(s['freq_mhz']) * 1e6:.0f}")
        print(f"rate_hz_per_s = {float(s['sweep_rate_hz_per_s'])!r}")
        print(f"period_s = {1.0 / float(s['sweep_repetition_hz'])!r}")
        print(f"sweep_bw_hz = {float(s['sweep_bw_hz'])!r}")
        print(f"lat = {float(s['tx_lat_deg'])!r}")
        print(f"lon = {float(s['tx_lon_deg'])!r}")
        gate = gate_period_s(s)
        if gate is not None:
            print(f"gate_period_s = {gate!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
