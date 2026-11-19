#!/usr/bin/env python3
"""Write data/chirp-seed.toml from codar-sounder's data/codar-stations.toml.

    uv run scripts/rebuild-seed.py ../codar-sounder/data/codar-stations.toml > data/chirp-seed.toml

The input is the HFRNet live list (11-08 rebuild).  No offsets enter.
"""
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


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
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
