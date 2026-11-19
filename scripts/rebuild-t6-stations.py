#!/usr/bin/env python3
"""Write data/t6-stations.toml from sigmondb on wd30.

    uv run scripts/rebuild-t6-stations.py > data/t6-stations.toml

Lists reporters whose arrivals carried origin = 't6_anchor' in the last 7 days,
joined with sigmondb.receiver positions (gpsdo preferred over typed_grid).
The file holds each station's 6-character grid and that square's CENTRE,
never the precise position: the file lives in a repo.  Run by hand, like
rebuild-seed.py.  `--from-tsv FILE` reads rows (reporter, pos_source, lat, lon)
from a file instead of querying wd30.
"""
import datetime
import subprocess
import sys

from chirp_recorder.config import grid6, grid_centre

QUERY = """
SELECT r.reporter_id, r.pos_source, r.lat, r.lon
FROM sigmondb.receiver AS r FINAL
WHERE r.reporter_id IN (SELECT DISTINCT reporter_id FROM sigmondb.arrival
                        WHERE origin = 't6_anchor' AND time > now() - INTERVAL 7 DAY)
  AND r.lat IS NOT NULL AND r.lon IS NOT NULL
  AND r.pos_source IN ('gpsdo', 'typed_grid')
FORMAT TSV
""".strip()

PREFERENCE = {"gpsdo": 0, "typed_grid": 1}


def build_row(reporter_id, lat, lon):
    """One station's row: grid6 and that square's centre, not (lat, lon)."""
    grid = grid6(lat, lon)
    clat, clon = grid_centre(grid)
    return {"reporter_id": reporter_id, "grid": grid,
            "lat": round(clat, 5), "lon": round(clon, 5)}


def pick_positions(rows):
    """rows of (reporter_id, pos_source, lat, lon) -> build_row dicts, one per
    reporter, gpsdo before typed_grid."""
    best = {}
    for rid, src, lat, lon in rows:
        rank = PREFERENCE.get(src, 9)
        if rid not in best or rank < best[rid][0]:
            best[rid] = (rank, lat, lon)
    return [build_row(rid, lat, lon) for rid, (_, lat, lon) in sorted(best.items())]


def render(stations, as_of):
    out = ["# T6 stations for chirp-recorder's picker.  Regenerate with",
           "# scripts/rebuild-t6-stations.py.  Do not edit.",
           "# Positions are 6-character grid-square CENTRES, never precise.",
           f'as_of = "{as_of}"']
    for s in stations:
        out += [f'\n[stations."{s["reporter_id"]}"]', f'grid = "{s["grid"]}"',
                f'lat = {s["lat"]!r}', f'lon = {s["lon"]!r}']
    return "\n".join(out) + "\n"


def _parse(tsv):
    rows = []
    for line in tsv.splitlines():
        if line.strip():
            rid, src, lat, lon = line.split("\t")
            rows.append((rid, src, float(lat), float(lon)))
    return rows


def main(argv):
    if len(argv) == 3 and argv[1] == "--from-tsv":
        tsv = open(argv[2]).read()
    elif len(argv) == 1:
        tsv = subprocess.run(["ssh", "wd30", "clickhouse-client"], input=QUERY,
                             check=True, capture_output=True, text=True).stdout
    else:
        print(__doc__, file=sys.stderr)
        return 2
    as_of = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    sys.stdout.write(render(pick_positions(_parse(tsv)), as_of))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
