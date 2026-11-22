"""The side-by-side comparison with codar-sounder, which gates its retirement.

Both clients write sigmondb arrival rows.  Each side's T6 rows form clusters
per (reporter, group), as stage2-groundwave/track.py clusters arrivals: sort
by delay, split where neighbours stand more than 120 us apart, keep a cluster
seen in at least six captures.  The verdict pairs the clusters and holds the
rows of decision 3 (2026-11-19):

  matched    gates   at least min_matched pairs, each within tol_ns (5 us)
  coverage   gates   pairs cover at least 90 % of codar-sounder's
                     unexplained clusters
  doppler    gates   90 % of pairs with a Doppler on both sides agree
                     within 0.02 Hz (about one codar-sounder bin)
  s4, sigma_phi      reported only: median |difference| (and *_max, largest)

One convention.  codar-sounder's delay holds tau - f/k: its stretch
processing puts each path's beat tone at -k*tau + f, and it reads tau as
-f_beat/k (codar-sounder sigmondb.py:91-93).  chirp-recorder's delay_ns holds
tau, with the Doppler removed (detect.py).  So each chirp-recorder row enters
the comparison as delay_ns - doppler_hz / rate_hz_per_s, codar-sounder's
quantity; a chirp-recorder row with no Doppler stays out.

Match mode (provisional, 2026-11-22; Michael decides).  codar-sounder takes
the integer range bin and refines nothing, so its delay moves in steps of
one bin, 1 / (period * |rate|) = 38.9 us for the 4.463 MHz group.
  "bin"    (default) a pair matches when |chirp - codar| <= bin / 2 + tol_ns
  "exact"  a pair matches when |chirp - codar| <= tol_ns (the brief's rule)
The verdict names the mode it used.

A codar-sounder cluster with no partner counts as explained, not missed,
when it sits on a ladder rung of a stronger codar-sounder cluster
(m x rung, 1 <= |m| <= 4, +-0.3 ms) or 1.632 ms after one (the KFS twin
fault, ledger codar-live-faults).
"""
from __future__ import annotations

import json
import math
import sqlite3
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PAIR_WINDOW_NS = 120_000          # clusters closer than this pair up (the cluster gap)
MATCH_MODES = ("bin", "exact")
RUNG_TOL_NS = 300_000
MAX_RUNG = 4
KFS_TWIN_NS = 1_632_000
SIGMONDB = "sigmondb"
ARRIVAL_TABLE = "arrival"


@dataclass(frozen=True)
class Cluster:
    reporter_id: str
    group_key: str
    producer: str
    median_delay_ns: int
    n: int
    median_doppler_hz: Optional[float]
    median_s4: Optional[float]
    median_sigma_phi_rad: Optional[float]
    # Beyond the brief's fields; left out of equality.
    median_level_db: Optional[float] = field(default=None, compare=False)
    spread_ns: Optional[int] = field(default=None, compare=False)   # max - min delay
    rate_hz_per_s: Optional[float] = field(default=None, compare=False)
    period_s: float = field(default=1.0, compare=False)


def period_of(group_key: str) -> float:
    """The sweep period a sigmondb group key names (its "Ns" token; 1 s if none)."""
    for tok in str(group_key).split()[2:]:
        if tok.endswith("s") and not tok.startswith("sweep"):
            try:
                return float(tok[:-1])
            except ValueError:
                pass
    return 1.0


def codar_bin_ns(rate_hz_per_s: float, period_s: float) -> float:
    """codar-sounder's range-bin spacing as a delay, in ns.

    Its dechirp turns FFT bins of one sweep (n = fs * period samples) into
    delay as fftfreq(n, 1/fs) / |rate| (codar-sounder dechirp.py:265-266), so
    neighbouring bins stand fs / n / |rate| = 1 / (period * |rate|) apart."""
    return 1e9 / (float(period_s) * abs(float(rate_hz_per_s)))


def rung_spacings(groups) -> dict[str, float]:
    """{group_key: rung spacing s} for the groups whose gate period is known:
    1 / (|rate| * gate period)."""
    return {g.waveform.group_key(): 1.0 / (abs(g.waveform.rate_hz_per_s)
                                           * g.waveform.gate_period_s)
            for g in groups if g.waveform.gate_period_s}


def to_codar_convention(rows: list[dict]) -> list[dict]:
    """chirp-recorder rows as codar-sounder measures them: delay_ns - f/k,
    folded into the sweep period.  Rows without a Doppler or a rate drop out."""
    out = []
    for r in rows:
        f, k = _f(r.get("doppler_hz")), _f(r.get("rate_hz_per_s"))
        if f is None or not k:
            continue
        p_ns = round(period_of(r.get("group_key", "")) * 1e9)
        out.append(dict(r, delay_ns=(int(r["delay_ns"]) - round(f / k * 1e9)) % p_ns))
    return out


@dataclass(frozen=True)
class VerdictRow:
    name: str
    value: float
    limit: float
    gating: bool
    passed: bool


def _num(x: float):
    """JSON-safe: NaN becomes None."""
    return None if isinstance(x, float) and math.isnan(x) else x


@dataclass(frozen=True)
class Verdict:
    matched: list           # [(chirp Cluster, codar Cluster, diff_ns = chirp - codar)]
    codar_unmatched: list   # [Cluster]
    explained: list         # [(codar Cluster, reason)]
    chirp_only: list        # [Cluster]
    rows: list              # [VerdictRow]
    status: str             # "pass" | "fail" | "insufficient"
    match: str = "bin"      # the match mode the verdict used
    codar_bin_ns: dict = field(default_factory=dict)   # group_key -> codar range bin

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "match": self.match,
            "codar_bin_ns": dict(self.codar_bin_ns),
            "rows": [{"name": r.name, "value": _num(r.value), "limit": _num(r.limit),
                      "gating": r.gating, "passed": r.passed} for r in self.rows],
            "matched": [[a.median_delay_ns, b.median_delay_ns, d, a.group_key]
                        for a, b, d in self.matched],
            "explained": [[c.median_delay_ns, why] for c, why in self.explained],
            "codar_unmatched": [c.median_delay_ns for c in self.codar_unmatched],
            "chirp_only": [c.median_delay_ns for c in self.chirp_only],
        }


def _is_set(v) -> bool:
    return v is not None and v != ""


def _f(v) -> Optional[float]:
    return float(v) if _is_set(v) else None


def _median(vals) -> Optional[float]:
    v = [x for x in vals if x is not None]
    return float(statistics.median(v)) if v else None


def clusters(rows: list[dict], *, producer: str, gap_ns: int = 120_000,
             min_n: int = 6, t6_only: bool = True) -> list[Cluster]:
    """T6 clusters (rows with u_epoch_ns set) per (reporter_id, group_key).

    Sorted by delay, split where neighbours differ by more than gap_ns; a
    cluster needs rows from at least min_n distinct capture seconds.
    t6_only=False takes every row, for a replay placed by radiod's counter,
    whose captures agree with each other but carry no T6 tie."""
    by: dict = {}
    for r in rows:
        if t6_only and not _is_set(r.get("u_epoch_ns")):
            continue
        by.setdefault((r.get("reporter_id", ""), r.get("group_key", "")), []).append(r)
    out: list[Cluster] = []
    for (rid, gkey), rs in sorted(by.items()):
        rs = sorted(rs, key=lambda r: int(r["delay_ns"]))
        runs, cur = [], [rs[0]]
        for r in rs[1:]:
            if int(r["delay_ns"]) - int(cur[-1]["delay_ns"]) > gap_ns:
                runs.append(cur)
                cur = []
            cur.append(r)
        runs.append(cur)
        for run in runs:
            if len({r.get("time") for r in run}) < min_n:
                continue
            d = [int(r["delay_ns"]) for r in run]
            out.append(Cluster(
                reporter_id=rid, group_key=gkey, producer=producer,
                median_delay_ns=int(round(statistics.median(d))), n=len(run),
                median_doppler_hz=_median(_f(r.get("doppler_hz")) for r in run),
                median_s4=_median(_f(r.get("s4")) for r in run),
                median_sigma_phi_rad=_median(_f(r.get("sigma_phi_rad")) for r in run),
                median_level_db=_median(_f(r.get("level_db")) for r in run),
                spread_ns=max(d) - min(d),
                rate_hz_per_s=_median(_f(r.get("rate_hz_per_s")) for r in run),
                period_s=period_of(gkey)))
    return out


def _pair(chirp: list[Cluster], codar: list[Cluster], window_ns=lambda c: PAIR_WINDOW_NS):
    """Nearest chirp cluster for each codar cluster, same reporter and group,
    within window_ns(codar cluster); each chirp cluster pairs at most once."""
    cands = []
    for i, b in enumerate(codar):
        for j, a in enumerate(chirp):
            if (a.reporter_id, a.group_key) != (b.reporter_id, b.group_key):
                continue
            d = a.median_delay_ns - b.median_delay_ns
            if abs(d) <= window_ns(b):
                cands.append((abs(d), i, j, d))
    used_i, used_j, pairs = set(), set(), []
    for _, i, j, d in sorted(cands):
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        pairs.append((chirp[j], codar[i], d))
    return pairs, used_i, used_j


def _strength(c: Cluster):
    return c.median_level_db if c.median_level_db is not None else -math.inf


def _explain(c: Cluster, parents: list[Cluster], rung_s: dict) -> Optional[str]:
    rung_ns = rung_s.get(c.group_key)
    for p in parents:
        if (p.reporter_id, p.group_key) != (c.reporter_id, c.group_key):
            continue
        d = c.median_delay_ns - p.median_delay_ns
        if abs(d - KFS_TWIN_NS) <= RUNG_TOL_NS:
            return f"KFS twin of {p.median_delay_ns / 1e6:.3f} ms"
        if rung_ns:
            L = rung_ns * 1e9
            m = round(d / L)
            if 1 <= abs(m) <= MAX_RUNG and abs(d - m * L) <= RUNG_TOL_NS:
                return f"rung {m:+d} of {p.median_delay_ns / 1e6:.3f} ms"
    return None


def compare(chirp_rows, codar_rows, *, tol_ns: int = 5_000, min_fraction: float = 0.9,
            min_matched: int = 3, doppler_tol_hz: float = 0.02,
            rung_s: dict[str, float], match: str = "bin") -> Verdict:
    if match not in MATCH_MODES:
        raise ValueError(f"match must be one of {MATCH_MODES}, not {match!r}")
    chirp = clusters(to_codar_convention(list(chirp_rows)), producer="chirp-recorder")
    codar = clusters(list(codar_rows), producer="codar-sounder")

    def bin_ns(c: Cluster) -> float:
        return codar_bin_ns(c.rate_hz_per_s, c.period_s) if c.rate_hz_per_s else math.nan

    def limit_ns(c: Cluster) -> float:
        if match == "exact":
            return float(tol_ns)
        b = bin_ns(c)
        return b / 2 + tol_ns if not math.isnan(b) else float(tol_ns)

    pairs, paired_codar, paired_chirp = _pair(
        chirp, codar, lambda c: max(PAIR_WINDOW_NS, limit_ns(c)))
    pairs.sort(key=lambda p: (p[1].group_key, p[1].median_delay_ns))

    # Strongest first; at equal strength a paired cluster comes first, then the
    # more often seen, then the earlier.  Only a cluster earlier in this order
    # explains a later one, so two equal clusters never explain each other.
    order = sorted(range(len(codar)), key=lambda i: (
        -_strength(codar[i]), i not in paired_codar, -codar[i].n, codar[i].median_delay_ns))
    explained, unmatched = [], []
    for k, i in enumerate(order):
        if i in paired_codar:
            continue
        why = _explain(codar[i], [codar[j] for j in order[:k]], rung_s)
        if why:
            explained.append((codar[i], why))
        else:
            unmatched.append(codar[i])
    unmatched.sort(key=lambda c: (c.group_key, c.median_delay_ns))
    chirp_only = [c for j, c in enumerate(chirp) if j not in paired_chirp]

    within = [p for p in pairs if abs(p[2]) <= limit_ns(p[1])]
    worst = float(max((abs(p[2]) for p in pairs), default=math.nan))
    # One group gives one limit; across groups the row shows the largest and
    # every pair still meets its own.
    lim = max((limit_ns(p[1]) for p in pairs), default=limit_ns(codar[0]) if codar
              else float(tol_ns))
    rows = [VerdictRow("matched", worst, float(lim), True,
                       len(pairs) >= min_matched and len(within) == len(pairs))]
    unexplained = len(codar) - len(explained)
    cov = len(within) / unexplained if unexplained else math.nan
    rows.append(VerdictRow("coverage", cov, min_fraction, True,
                           bool(unexplained) and cov >= min_fraction))
    dop = [(a.median_doppler_hz, b.median_doppler_hz) for a, b, _ in pairs
           if a.median_doppler_hz is not None and b.median_doppler_hz is not None]
    agree = (sum(abs(x - y) <= doppler_tol_hz + 1e-12 for x, y in dop) / len(dop)
             if dop else math.nan)
    rows.append(VerdictRow("doppler", agree, 0.9, True, bool(dop) and agree >= 0.9))
    for name, attr in (("s4", "median_s4"), ("sigma_phi", "median_sigma_phi_rad")):
        diffs = [abs(getattr(a, attr) - getattr(b, attr)) for a, b, _ in pairs
                 if getattr(a, attr) is not None and getattr(b, attr) is not None]
        med = float(statistics.median(diffs)) if diffs else math.nan
        big = float(max(diffs)) if diffs else math.nan
        rows.append(VerdictRow(name, med, math.nan, False, True))
        rows.append(VerdictRow(f"{name}_max", big, math.nan, False, True))

    if len(chirp) < min_matched or len(codar) < min_matched or not dop:
        status = "insufficient"         # too few clusters, or no Doppler to judge
    else:
        status = "pass" if all(r.passed for r in rows if r.gating) else "fail"
    bins = {c.group_key: bin_ns(c) for c in codar if c.rate_hz_per_s}
    return Verdict(matched=pairs, codar_unmatched=unmatched, explained=explained,
                   chirp_only=chirp_only, rows=rows, status=status, match=match,
                   codar_bin_ns=bins)


def read_sink_rows(db: Path, *, producer: str, reporter_id: str, since_iso: str,
                   until_iso: str) -> list[dict]:
    """Arrival rows one producer wrote for one reporter, since <= time < until.

    Reads the sink's pending_uploads queue (hs_uploader.sink.Writer's table),
    read-only.  Rows the uploader has already drained are gone from it."""
    uri = f"file:{Path(db)}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        cur = conn.execute(
            "SELECT payload_json FROM pending_uploads "
            "WHERE target_db = ? AND target_table = ? AND producer = ? "
            "AND json_extract(payload_json, '$.reporter_id') = ? "
            "AND json_extract(payload_json, '$.time') >= ? "
            "AND json_extract(payload_json, '$.time') < ? ORDER BY id",
            (SIGMONDB, ARRIVAL_TABLE, producer, reporter_id, since_iso, until_iso))
        return [json.loads(p) for (p,) in cur]


def read_jsonl_rows(rows_dir: Path, *, reporter_id: str, since_iso: str,
                    until_iso: str) -> list[dict]:
    """Arrival rows from chirp-recorder's own daily JSONL (<output_dir>/rows/
    YYYY-MM-DD.jsonl), for one reporter, since <= time < until.

    The writer keeps this copy on the station and nothing drains it, unlike
    the sink's queue.  Lines that do not parse are skipped."""
    d = Path(rows_dir)
    if not d.is_dir():
        return []
    out = []
    for f in sorted(d.glob("????-??-??.jsonl")):
        if not (since_iso[:10] <= f.stem <= until_iso[:10]):
            continue
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if (r.get("table") != "arrival" or r.get("reporter_id") != reporter_id
                    or not (since_iso <= str(r.get("time", "")) < until_iso)):
                continue
            r.pop("table", None)
            out.append(r)
    return out
