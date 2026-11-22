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

Half a sweep.  codar-sounder's replica starts its sweep at the channel
centre, a half sweep from where the transmitted chirp starts, so its delay
reads tau + sweep/2 (from_codar_convention takes that back off first).

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
import logging
import math
import os
import sqlite3
import stat
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

PAIR_WINDOW_NS = 120_000          # clusters closer than this pair up (the cluster gap)
MATCH_MODES = ("bin", "exact")
# |1 - g| for a coarse row (tau - g f/k) against codar-sounder's tau - f/k:
# |g| <= 1, so at most 2 (real captures showed g down to -0.7).
COARSE_COUPLING_FACTOR = 2.0
# The first codar-sounder release whose delay_ns starts at the sweep start
# (the half-sweep fix), as its detector_version reads: "codar-sounder/X.Y.Z".
# None: no release carries the fix yet, so every codar-sounder row gets
# corrected.  Once codar-sounder ships the fix and bumps its version, set
# this to that version string; rows of that version or later then pass
# through unchanged, and earlier rows keep their correction.
CODAR_HALF_SWEEP_FIXED_FROM: Optional[str] = None
# The JSONL readers run as root (sigmond's check) on files another account
# can write: no symlinks, regular files only, at most this many bytes each.
MAX_JSONL_BYTES = 64 * 1024 * 1024
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


def sweep_of(group_key: str) -> float:
    """The sweep duration a group key names ("sweepXs"; the period if none)."""
    for tok in str(group_key).split()[2:]:
        if tok.startswith("sweep") and tok.endswith("s"):
            try:
                return float(tok[5:-1])
            except ValueError:
                pass
    return period_of(group_key)


def _version(text: str) -> Optional[tuple]:
    """(major, minor, patch) of "codar-sounder/X.Y.Z", else None."""
    name, _, ver = str(text or "").partition("/")
    if name != "codar-sounder":
        return None
    try:
        return tuple(int(p) for p in ver.split(".")[:3])
    except ValueError:
        return None


def _half_sweep_fixed(row: dict) -> bool:
    if CODAR_HALF_SWEEP_FIXED_FROM is None:
        return False
    have, fixed = _version(row.get("detector_version")), _version(CODAR_HALF_SWEEP_FIXED_FROM)
    return have is not None and fixed is not None and have >= fixed


def from_codar_convention(rows: list[dict]) -> list[dict]:
    """codar-sounder rows moved half a sweep back, onto chirp-recorder's sweep start.

    codar-sounder's replica sweeps from its channel centre, f = k t from t = 0
    (codar-sounder dechirp.make_replica); the transmitted chirp sweeps through
    the centre at mid-sweep (hamsci-dsp's waveform, codar-mf/clean.py; the
    golden captures' spectra sit +-B/2 around 4.463 MHz).  So codar-sounder's
    beat reads tau + S/2, S the sweep duration, and its delay_ns sits half a
    sweep (0.5 s at 4.463 MHz) from the schema's sweep start (Task 18 round
    2, measured on codar-sounder's own rows_for_cpi; review round 2, section 2).

    Assumption: codar-sounder's channel centre equals the group's centre.  It
    tunes one channel to its first transmitter's centre; a transmitter at
    another centre in that channel shifts by a further (f_tx - f_channel)/k,
    which this function does not correct.  At 4.463 MHz the two agree.

    Rows whose detector_version is CODAR_HALF_SWEEP_FIXED_FROM or later pass
    through unchanged (None: no release carries the fix, all get corrected)."""
    out = []
    for r in rows:
        if _half_sweep_fixed(r):
            out.append(dict(r))
            continue
        gk = r.get("group_key", "")
        p_ns = round(period_of(gk) * 1e9)
        half = round(sweep_of(gk) * 1e9 / 2)
        out.append(dict(r, delay_ns=(int(r["delay_ns"]) - half) % p_ns))
    return out


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
    coarse: list = field(default_factory=list)   # [(chirp coarse Cluster, codar Cluster, diff_ns, limit_ns)]
    tiers: tuple = ("fine", "coarse")
    mean_offset: dict = field(default_factory=dict)   # {"gating", "n", "mean_ns"}

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
            "matched_coarse": [[a.median_delay_ns, b.median_delay_ns, d, lim, a.group_key]
                               for a, b, d, lim in self.coarse],
            "tiers": list(self.tiers),
            "mean_offset": {k: _num(v) for k, v in self.mean_offset.items()},
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


def is_fine(row: dict) -> bool:
    """True for a row whose published delay detect.py corrected: it carries a
    Doppler, and no flag says the correction was skipped (ambiguous), gated
    off (low coherence) or clipped.  Only such a row converts by C1."""
    if not _is_set(row.get("doppler_hz")):
        return False
    eng = row.get("engineering") or {}
    return not any(str(eng.get(k, "")) in ("1", "True", "true")
                   for k in ("doppler_ambiguous", "doppler_low_coherence", "doppler_clipped"))


def compare(chirp_rows, codar_rows, *, tol_ns: int = 5_000, min_fraction: float = 0.9,
            min_matched: int = 3, doppler_tol_hz: float = 0.02,
            rung_s: dict[str, float], match: str = "bin", coarse: bool = True,
            mean_offset: bool = False, mean_offset_ns: int = 5_000,
            mean_offset_min_n: int = 20) -> Verdict:
    """Decision 3's verdict, in two tiers.

    Fine tier: chirp-recorder rows with a usable Doppler (is_fine), converted
    to codar-sounder's tau - f/k.  It alone feeds `matched` and `doppler`.
    Coarse tier (coarse=True): every other T6 row, on its own uncorrected
    delay.  Such a row holds tau - g f/k with |g| <= 1 (Task 18 review,
    R1-I4, M-a of round 2), so against codar-sounder's tau - f/k it differs by (1 - g) f/k: a
    coarse cluster covers a codar cluster when |d| <= limit + 2 |f/k|, f the
    codar cluster's median Doppler.  `coverage` counts both tiers;
    `matched_coarse` reports the coarse pairs and gates nothing.

    `mean_offset` reports |mean(chirp - codar)| over the fine pairs inside
    their limit; it gates (and needs at least mean_offset_min_n of them, else
    "insufficient") only when mean_offset=True.  Provisional: Michael decides.
    """
    if match not in MATCH_MODES:
        raise ValueError(f"match must be one of {MATCH_MODES}, not {match!r}")
    rows_in = list(chirp_rows)
    chirp = clusters(to_codar_convention([r for r in rows_in if is_fine(r)]),
                     producer="chirp-recorder")
    chirp_c = clusters([r for r in rows_in if not is_fine(r)], producer="chirp-recorder") \
        if coarse else []
    codar = clusters(from_codar_convention(list(codar_rows)), producer="codar-sounder")

    def bin_ns(c: Cluster) -> float:
        return codar_bin_ns(c.rate_hz_per_s, c.period_s) if c.rate_hz_per_s else math.nan

    def limit_ns(c: Cluster) -> float:
        if match == "exact":
            return float(tol_ns)
        b = bin_ns(c)
        return b / 2 + tol_ns if not math.isnan(b) else float(tol_ns)

    def coarse_limit_ns(c: Cluster) -> float:
        f = c.median_doppler_hz or 0.0
        k = c.rate_hz_per_s
        return limit_ns(c) + (COARSE_COUPLING_FACTOR * abs(f / k) * 1e9 if k else 0.0)

    pairs, paired_codar, paired_chirp = _pair(
        chirp, codar, lambda c: max(PAIR_WINDOW_NS, limit_ns(c)))
    pairs.sort(key=lambda p: (p[1].group_key, p[1].median_delay_ns))
    rest = [i for i in range(len(codar)) if i not in paired_codar]
    cpairs, cpaired_rest, paired_chirp_c = _pair(
        chirp_c, [codar[i] for i in rest], lambda c: max(PAIR_WINDOW_NS, coarse_limit_ns(c)))
    cpairs = sorted(((a, b, d, coarse_limit_ns(b)) for a, b, d in cpairs),
                    key=lambda p: (p[1].group_key, p[1].median_delay_ns))
    paired_any = set(paired_codar) | {rest[i] for i in cpaired_rest}

    # Strongest first; at equal strength a paired cluster comes first, then the
    # more often seen, then the earlier.  Only a cluster earlier in this order
    # explains a later one, so two equal clusters never explain each other.
    order = sorted(range(len(codar)), key=lambda i: (
        -_strength(codar[i]), i not in paired_any, -codar[i].n, codar[i].median_delay_ns))
    explained, unmatched = [], []
    for k, i in enumerate(order):
        if i in paired_any:
            continue
        why = _explain(codar[i], [codar[j] for j in order[:k]], rung_s)
        if why:
            explained.append((codar[i], why))
        else:
            unmatched.append(codar[i])
    unmatched.sort(key=lambda c: (c.group_key, c.median_delay_ns))
    chirp_only = sorted([c for j, c in enumerate(chirp) if j not in paired_chirp]
                        + [c for j, c in enumerate(chirp_c) if j not in paired_chirp_c],
                        key=lambda c: (c.group_key, c.median_delay_ns))

    within = [p for p in pairs if abs(p[2]) <= limit_ns(p[1])]
    cwithin = [p for p in cpairs if abs(p[2]) <= p[3]]
    worst = float(max((abs(p[2]) for p in pairs), default=math.nan))
    # One group gives one limit; across groups the row shows the largest and
    # every pair still meets its own.
    lim = max((limit_ns(p[1]) for p in pairs), default=limit_ns(codar[0]) if codar
              else float(tol_ns))
    rows = [VerdictRow("matched", worst, float(lim), True,
                       len(pairs) >= min_matched and len(within) == len(pairs))]
    unexplained = len(codar) - len(explained)
    cov = (len(within) + len(cwithin)) / unexplained if unexplained else math.nan
    rows.append(VerdictRow("coverage", cov, min_fraction, True,
                           bool(unexplained) and cov >= min_fraction))
    rows.append(VerdictRow(
        "matched_coarse", float(max((abs(p[2]) for p in cpairs), default=math.nan)),
        float(max((p[3] for p in cpairs), default=math.nan)), False,
        len(cwithin) == len(cpairs)))
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
    ds = [p[2] for p in within]
    mean = float(statistics.fmean(ds)) if ds else math.nan
    rows.append(VerdictRow("mean_offset", abs(mean), float(mean_offset_ns), bool(mean_offset),
                           len(ds) >= mean_offset_min_n and abs(mean) <= mean_offset_ns))

    if (len(chirp) < min_matched or len(codar) < min_matched or not dop
            or (mean_offset and len(ds) < mean_offset_min_n)):
        status = "insufficient"   # too few fine clusters, no Doppler, or too few for the mean
    else:
        status = "pass" if all(r.passed for r in rows if r.gating) else "fail"
    bins = {c.group_key: bin_ns(c) for c in codar if c.rate_hz_per_s}
    return Verdict(matched=pairs, codar_unmatched=unmatched, explained=explained,
                   chirp_only=chirp_only, rows=rows, status=status, match=match,
                   codar_bin_ns=bins, coarse=cpairs,
                   tiers=("fine", "coarse") if coarse else ("fine",),
                   mean_offset={"gating": bool(mean_offset), "n": len(ds), "mean_ns": mean})


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
                    until_iso: str, require_table: bool = True,
                    pattern: str = "????-??-??.jsonl", day_of=lambda stem: stem) -> list[dict]:
    """Arrival rows from chirp-recorder's own daily JSONL (<output_dir>/rows/
    YYYY-MM-DD.jsonl), for one reporter, since <= time < until.

    The writer keeps this copy on the station and nothing drains it, unlike
    the sink's queue.  Lines that do not parse are skipped."""
    d = Path(rows_dir)
    if not d.is_dir():
        return []
    out = []
    for f in sorted(d.glob(pattern)):
        if not (since_iso[:10] <= day_of(f.stem) <= until_iso[:10]):
            continue
        text = _read_capped(f)
        if text is None:
            continue
        for line in text.splitlines():
            try:
                r = json.loads(line)
            except (ValueError, RecursionError):
                continue
            if not isinstance(r, dict):
                continue
            table = r.get("table", None if not require_table else "")
            if (table not in ("arrival", None) or (require_table and table != "arrival")
                    or r.get("reporter_id") != reporter_id
                    or not (since_iso <= str(r.get("time", "")) < until_iso)):
                continue
            r.pop("table", None)
            out.append(r)
    return out


def _read_capped(path: Path) -> Optional[str]:
    """A file's text, or None (logged) when it is a symlink, not a regular
    file, or larger than MAX_JSONL_BYTES.  Opens with O_NOFOLLOW and
    O_NONBLOCK (a FIFO never blocks), then checks the open file by fstat."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        log.warning("%s skipped: cannot open safely (%s)", path.name, exc)
        return None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            log.warning("%s skipped: not a regular file", path.name)
            return None
        if st.st_size > MAX_JSONL_BYTES:
            log.warning("%s skipped: %d bytes, over the %d-byte cap", path.name,
                        st.st_size, MAX_JSONL_BYTES)
            return None
        with os.fdopen(fd, "rb", closefd=True) as fh:
            fd = -1
            data = fh.read(MAX_JSONL_BYTES + 1)
        if len(data) > MAX_JSONL_BYTES:          # grew while we read
            log.warning("%s skipped: over the %d-byte cap", path.name, MAX_JSONL_BYTES)
            return None
        return data.decode("utf-8", errors="replace")
    except OSError as exc:
        log.warning("%s skipped: %s", path.name, exc)
        return None
    finally:
        if fd >= 0:
            os.close(fd)


def read_codar_jsonl_rows(rows_dir: Path, *, reporter_id: str, since_iso: str,
                          until_iso: str) -> list[dict]:
    """codar-sounder's arrival rows from its daily JSONL, for one reporter,
    since <= time < until.

    Format (codar-sounder bf3f360, Emitter._keep_arrivals): files
    <rows_dir>/arrivals-YYYYMMDD.jsonl, by default /var/lib/codar-sounder/
    arrivals/, named for the UTC day of each row's time and kept 30 days.
    One JSON object per line: the very row dict codar-sounder hands the sink
    writer for sigmondb.arrival (hamsci_dsp.sigmondb_rows.ARRIVAL_COLUMNS:
    time, reporter_id, group_key, rate_hz_per_s, delay_ns, doppler_hz, s4,
    sigma_phi_rad, u_epoch_ns, ...), with no wrapper keys.  Its delay_ns keeps
    codar-sounder's convention (tau - f/k + sweep/2); compare converts.  Lines
    that do not parse are skipped; other files are ignored.  Symlinks,
    non-regular files and files over MAX_JSONL_BYTES are skipped and logged
    (this runs as root on files the codar-sounder account writes)."""
    return read_jsonl_rows(rows_dir, reporter_id=reporter_id, since_iso=since_iso,
                           until_iso=until_iso, require_table=False,
                           pattern="arrivals-????????.jsonl",
                           day_of=lambda stem: f"{stem[9:13]}-{stem[13:15]}-{stem[15:17]}")
