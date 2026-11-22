"""The golden check, on Task 17's night of 4.463 MHz captures at K3LR and W2NAF.

1. replay() against the reference: tasks/codar-mf/clean.py on the same
   captures, labelled the same way.  Each arrival replay() keeps must sit
   within 2 us of a CLEAN "A" component.
2. The published delay stays inside the physics bound: |delay_ns -
   tau_uncorrected| <= |f/k| + 1 us, f the row's own Doppler (none: 0).
3. The seed's groups, the production path, reproduce the reference too.
4. The pair: the clusters each station hears by ground wave must hold their
   delay through the night within 2 us, as stage2-groundwave found on 11-07.

The captures live outside the repo (tests/golden/README.md).  Absent
captures fail these tests; they never skip.

Labels.  The hf-timestd minute rows for this night were pruned before the
pull, so relabel.py cannot run.  Check 1 labels each capture from its own
index.txt line (radiod rtp_to_utc plus hf-timestd's authority) on both
sides, so the label cancels out.  Those labels claim no T6: they scatter
200-420 ms against radiod's input counter.  Check 4 places each capture by
that counter instead (replay.counter_labels): one unknown offset per run,
exact spacing between captures, which is all a delay hold needs.

What check 1 compares.  clean.py measures each arrival's delay at the CLEAN
peak, with no range-Doppler correction.  replay() keeps that delay as
engineering["tau_uncorrected_ns"]; its delay_ns adds detect.py's Doppler
correction, which check 2 bounds.

Gate.  clean.py fixes the gate period at 1945.6 us; check 1 gives replay()
that waveform.  The seed now carries the same gate for the Mid-Atlantic
4.463 MHz group, so check 3's production path needs no gate fit.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
from datetime import datetime
from pathlib import Path

import pytest
from hamsci_dsp.chirp.waveform import Waveform

from chirp_recorder.core.channels import Group, groups_for
from chirp_recorder.core.compare import clusters
from chirp_recorder.core.replay import counter_labels, relabel_rows, replay

GOLDEN = Path(os.environ.get("CHIRP_GOLDEN_DIR", "/home/mjh/hamsci/tasks/chirp-registry/golden"))
CLEAN_PY = Path(os.environ.get("CHIRP_CLEAN_PY", "/home/mjh/hamsci/tasks/codar-mf/clean.py"))
STATIONS = ("K3LR", "W2NAF")
FS = 64_000
CENTRE_HZ = 4.463e6
PERIOD_NS = 10**9
TOL_NS = 2_000            # the golden tolerance, both checks
README = "tests/golden/README.md"
_PAT = re.compile(r"saved (\S+) (\d+) samples fs=(\d+) f=(\d+) t0=(\S+) .*labels=(\{.*\})")


def _need(path: Path, what: str) -> None:
    if not path.exists():
        pytest.fail(f"{what} missing: {path}.  The golden captures live outside the repo; "
                    f"{README} says where and how to set CHIRP_GOLDEN_DIR.", pytrace=False)


def _station_dir(st: str) -> Path:
    d = GOLDEN / st
    _need(d / "index.txt", f"{st} golden index")
    if not list(d.glob("*.npy")):
        pytest.fail(f"no {st} captures (*.npy) in {d}; see {README}", pytrace=False)
    return d


def _clean():
    _need(CLEAN_PY, "reference clean.py")
    spec = importlib.util.spec_from_file_location("golden_reference_clean", CLEAN_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _t0_ns(iso: str) -> int:
    t = datetime.fromisoformat(iso)
    return (int(t.replace(microsecond=0).timestamp()) * 10**9) + t.microsecond * 1000


def _captures(d: Path):
    """(name, t0_ns, labels) per 4.463 MHz index line whose file exists."""
    out = []
    for line in (d / "index.txt").read_text().splitlines():
        m = _PAT.search(line)
        if m and int(m.group(4)) == int(CENTRE_HZ) and (d / Path(m.group(1)).name).exists():
            out.append((Path(m.group(1)).name, _t0_ns(m.group(5)), json.loads(m.group(6))))
    return out


def _reference_groups(clean) -> list[Group]:
    """The waveform clean.py synthesises: its rate, 1 s period, its gate."""
    return [Group(Waveform(CENTRE_HZ, clean.K, clean.P, None, clean.TG, clean.DUTY), ("ref",))]


def _seed_groups() -> list[Group]:
    from chirp_recorder.core.catalogue import load_seed
    txs = [t for t in load_seed().values()
           if t.centre_hz == CENTRE_HZ and t.rate_hz_per_s == -25_733.913]
    return list(groups_for(txs))


_cache: dict = {}


def _replayed(st: str, which: str = "reference"):
    key = (st, which)
    if key not in _cache:
        d = _station_dir(st)
        groups = _reference_groups(_clean()) if which == "reference" else _seed_groups()
        _cache[key] = replay(d, sorted(d.glob("minutes-*.jsonl")), groups, fs=FS,
                             channel_centre_hz=CENTRE_HZ, reporter_id=st)
    return _cache[key]


def _reference(st: str):
    """{capture: (label_ns, [tau_ns of each CLEAN "A" component])}."""
    key = (st, "clean")
    if key not in _cache:
        import numpy as np
        clean = _clean()
        out = {}
        for name, t0_ns, _ in _captures(_station_dir(st)):
            x = np.load(GOLDEN / st / name)
            comps, _ = clean.arrivals(x, floor_db=10, max_comp=12)
            taus = [round(tau * 1e9) for tau, _db, tag in clean.ladder_report(comps) if not tag]
            out[name] = (t0_ns, taus)
        _cache[key] = out
    return _cache[key]


def _circ(a: int, b: int) -> int:
    return (a - b + PERIOD_NS // 2) % PERIOD_NS - PERIOD_NS // 2


def _match(st: str, which: str = "reference"):
    """(matched [(capture, diff_unc_ns, diff_corrected_ns)], unmatched [(capture, tau)])."""
    ref = _reference(st)
    matched, unmatched = [], []
    for r in _replayed(st, which):
        eng = r["engineering"]
        cap = eng["capture"]
        t0_ns, taus = ref[cap]
        label = int(eng["label_utc_ns"])
        if eng["label_source"] == "index":
            assert label == t0_ns, f"{cap}: replay's index label differs from the reference's"
        unc = (label + int(eng["tau_uncorrected_ns"])) % PERIOD_NS
        ref_phases = [(t0_ns + t) % PERIOD_NS for t in taus]
        if not ref_phases:
            unmatched.append((cap, unc))
            continue
        best = min(ref_phases, key=lambda p: abs(_circ(unc, p)))
        if abs(_circ(unc, best)) > TOL_NS:
            unmatched.append((cap, unc))
            continue
        matched.append((cap, _circ(unc, best), _circ(int(r["delay_ns"]), best)))
    return matched, unmatched


@pytest.mark.parametrize("st", STATIONS)
def test_replay_matches_the_reference_within_2_us(st):
    matched, unmatched = _match(st)
    n_caps = len(_captures(_station_dir(st)))
    worst = max(abs(m[1]) for m in matched) if matched else None
    worst_corr = max(abs(m[2]) for m in matched) if matched else None
    print(f"\n{st}: {n_caps} captures, {len(matched)} arrivals matched, "
          f"{len(unmatched)} unmatched; worst |replay - clean.py| {worst} ns "
          f"(uncorrected delay); corrected delay_ns differs by up to {worst_corr} ns")
    assert not unmatched, f"{st}: replay arrivals with no CLEAN partner within 2 us: {unmatched}"
    assert len(matched) >= n_caps, f"{st}: only {len(matched)} arrivals over {n_caps} captures"
    assert worst <= TOL_NS


@pytest.mark.parametrize("st", STATIONS)
def test_published_delay_stays_inside_the_physics_bound(st):
    """|delay_ns - tau_uncorrected| <= |f/k| + 1 us: a Doppler correction never
    moves a delay further than the coupling it removes (Task 18 review, I1;
    before the fix, W2NAF 09:00Z moved 44.7 us at f = -0.223 Hz)."""
    worst, n_low, n_clip, n = -float("inf"), 0, 0, 0
    for r in _replayed(st):
        eng = r["engineering"]
        k = float(r["rate_hz_per_s"])
        f = r["doppler_hz"]
        bound = abs((f or 0.0) / k) * 1e9 + 1_000
        unc = (int(eng["label_utc_ns"]) + int(eng["tau_uncorrected_ns"])) % PERIOD_NS
        moved = abs(_circ(int(r["delay_ns"]), unc))
        worst = max(worst, moved - bound)
        n += 1
        n_low += eng["doppler_low_coherence"] == "1"
        n_clip += eng["doppler_clipped"] == "1"
        assert moved <= bound, (st, eng["capture"], moved, bound, f)
    print(f"\n{st}: {n} arrivals, {n_low} with Doppler gated off (low coherence), "
          f"{n_clip} clipped; tightest margin {-worst:.0f} ns")


def test_seed_groups_reproduce_the_reference():
    """The production path: the seed's 4.463 MHz group.  The seed gives the
    group's gate (1.9456 ms), so detect.py fits none; before it did, the fit
    locked on two arrivals' spacing on 5 of K3LR's 16 captures."""
    groups = _seed_groups()
    assert {g.waveform.gate_period_s for g in groups} == {1.9456e-3}
    matched, unmatched = _match("K3LR", "seed")
    print(f"\nK3LR, seed groups: {len(matched)} matched, {len(unmatched)} unmatched")
    assert not unmatched, f"K3LR, seed groups: {len(unmatched)} arrivals with no CLEAN partner"
    assert len(matched) >= len(_captures(_station_dir("K3LR")))


GROUND_WAVE_MAX_DOPPLER_HZ = 0.02     # about one codar-sounder Doppler bin


def _counter_placed(st: str):
    d = _station_dir(st)
    labels = counter_labels(d, fs=FS, channel_centre_hz=CENTRE_HZ)
    return relabel_rows(_replayed(st), labels, source="counter")


@pytest.mark.xfail(strict=True, reason=(
    "This night holds no ground wave to test.  Placed by radiod's input counter, K3LR's "
    "16 captures form 10 clusters of >= 6 captures and W2NAF's 19 form 7; every one "
    "spreads 116-614 us over the night.  None at K3LR has |median Doppler| <= 0.02 Hz; "
    "W2NAF's two that do (30.44 ms, 68.96 ms) spread 296 and 607 us: merged night sky "
    "wave.  stage2-groundwave saw the day tracks vanish 04-10Z; this window is mostly "
    "night.  Needs a night with daytime captures (Task 18 report, round 1)."))
def test_ground_wave_clusters_hold_their_delay_through_the_night():
    """Each station's ground-wave clusters (median |Doppler| <= 0.02 Hz, seen
    in at least six captures) hold their delay within 2 us over the night.

    Captures stand where radiod's input counter puts them; the index labels'
    tier and sigma describe hf-timestd's authority, not those labels, so no
    capture is dropped on them."""
    found = {}
    for st in STATIONS:
        rows = _counter_placed(st)
        cs = clusters(rows, producer="chirp-recorder", t6_only=False)
        ground = [c for c in cs if c.median_doppler_hz is not None
                  and abs(c.median_doppler_hz) <= GROUND_WAVE_MAX_DOPPLER_HZ]
        found[st] = ground
        print(f"\n{st}: {len(rows)} counter-placed rows, {len(cs)} clusters of >= 6 captures: "
              + "; ".join(f"{c.median_delay_ns / 1e6:.4f} ms n={c.n} spread {c.spread_ns} ns "
                          f"dop {c.median_doppler_hz}" for c in cs))
    for st, ground in found.items():
        assert ground, f"{st}: no ground-wave cluster to test"
        for c in ground:
            assert c.spread_ns <= TOL_NS, (
                f"{st}: cluster {c.median_delay_ns} ns spreads {c.spread_ns} ns")
