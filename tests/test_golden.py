"""The golden check, on Task 17's night of 4.463 MHz captures at K3LR and W2NAF.

1. replay() against the reference: tasks/codar-mf/clean.py on the same
   captures, labelled the same way.  Each arrival replay() keeps must sit
   within 2 us of a CLEAN "A" component.
2. The pair: the T6 clusters each station hears by ground wave must hold
   their delay through the night within 2 us, as stage2-groundwave found on
   11-07.

The captures live outside the repo (tests/golden/README.md).  Absent
captures fail these tests; they never skip.

Labels.  The hf-timestd minute rows for this night were pruned before the
pull, so relabel.py cannot run.  Each capture takes the label on its own
index.txt line (radiod rtp_to_utc plus hf-timestd's authority), on both sides
of check 1, so the label cancels out of that comparison.  replay() still
takes minute rows when a night has them; the golden directory's
minutes-*.jsonl, if any, go to it, and the reference then reads replay's
recorded label.

What check 1 compares.  clean.py measures each arrival's delay at the CLEAN
peak, with no range-Doppler correction.  replay() keeps that delay as
engineering["tau_uncorrected_ns"]; its delay_ns adds detect.py's Doppler
correction.  Check 1 compares the like-for-like quantity, the uncorrected
delay.  It prints the corrected delay's difference too, for the record.

Gate.  clean.py fixes the gate period at 1945.6 us.  Check 1 gives replay()
the same waveform.  test_seed_groups_reproduce_the_reference runs the
production path, the seed's groups with no gate period, where detect.py fits
the gate from the profile.
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
from chirp_recorder.core.replay import replay

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


@pytest.mark.xfail(strict=True, reason=(
    "detect.py fits the gate from the profile when the seed gives none (data/chirp-seed.toml "
    "holds no gate_period_s), and on 5 of K3LR's 16 golden captures (none of W2NAF's 19) "
    "fit_gate locked on the spacing of two arrivals (2.25-5.26 ms) instead of 1.9456 ms; CLEAN then subtracts the "
    "wrong ladder and reports rungs ~19.97 ms from their parent as arrivals.  "
    "Task 18 report, concern 2."))
def test_seed_groups_reproduce_the_reference():
    """The production path: the seed's 4.463 MHz group, gate fitted per capture."""
    matched, unmatched = _match("K3LR", "seed")
    assert not unmatched, f"K3LR, seed groups: {len(unmatched)} arrivals with no CLEAN partner"


GROUND_WAVE_MAX_DOPPLER_HZ = 0.02     # about one codar-sounder Doppler bin


@pytest.mark.xfail(strict=True, reason=(
    "No T6 ground-wave cluster on this night at either station.  The index.txt labels "
    "scatter capture to capture against radiod's own input counter (200 ms at K3LR, 420 ms "
    "at W2NAF), so no T6 cluster of 6 captures forms (largest: 3 captures at K3LR, 1 at W2NAF).  Placed by "
    "the counter instead, the clusters that recur are night sky wave: spread 16-520 us, "
    "median Doppler to 0.15 Hz.  stage2-groundwave saw the day tracks vanish 04-10Z too.  "
    "Task 18 report, concern 1."))
def test_ground_wave_clusters_hold_their_delay_through_the_night():
    """Each station's ground-wave T6 clusters (median |Doppler| <= 0.02 Hz) hold
    their delay within 2 us over the night.

    Rows whose label sigma exceeds the 2 us tolerance stay out: a label that
    uncertain cannot show a 2 us hold (K3LR keeps 6 of its 14 T6 captures,
    W2NAF all 11)."""
    found = {}
    for st in STATIONS:
        rows = [r for r in _replayed(st)
                if int(r["engineering"].get("label_sigma_ns") or 0) <= TOL_NS]
        cs = clusters(rows, producer="chirp-recorder")
        ground = [c for c in cs if c.median_doppler_hz is not None
                  and abs(c.median_doppler_hz) <= GROUND_WAVE_MAX_DOPPLER_HZ]
        found[st] = ground
        print(f"\n{st}: {len(rows)} low-sigma rows, {len(cs)} T6 clusters, "
              f"{len(ground)} ground wave: "
              + ", ".join(f"{c.median_delay_ns / 1e6:.4f} ms spread {c.spread_ns} ns n={c.n}"
                          for c in ground))
    for st, ground in found.items():
        assert ground, f"{st}: no T6 ground-wave cluster to test"
        for c in ground:
            assert c.spread_ns <= TOL_NS, (
                f"{st}: cluster {c.median_delay_ns} ns spreads {c.spread_ns} ns")
