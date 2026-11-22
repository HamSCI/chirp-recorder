"""Replay codar_grab captures through the detector and the row builder.

A capture directory holds codar_grab.py's `index.txt` (one `saved ...` line
per capture, with every label of x[0]) and the `.npy` files it names.  Each
capture runs through detect_group for every group, and rows_for_group turns
the result into sigmondb arrival rows, as the daemon does.

The label, two ways, and every row says which (engineering["label_source"]):

  relabel  minute rows given: relabel.py's arithmetic.  The anchored minute
           nearest the capture pairs an input index with UTC on T6's ruler,
           so utc(x[0]) = anchor_utc + (in_idx0 - anchor_idx) / in_rate.  A
           capture whose nearest anchor lands over 1 s from its own label
           belongs to another radiod run, and one whose neighbouring anchors
           disagree by over 1 ms saw a step; both are skipped, as relabel.py
           skips them.  Rows carry the T6 block.
  index    no minute rows: the capture's own `t0`, which is radiod's
           rtp_to_utc plus hf-timestd's authority offset at capture time.
           It never earns the T6 block: rows carry the null block,
           delay_basis "index", and the line's tier and sigma_ns as
           engineering (label_tier, label_sigma_ns).  Those describe the
           authority, not the label, which scattered 200-420 ms against
           radiod's counter on the golden night.

  counter  label="counter": radiod's own input counter.  utc(x[0]) =
           in_idx0 / in_rate + the run's input epoch (the median of the
           run's lines, GPS seconds) + ka9q's GPS-to-UTC step.  Captures of
           one run then stand exactly as far apart as the counter says; the
           whole run shares one unknown offset (radiod's GPS_TIME lateness).
           Rows carry the null block and delay_basis "counter".  Holding a
           delay across a night needs only this consistency, not UTC.

No host clock enters.  Instants stay integer nanoseconds.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from hamsci_dsp.sigmondb_rows import provenance

from chirp_recorder import __version__
from chirp_recorder.core.detect import detect_group
from chirp_recorder.core.rows import rows_for_group
from chirp_recorder.core.timing import Label
from ka9q.rtp_recorder import GPS_LEAP_SECONDS, GPS_UTC_OFFSET

log = logging.getLogger(__name__)

INDEX_LINE = re.compile(
    r"saved (\S+) (\d+) samples fs=(\d+) f=(\d+) t0=(\S+) .*labels=(\{.*\})")
OTHER_RUN_S = 1.0                  # relabel.py: past this the anchor is from another run
MAX_ANCHOR_DISAGREE_NS = 1_000_000  # relabel.py: neighbouring anchors agree within 1 ms
SAME_RUN_S = 2.0                    # timing.SAME_RUN_TOL_S: one radiod run's epochs agree
LABEL_MODES = ("auto", "counter")


def _lines(d: Path, fs: int, channel_centre_hz: float):
    """(name, t0 ISO, labels) for each index.txt line of this frequency and rate."""
    for line in (d / "index.txt").read_text().splitlines():
        m = INDEX_LINE.search(line)
        if not m or int(m.group(4)) != int(round(channel_centre_hz)) or int(m.group(3)) != fs:
            continue
        yield Path(m.group(1)).name, m.group(5), json.loads(m.group(6))


def counter_labels(capture_dir: Path, *, fs: int, channel_centre_hz: float) -> dict:
    """{capture: utc_ns of x[0]} from radiod's input counter, one epoch per run.

    Lines whose in_epoch sits more than SAME_RUN_S from the run's median, or
    that carry no counter map, get no label."""
    rows = [(n, lab) for n, _, lab in _lines(Path(capture_dir), fs, channel_centre_hz)
            if lab.get("in_idx0") is not None and lab.get("in_rate") and lab.get("in_epoch")]
    if not rows:
        return {}
    epochs = sorted(float(lab["in_epoch"]) for _, lab in rows)
    mid = float(np.median(epochs))
    run = [e for e in epochs if abs(e - mid) <= SAME_RUN_S]
    epoch_ns = round(float(np.median(run)) * 1e9)
    gps_to_utc = (GPS_UTC_OFFSET - GPS_LEAP_SECONDS) * 1_000_000_000
    out = {}
    for name, lab in rows:
        if abs(float(lab["in_epoch"]) - mid) > SAME_RUN_S:
            log.warning("%s: another radiod run (epoch %.3f s); no counter label",
                        name, float(lab["in_epoch"]))
            continue
        idx, rate = int(round(float(lab["in_idx0"]))), int(lab["in_rate"])
        out[name] = idx * 1_000_000_000 // rate + epoch_ns + gps_to_utc
    return out


def relabel_rows(rows: list[dict], labels: dict, *, source: str) -> list[dict]:
    """Copies of replay rows moved onto new labels ({capture: utc_ns}): each
    delay re-folds by the label's change.  Rows of unlabelled captures drop."""
    from hamsci_dsp.sigmondb_rows import provenance as _prov
    from chirp_recorder.core.compare import period_of
    out = []
    for r in rows:
        eng = dict(r["engineering"])
        new = labels.get(eng.get("capture"))
        if new is None:
            continue
        p_ns = round(period_of(r.get("group_key", "")) * 1e9)
        shift = int(new) - int(eng["label_utc_ns"])
        eng.update(label_source=source, label_utc_ns=str(int(new)), delay_basis=source,
                   t_level="")
        out.append({**r, **_prov(None), "delay_ns": (int(r["delay_ns"]) + shift) % p_ns,
                    "engineering": eng})
    return out


def _iso_to_ns(iso: str) -> int:
    """Integer UTC nanoseconds of an ISO time (microsecond text)."""
    t = datetime.fromisoformat(iso)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    whole = int(t.replace(microsecond=0).timestamp())
    return whole * 1_000_000_000 + t.microsecond * 1000


def load_anchors(minutes: list[Path]) -> list[tuple[int, float]]:
    """(anchor_utc_ns, anchor_input_index) from hf-timestd minute rows, sorted."""
    out = []
    for fn in minutes or []:
        for line in Path(fn).read_text().splitlines():
            if not line.strip():
                continue
            s = (json.loads(line).get("t3_shadow") or {})
            if s.get("t6_anchor_input_index") and s.get("t6_anchor_utc_ns"):
                out.append((int(s["t6_anchor_utc_ns"]), float(s["t6_anchor_input_index"])))
    return sorted(out)


def _relabel(t0_ns: int, in_idx0: float, rate: float, anchors, name: str) -> Optional[int]:
    """relabel.py's T6 label of x[0], or None when relabel.py would skip it."""
    au = np.array([a[0] for a in anchors], dtype=np.int64)
    k = int(np.argmin(np.abs(au - t0_ns)))

    def at(j: int) -> int:
        return int(anchors[j][0]) + int(round((in_idx0 - anchors[j][1]) * 1e9 / rate))

    t6 = at(k)
    if abs(t0_ns - t6) > OTHER_RUN_S * 1e9:
        log.warning("%s skipped: nearest T6 anchor is from another radiod run "
                    "(label - T6 = %.0f s)", name, (t0_ns - t6) / 1e9)
        return None
    if len(anchors) > 1:
        j = k + 1 if (au[k] <= t0_ns and k + 1 < len(au)) or k == 0 else k - 1
        t6j = at(j)
        if abs(t6 - t6j) > MAX_ANCHOR_DISAGREE_NS and abs(t0_ns - t6j) <= OTHER_RUN_S * 1e9:
            log.warning("%s skipped: neighbouring T6 anchors disagree by %.3f ms",
                        name, (t6 - t6j) / 1e6)
            return None
    return t6


def _label(source: str, utc_ns: int, lab: dict, name: str) -> Label:
    tier = lab.get("tier")
    sigma = lab.get("sigma_ns")
    epoch = lab.get("in_epoch")
    epoch_id = f"{float(epoch):.0f}" if epoch is not None else ""
    eng = {"label_source": source, "label_utc_ns": str(utc_ns), "capture": name,
           "label_tier": tier or "", "label_sigma_ns": "" if sigma is None else str(int(sigma))}
    if source == "relabel":
        return Label(utc_ns=utc_ns, delay_basis="t6", t_level="T6",
                     provenance=provenance("T6", epoch_id),
                     engineering={"delay_basis": "t6", "t_level": "T6", **eng})
    if source == "counter":
        return Label(utc_ns=utc_ns, delay_basis="counter", t_level=None,
                     provenance=provenance(None),
                     engineering={"delay_basis": "counter", "t_level": "", **eng})
    # An index label never earns the T6 block, whatever tier its line names.
    # The line's tier and sigma describe hf-timestd's authority, not this
    # label: on the golden night the labels scattered 200-420 ms against
    # radiod's input counter while claiming 1-5 us.
    return Label(utc_ns=utc_ns, delay_basis="index", t_level=None,
                 provenance=provenance(None),
                 engineering={"delay_basis": "index", "t_level": "", **eng})


def replay(capture_dir: Path, minutes: list[Path], groups: list, *, fs: int,
           channel_centre_hz: float, reporter_id: Optional[str] = None,
           radiod: str = "replay", label: str = "auto") -> list[dict]:
    """Arrival rows for every capture in capture_dir (see the module text).

    label "auto": relabel by the minute rows when given, else the index
    label; "counter": radiod's input counter.  reporter_id defaults to the
    directory's name.  Lines for another frequency or sample rate, and files
    that are absent, are left out."""
    if label not in LABEL_MODES:
        raise ValueError(f"label must be one of {LABEL_MODES}, not {label!r}")
    d = Path(capture_dir)
    rid = reporter_id if reporter_id is not None else d.name
    anchors = load_anchors(minutes) if label == "auto" else []
    counter = counter_labels(d, fs=fs, channel_centre_hz=channel_centre_hz) \
        if label == "counter" else {}
    detector_version = f"chirp-recorder/{__version__}"
    out: list[dict] = []
    for line in (d / "index.txt").read_text().splitlines():
        m = INDEX_LINE.search(line)
        if not m or int(m.group(4)) != int(round(channel_centre_hz)) or int(m.group(3)) != fs:
            continue
        name = Path(m.group(1)).name
        path = d / name
        lab = json.loads(m.group(6))
        if not path.exists():
            log.warning("%s: listed in index.txt but absent", name)
            continue
        t0_ns = _iso_to_ns(m.group(5))
        if label == "counter":
            if name not in counter:
                continue
            label_ = _label("counter", counter[name], lab, name)
        elif anchors:
            if lab.get("in_idx0") is None or not lab.get("in_rate"):
                log.warning("%s skipped: no input index to relabel by", name)
                continue
            utc = _relabel(t0_ns, float(lab["in_idx0"]), float(lab["in_rate"]), anchors, name)
            if utc is None:
                continue
            label_ = _label("relabel", utc, lab, name)
        else:
            label_ = _label("index", t0_ns, lab, name)
        try:
            x = np.load(path)
        except ValueError as exc:
            log.warning("%s: unreadable (%s)", name, exc)
            continue
        for g in groups:
            result = detect_group(x, fs=fs, channel_centre_hz=channel_centre_hz, group=g)
            _, rows = rows_for_group(
                result, label=label_, reporter_id=rid, radiod=radiod, channel=name,
                duration_s=len(x) / fs, detector_version=detector_version,
                input_epoch_gps_s=float(lab.get("in_epoch") or 0.0),
                interval_s=len(x) / fs, position_source="replay")
            out.extend(rows)
    return out
