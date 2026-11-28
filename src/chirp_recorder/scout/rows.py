"""The scout's rows for sigmondb.sweep and sigmondb.scout_dwell, and the
screen that decides which streaks become sweep rows.

Columns equal the server's DDL less received_at and ingested_at, which the
server fills.  Rows carry no position.  Values are plain Python so
json.dumps needs no default hook.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

from hamsci_dsp.chirp.scout_screen import (artefact_reason, match_key, persistent,
                                           reject_match)
from hamsci_dsp.chirp.streaks import Streak

from chirp_recorder.scout.plan import Dwell
from chirp_recorder.scout.published import Published

SUBHARMONIC_K = (2, 3, 4)
SUBHARMONIC_TOL = 0.01
# A twin fits every other sweep, so its rate wanders more than a source
# match allows: SIGMA's 5.263 MHz CODAR, 11-28, gave -25747 Hz/s at 1.0005 s
# and -26278 Hz/s (2.1 % off, past match_key's 500 Hz/s) at 1.9986 s.
SUBHARMONIC_RATE_TOL = 0.05
SUBHARMONIC_CENTRE_FRAC = 0.25

DETECTOR_VERSION = "scout-1"
SCHEMA_VERSION = 1


def _iso(t: datetime) -> str:
    if t.tzinfo is None or t.utcoffset() != timedelta(0):
        raise ValueError("start_utc must be timezone-aware UTC")
    t = t.astimezone(timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


def _f(x) -> Optional[float]:
    return None if x is None else float(x)


def _eng(d: dict) -> dict:
    return {str(k): str(v) for k, v in d.items()}


def sweep_row(s: Streak, *, dwell: Dwell, start_utc: datetime, reporter_id: str,
              frame_s: float, matched_key: str, engineering: dict) -> dict:
    if s.period_s is None:
        raise ValueError("a streak without a period cannot make a sweep row: "
                         "sigmondb.sweep.period_s is not nullable")
    return {
        "time": _iso(start_utc), "reporter_id": reporter_id, "view": dwell.view,
        "centre_hz": float(s.centre_hz), "span_hz": float(s.span_hz),
        "rate_hz_per_s": float(s.rate_hz_per_s), "period_s": float(s.period_s),
        "duration_s": _f(s.duration_s), "snr_db": float(s.snr_db),
        "n_sweeps": int(s.n_sweeps), "fit_score": _f(s.fit_score),
        "clipped": bool(s.clipped), "matched_key": matched_key,
        "bin_hz": float(dwell.bin_hz), "frame_s": float(frame_s),
        "detector_version": DETECTOR_VERSION, "schema_version": SCHEMA_VERSION,
        "engineering": _eng(engineering),
    }


def dwell_row(*, dwell: Dwell, start_utc: datetime, reporter_id: str,
              frame_s: float, duration_s: float, n_frames: int,
              noise_db: Optional[float], n_streaks: int, engineering: dict) -> dict:
    return {
        "time": _iso(start_utc), "reporter_id": reporter_id, "view": dwell.view,
        "centre_hz": float(dwell.centre_hz), "span_hz": float(dwell.span_hz),
        "bin_hz": float(dwell.bin_hz), "frame_s": float(frame_s),
        "duration_s": float(duration_s), "n_frames": int(n_frames),
        "noise_db": _f(noise_db), "n_streaks": int(n_streaks),
        "detector_version": DETECTOR_VERSION, "schema_version": SCHEMA_VERSION,
        "engineering": _eng(engineering),
    }


def screen(streaks: Sequence[Streak], published: Published, *,
           nyquist_hz: Optional[float], min_sweeps: int, min_fit: float,
           min_span_hz: float = 10_000.0
           ) -> tuple[list[tuple[Streak, str, dict]], dict[str, int]]:
    """Streaks that pass, each with its matched key and engineering tags, and
    counts of those dropped by reason.  A locally rejected streak passes: the
    server records it as heard while rejected.  ``min_fit`` goes to
    ``persistent``; the runner passes 0.6, since SIGMA's CODAR fits 0.71.
    A streak narrower than ``min_span_hz`` goes as "narrow": at min_fit 0.6
    SIGMA's drifting carriers (1-4 kHz wide) passed as slow sweeps.
    Last, a streak that repeats another's centre and rate at 2, 3 or 4
    times its period (the same source, every other sweep) goes as
    "subharmonic"."""
    kept: list[tuple[Streak, str, dict]] = []
    dropped = {"carrier": 0, "alias": 0, "not_persistent": 0, "subharmonic": 0,
               "narrow": 0}
    for s in streaks:
        why = artefact_reason(s, streaks, nyquist_hz=nyquist_hz)
        if why:
            dropped[why] += 1
        elif s.span_hz < min_span_hz:
            dropped["narrow"] += 1
        elif not persistent(s, min_sweeps=min_sweeps, min_fit=min_fit):
            dropped["not_persistent"] += 1
        else:
            eng: dict = {}
            rej = reject_match(s, published.rejects)
            if rej:
                eng["local_reject"] = rej
            kept.append((s, match_key(s, published.chirps), eng))
    out = [k for k in kept if not _subharmonic(k[0], [o[0] for o in kept])]
    dropped["subharmonic"] = len(kept) - len(out)
    return out, dropped


def _subharmonic(s: Streak, others: Sequence[Streak]) -> bool:
    """True when another streak with a shorter period shares s's sweep: rate
    of the same sign within SUBHARMONIC_RATE_TOL of the larger, centres within
    SUBHARMONIC_CENTRE_FRAC of the wider span; and s's period is k = 2, 3 or 4
    times that streak's, within 1 % of s's period."""
    if s.period_s is None:
        return False
    for o in others:
        if o is s or o.period_s is None or o.period_s >= s.period_s:
            continue
        if (s.rate_hz_per_s > 0) != (o.rate_hz_per_s > 0):
            continue
        big = max(abs(s.rate_hz_per_s), abs(o.rate_hz_per_s))
        if abs(s.rate_hz_per_s - o.rate_hz_per_s) > SUBHARMONIC_RATE_TOL * big:
            continue
        if abs(s.centre_hz - o.centre_hz) > SUBHARMONIC_CENTRE_FRAC * max(s.span_hz, o.span_hz):
            continue
        if any(abs(s.period_s - k * o.period_s) <= SUBHARMONIC_TOL * s.period_s
               for k in SUBHARMONIC_K):
            return True
    return False
