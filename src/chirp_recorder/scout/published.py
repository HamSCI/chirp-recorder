"""The chirps the fleet has published, and this station's local rejects.

load_published asks the public ClickHouse hosts anonymously over HTTPS,
caches the answer in state_dir/published.json, and falls back to that file,
then to nothing, so the scout still runs with the hosts unreachable.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence
from urllib.parse import quote

from hamsci_dsp.chirp.scout_screen import Signature

log = logging.getLogger(__name__)

CACHE_NAME = "published.json"

CHIRPS_SQL = ("SELECT group_key, centre_mhz, rate_khz_s, period_s, span_khz "
              "FROM sigmondb_public.chirps "
              "WHERE state IN ('established','dormant','candidate','probing') "
              "FORMAT JSONEachRow")
REJECTS_SQL = ("SELECT chirp_id, centre_mhz, rate_khz_s, period_s, span_khz "
               "FROM sigmondb_public.chirp_reject_local "
               "WHERE reporter_id = {reporter_id:String} FORMAT JSONEachRow")


@dataclass(frozen=True)
class Published:
    chirps: tuple
    rejects: tuple
    fetched_at: Optional[float]
    source: str     # a host, "cache" or "none"


def _https_get(url: str, timeout: float) -> bytes:
    """GET url anonymously; only https:// URLs."""
    if not url.startswith("https://"):
        raise ValueError("only https:// URLs are fetched")
    with urllib.request.urlopen(url, timeout=timeout) as r:     # noqa: S310
        return r.read()


def _scaled(v, k: float) -> Optional[float]:
    return None if v is None else float(v) * k


def _parse(body: bytes, key_field: str) -> tuple:
    out = []
    for line in body.decode("utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        sig = Signature(
            centre_hz=float(r["centre_mhz"]) * 1e6,
            rate_hz_per_s=float(r["rate_khz_s"]) * 1e3,
            period_s=None if r.get("period_s") is None else float(r["period_s"]),
            span_hz=_scaled(r.get("span_khz"), 1e3),
            key=str(r[key_field]))
        # json.loads takes NaN and Infinity; a row holding one is malformed,
        # and the host counts as failed.
        if not all(math.isfinite(v) for v in (sig.centre_hz, sig.rate_hz_per_s,
                                              sig.period_s, sig.span_hz) if v is not None):
            raise ValueError(f"non-finite number in a published row: {line!r}")
        out.append(sig)
    return tuple(out)


def _sig_dict(s: Signature) -> dict:
    return {"centre_hz": s.centre_hz, "rate_hz_per_s": s.rate_hz_per_s,
            "period_s": s.period_s, "span_hz": s.span_hz, "key": s.key}


def _sig_load(d: dict) -> Signature:
    return Signature(centre_hz=float(d["centre_hz"]),
                     rate_hz_per_s=float(d["rate_hz_per_s"]),
                     period_s=d.get("period_s"), span_hz=d.get("span_hz"),
                     key=str(d["key"]))


def _read_cache(path: Path) -> Optional[Published]:
    try:
        d = json.loads(path.read_text())
        return Published(tuple(_sig_load(x) for x in d["chirps"]),
                         tuple(_sig_load(x) for x in d["rejects"]),
                         float(d["fetched_at"]), "cache")
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError) as e:
        log.warning("published cache %s unusable (%s)", path, type(e).__name__)
        return None


def _write_cache(state_dir: Path, p: Published) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / CACHE_NAME
    tmp = state_dir / (CACHE_NAME + ".tmp")
    tmp.write_text(json.dumps({
        "fetched_at": p.fetched_at, "source": p.source,
        "chirps": [_sig_dict(s) for s in p.chirps],
        "rejects": [_sig_dict(s) for s in p.rejects]}))
    os.replace(tmp, path)


def load_published(state_dir: Path, hosts: Sequence[str], reporter_id: str, *,
                   fetch: Callable[[str, float], bytes] = _https_get,
                   now: Callable[[], float] = time.time,
                   timeout_s: float = 20.0) -> Published:
    """Chirps and local rejects from the first host that answers both
    queries; else the cache; else empty."""
    state_dir = Path(state_dir)
    for host in hosts:
        base = host.rstrip("/")
        try:
            chirps = _parse(fetch(f"{base}/?query={quote(CHIRPS_SQL, safe='')}",
                                  timeout_s), "group_key")
            rejects = _parse(fetch(
                f"{base}/?query={quote(REJECTS_SQL, safe='')}"
                f"&param_reporter_id={quote(reporter_id, safe='')}",
                timeout_s), "chirp_id")
        except Exception as e:      # any failure: try the next host
            log.warning("published chirps: host %s failed (%s)", host, type(e).__name__)
            continue
        p = Published(chirps, rejects, now(), host)
        try:
            _write_cache(state_dir, p)
        except OSError as e:
            log.warning("published cache not written (%s)", type(e).__name__)
        return p
    return _read_cache(state_dir / CACHE_NAME) or Published((), (), None, "none")
