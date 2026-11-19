"""The tie from the RX888 input counter to UTC.

`Tie.label(iv)` gives one interval a UTC label and says how it got it:

  t6         the T6 anchor file projects the interval's input index to UTC
             nanoseconds.  It applies only when the file reads fresh, names
             this radiod, and belongs to the interval's run.
  authority  radiod's own RTP-to-GPS arithmetic plus hf-timestd's published
             RTP-to-UTC offset, as codar-sounder's acquire_anchor_utc applies it.
  radiod     radiod's arithmetic alone, when no authority offset is usable.

Only a T6 label carries the T6 provenance block.  The other two carry the
null block, which keeps offsets T6-only, and name themselves in
`engineering` (`delay_basis`, `t_level`, and for them `authority_sigma_ns`
and `chain_delay_ns`).

`label()` stays pure: it reads the interval's frozen StatusPair, never a
channel or the network.  The host clock only judges the anchor file's age.
Every instant stays an integer; only the sub-second term rounds.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hamsci_dsp.sigmondb_rows import provenance
from hamsci_dsp.t6_anchor import ANCHOR_PATH, read_t6_anchor

# The GPS-to-UTC step ka9q-python's rtp_to_utc applies (rtp_recorder.py:204:
# gps_time + BILLION * (GPS_UTC_OFFSET - GPS_LEAP_SECONDS)).  Imported, not
# copied, so a leap-second change in ka9q-python reaches this client.
from ka9q.rtp_recorder import GPS_LEAP_SECONDS, GPS_UTC_OFFSET

from chirp_recorder.core.source import Interval, _s32

log = logging.getLogger(__name__)

COORDINATION_ENV = Path("/etc/sigmond/coordination.env")

# Same-run tolerance for the counter epoch.  hf-timestd allows 2.0 s
# (COUNTER_EPOCH_SKEW_ALLOWANCE_S, counter_epoch_tracker.py:70) because
# radiod's GPS_TIME arrives up to 816 ms late on B4, so two readers' least-late
# epochs differ by that much within one run.
SAME_RUN_TOL_S = 2.0


@dataclass(frozen=True)
class Label:
    utc_ns: int
    delay_basis: str            # "t6" | "authority" | "radiod"
    t_level: Optional[str]
    provenance: dict
    engineering: dict


def normalise_status(name: str) -> str:
    """Lower case, port dropped, trailing dot and `.local` dropped.  An IPv6
    literal keeps all its groups (`[v6]:port` loses only the port)."""
    n = (name or "").strip().lower()
    if n.startswith("["):
        end = n.find("]")
        n = n[1:end] if end > 0 else n
    elif n.count(":") == 1:
        host, port = n.split(":")
        if port.isdigit():
            n = host
    n = n.rstrip(".")
    if n.endswith(".local"):
        n = n[: -len(".local")]
    return n.rstrip(".")


def _env_key(radiod_id: str) -> str:
    """sigmond's coordination._env_key: upper case, non-alphanumerics to `_`, runs collapsed."""
    out = []
    for ch in str(radiod_id):
        c = ch.upper() if ch.isalnum() else "_"
        if c == "_" and out and out[-1] == "_":
            continue
        out.append(c)
    return "".join(out).strip("_")


def radiod_id_for_status(status: str, path: Path = COORDINATION_ENV) -> str:
    """The radiod id sigmond files `status` under: the `<id>` of the
    `RADIOD_<id>_STATUS=<dns>` line whose value names the same radiod.  With no
    such line, the status name without its `.local` suffix."""
    want = normalise_status(status)
    try:
        for line in Path(path).read_text().splitlines():
            k, sep, v = line.strip().partition("=")
            if (sep and k.startswith("RADIOD_") and k.endswith("_STATUS")
                    and normalise_status(v.strip().strip("'\"")) == want):
                return k[len("RADIOD_"):-len("_STATUS")]
    except OSError:
        pass
    return want


def chain_delay_from_env(radiod_id: str, path: Path = COORDINATION_ENV) -> Optional[int]:
    """`RADIOD_<id>_CHAIN_DELAY_NS` from coordination.env, or None."""
    key = f"RADIOD_{_env_key(radiod_id)}_CHAIN_DELAY_NS"
    try:
        for line in Path(path).read_text().splitlines():
            k, sep, v = line.strip().partition("=")
            if sep and k == key:
                return int(v.strip().strip("'\""))
    except (OSError, ValueError):
        return None
    return None


def _radiod_utc_ns(iv: Interval) -> int:
    p = iv.status_pair
    delta = _s32(iv.first_rtp - p.rtp_timesnap)
    utc_at_snap = p.gps_time_ns + 1_000_000_000 * (GPS_UTC_OFFSET - GPS_LEAP_SECONDS)
    return utc_at_snap + round(delta * 1_000_000_000 / p.output_samprate)


class Tie:
    def __init__(self, *, radiod_status: str, anchor_path: Path = ANCHOR_PATH,
                 authority=None, chain_delay_ns: Optional[int] = None,
                 clock_ns=time.time_ns):
        self.radiod_status = radiod_status
        self.anchor_path = Path(anchor_path)
        self.authority = authority
        self.chain_delay_ns = chain_delay_ns
        self.clock_ns = clock_ns
        self._state: dict = {}          # channel -> last T6 verdict
        self._auth_failed = False

    def _note(self, iv: Interval, state, msg: str = "", *args) -> None:
        """Log INFO when a channel's T6 verdict changes; say nothing otherwise."""
        if self._state.get(iv.channel) != state:
            prev = self._state.get(iv.channel)
            self._state[iv.channel] = state
            if msg:
                log.info(msg, *args)
            elif prev is not None:
                log.info("channel %s: T6 anchor applies again", iv.channel)

    def label(self, iv: Interval) -> Label:
        anchor = read_t6_anchor(self.anchor_path, now_ns=self.clock_ns())
        if anchor is not None:
            mine, theirs = normalise_status(self.radiod_status), normalise_status(anchor.radiod_status)
            if mine != theirs:
                self._note(iv, ("name", mine, theirs),
                               "channel %s: T6 anchor names radiod %r, this channel reads %r; "
                               "not using T6", iv.channel, theirs, mine)
            elif not anchor.same_run(iv.input_epoch_gps_s, SAME_RUN_TOL_S):
                self._note(iv, ("run",),
                               "channel %s: T6 anchor belongs to another run (epoch %.3f s, "
                               "interval %.3f s); not using T6", iv.channel,
                               anchor.input_epoch_gps_s, iv.input_epoch_gps_s)
            else:
                self._note(iv, ("t6",))
                return Label(
                    utc_ns=anchor.utc_ns_at(iv.first_input_index),
                    delay_basis="t6", t_level="T6",
                    provenance=provenance("T6", anchor.counter_epoch_id,
                                          u_epoch_ns=anchor.u_epoch_ns, k=anchor.k, p=anchor.p,
                                          a_level_provenance=anchor.a_level_provenance),
                    engineering={"delay_basis": "t6", "t_level": "T6"})

        else:
            self._note(iv, ("noanchor",),
                       "channel %s: no fresh T6 anchor; not using T6", iv.channel)

        base = _radiod_utc_ns(iv)
        snap = None
        if self.authority is not None:
            try:
                snap = self.authority.read()
                if self._auth_failed:
                    log.info("authority read works again")
                self._auth_failed = False
            except Exception as exc:  # noqa: BLE001 - never crash the audio path
                if not self._auth_failed:
                    log.warning("authority read failed: %s", exc)
                self._auth_failed = True
        if snap is not None and snap.offset_usable:
            eng = {"delay_basis": "authority", "t_level": snap.t_level_active,
                   "authority_sigma_ns": snap.sigma_ns, "chain_delay_ns": self.chain_delay_ns}
            return Label(utc_ns=base + int(snap.rtp_to_utc_offset_ns), delay_basis="authority",
                         t_level=snap.t_level_active, provenance=provenance(None),
                         engineering=eng)
        eng = {"delay_basis": "radiod", "t_level": None,
               "authority_sigma_ns": None, "chain_delay_ns": self.chain_delay_ns}
        return Label(utc_ns=base, delay_basis="radiod", t_level=None,
                     provenance=provenance(None), engineering=eng)
