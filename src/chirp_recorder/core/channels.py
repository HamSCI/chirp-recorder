"""Channels and groups: what one radiod channel can hold.

A channel holds the transmitters whose sweeps fit inside USABLE_FRACTION of
its sample rate.  Transmitters that share a waveform (TDMA sites) form one
group.  channel_errors refuses any transmitter its channel cannot receive.
"""
from __future__ import annotations

from dataclasses import dataclass

from hamsci_dsp.chirp.waveform import Waveform, samples_per_period

from chirp_recorder.core.catalogue import Transmitter

SAMPLE_RATES_HZ = (64_000, 96_000, 128_000, 192_000)
USABLE_FRACTION = 0.8


@dataclass(frozen=True)
class Group:
    waveform: Waveform
    tx_ids: tuple[str, ...]


@dataclass(frozen=True)
class ChannelPlan:
    name: str
    status: str
    centre_hz: float
    sample_rate_hz: int
    groups: tuple[Group, ...]


def groups_for(txs: list[Transmitter]) -> tuple[Group, ...]:
    """One group per waveform group key, in first-seen order."""
    by_key: dict[str, tuple[Waveform, list[str]]] = {}
    for t in txs:
        wf = t.waveform()
        by_key.setdefault(wf.group_key(), (wf, []))[1].append(t.id)
    return tuple(Group(wf, tuple(ids)) for wf, ids in by_key.values())


def _edges(t: Transmitter) -> tuple[float, float]:
    return t.centre_hz - t.sweep_bw_hz / 2, t.centre_hz + t.sweep_bw_hz / 2


def form_channels(txs: list[Transmitter], status: str, *,
                  max_rate_hz: int = 192_000) -> list[ChannelPlan]:
    rates = [r for r in SAMPLE_RATES_HZ if r <= max_rate_hz] or [max_rate_hz]
    limit = USABLE_FRACTION * max(rates)
    pending = sorted(txs, key=lambda t: (t.centre_hz, t.id))
    out: list[ChannelPlan] = []
    while pending:
        members = [pending.pop(0)]
        lo, hi = _edges(members[0])
        while pending:
            plo, phi = _edges(pending[0])
            if max(hi, phi) - min(lo, plo) > limit:
                break
            lo, hi = min(lo, plo), max(hi, phi)
            members.append(pending.pop(0))
        span = hi - lo
        rate = next((r for r in rates if USABLE_FRACTION * r >= span), max(rates))
        centre = (lo + hi) / 2
        out.append(ChannelPlan(f"{centre / 1e6:.4f}MHz", status, centre, rate,
                               groups_for(members)))
    return out


def channel_errors(ch: ChannelPlan, txs: dict[str, Transmitter]) -> list[str]:
    errs: list[str] = []
    if ch.status.strip().startswith("<"):
        errs.append(f"channel {ch.name}: status {ch.status!r} is a placeholder")
    half_usable = USABLE_FRACTION * ch.sample_rate_hz / 2
    for g in ch.groups:
        for tid in g.tx_ids:
            t = txs.get(tid)
            if t is None:
                errs.append(f"channel {ch.name}: transmitter {tid} is not in the catalogue")
                continue
            lo, hi = _edges(t)
            if lo < ch.centre_hz - half_usable or hi > ch.centre_hz + half_usable:
                errs.append(
                    f"channel {ch.name}: transmitter {tid} sweeps "
                    f"{lo / 1e6:.4f}-{hi / 1e6:.4f} MHz, outside the usable band "
                    f"of {ch.sample_rate_hz} Hz around {ch.centre_hz / 1e6:.4f} MHz")
        try:
            samples_per_period(g.waveform, ch.sample_rate_hz)
        except ValueError:
            errs.append(
                f"channel {ch.name}: group {g.waveform.group_key()} period "
                f"{g.waveform.period_s} s holds no whole number of samples at "
                f"{ch.sample_rate_hz} Hz")
    return errs
