"""The scout's dwell plan: which 2 MHz window the fine channel watches next.

DwellCycle is pure.  Callers pass ``now`` (seconds, any steady epoch) and the
published chirps; it reads no clock and opens no socket.  Its order: a
flagged window first (from a coarse hit, oldest first, once each, dropped
after an hour); then a round robin that alternates the fixed targets with
windows centred on published chirps no target window covers.
"""
from __future__ import annotations

from dataclasses import dataclass

from chirp_recorder.scout.published import Published
from chirp_recorder.scout.settings import ScoutSettings

FLAG_MAX_AGE_S = 3600.0


@dataclass(frozen=True)
class Dwell:
    view: str           # "coarse" or "fine"
    centre_hz: float
    span_hz: float
    bin_hz: float
    reason: str         # "band", "chirp" or "flag"


class DwellCycle:
    def __init__(self, settings: ScoutSettings):
        self._s = settings
        self._flags: list[tuple[float, float]] = []     # (centre_hz, flagged at)
        self._ti = 0            # next fixed target
        self._ci = 0            # next chirp window
        self._want_target = True

    def flag(self, centre_hz: float, now: float) -> None:
        """Queue a fine window on a coarse hit."""
        self._flags.append((float(centre_hz), float(now)))

    def _clamp(self, centre_hz: float) -> float:
        half = self._s.fine_span_hz / 2
        return min(max(centre_hz, half), self._s.nyquist_hz - half)

    def _dwell(self, centre_hz: float, reason: str) -> Dwell:
        return Dwell("fine", self._clamp(centre_hz), self._s.fine_span_hz,
                     self._s.fine_bin_hz, reason)

    def _chirp_windows(self, published: Published) -> list[float]:
        """Centres of one window per uncovered chirp, near duplicates merged."""
        half = self._s.fine_span_hz / 2
        targets = [t * 1e6 for t in self._s.targets_mhz]
        wins: list[float] = []
        for c in sorted(s.centre_hz for s in published.chirps):
            if any(abs(c - t) <= half for t in targets):
                continue
            w = self._clamp(c)
            if any(abs(c - e) <= half for e in wins):
                continue
            wins.append(w)
        return wins

    def next(self, published: Published, now: float) -> Dwell:
        self._flags = [f for f in self._flags if now - f[1] <= FLAG_MAX_AGE_S]
        if self._flags:
            centre, _ = self._flags.pop(0)
            return self._dwell(centre, "flag")
        targets = [t * 1e6 for t in self._s.targets_mhz]
        chirps = self._chirp_windows(published)
        take_target = (self._want_target and bool(targets)) or not chirps
        self._want_target = not take_target
        if take_target:
            c = targets[self._ti % len(targets)]
            self._ti += 1
            return self._dwell(c, "band")
        c = chirps[self._ci % len(chirps)]
        self._ci += 1
        return self._dwell(c, "chirp")
