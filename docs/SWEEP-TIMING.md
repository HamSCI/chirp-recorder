# Sweep timing: every delay counts from the sweep's onset

> **Audience:** developers, and anyone who reads chirp-recorder's delays or offsets
> **Status:** current
> **Verified against:** chirp-recorder 3506baf on 2026-11-22 — docs
> **Canonical for:** where a sweep's time reference sits, for chirp-recorder and the sigmondb rows

## The rule

chirp-recorder times every sweep from its **onset**: the instant the transmitter's sweep begins, at
its start frequency.  Four quantities follow from that one reference.

| Quantity | Meaning |
|---|---|
| `delay_ns` in a sigmondb row | the arrival's time after the sweep's onset on UTC, modulo the period |
| a TDMA offset (`tdma_offset_samples`) | the onset's phase after the UTC second, in samples at the channel rate |
| a ground-wave transmitter offset (wd30's solver) | the onset's phase after the UTC second, less the path's travel time, in [-P/2, P/2) |
| radiod's sweep time 0 (`SWEEP_START`, the sweep-epoch pair) | the onset, because the channel's `freq` names the sweep's start frequency |

A chirp ionosonde also gets its **0 MHz chirp time**: the onset extrapolated back along the sweep
to 0 MHz.  With the sweep rate known, the onset and the 0 MHz time convert into each other exactly.

## Why the onset

**Established practice uses it.**  The two communities whose transmitters we hear both time a
sweep from where it starts.
- CODAR SeaSondes start their sweeps from a common GPS time base.  Radars that share a frequency
  each wait a set delay after it, typically about a millisecond (CODAR patents US 6,856,276 and
  US 9,442,188).  The operator sets that delay as "Sweep Delay" in SeaSondeController.plist.  The
  IOOS range-series archive publishes it, and Troyer et al. 2025 (Radio Science) tabulate it.
- Chirp ionosondes state their timing as the time of the first sweep from 0 MHz after the full
  hour.  A sweep that begins above 0 MHz extrapolates linearly back to 0 MHz.  HamSCI's
  chirpsounder2 records a `t0`, the start time of the whole chirp.

Nobody publishes a centre crossing.  So an offset we measure from the onset compares directly
with what an operator configured and what others have published.  One measured from the centre
would need a correction before anyone could compare it, and that correction would differ from one
transmitter to the next.

**The onset belongs to the transmitter.**  The transmitter decides when its sweep starts.  The
centre crossing depends on what the receiver did: tune a channel to the middle of the sweep, and
the sweep crosses zero there, half a sweep after the onset.  A reference set by the receiver's
tuning changes whenever the tuning changes.

**One reference everywhere stops a class of faults.**  On 2026-11-22 three faults turned up, all
of one kind: two programs that disagreed about where a sweep starts.
- codar-sounder tuned to the sweep's centre and reported the centre crossing.  Its delays ran
  half a sweep late, 0.5 s at 4.463 MHz (ops ledger `codar-sounder-half-sweep-delay`).
- codar-sounder's replica and its TDMA scan applied an offset with opposite signs.  A transmitter
  starting 0.2 s after the second, on a 2 ms path, read 402 ms (ledger
  `codar-sounder-tdma-offset-sign`).  chirp-recorder's physics stopgap inherited that, and it
  also placed sweeps on whatever instant a capture began, not on the UTC second.
- Up-chirps landed on negative range in the same stopgap, and the stopgap threw them away.
The two offset faults hid because each program tested only offsets of 0 and half a period.  At
those two values the conventions agree by accident.

## What the rule asks of code

- **State the reference.**  Any function that takes or returns a sweep time says, in its
  docstring, that it counts from the onset.
- **Tune to the start frequency** when radiod dechirps.  radiod's oscillator sits at the channel's
  `freq` at sweep time 0, so `freq` must name the start frequency.  radiod's man page already
  defines it so.  A client that thinks in centre and bandwidth converts:
  start = centre − rate × duration / 2.
- **Test away from the coincidences.**  Timing tests use offsets other than 0 and P/2 (0.2 s and
  0.7 s, say), and both sweep directions.  They check the true range, not agreement with another
  program.
- **Correct, never redefine, an outside quantity.**  When a source reports from another reference,
  as codar-sounder before 0.8.0 does, the reader corrects it on the way in (chirp-recorder's
  `compare`, wd30's `half_sweep.py`).  The stored meaning stays the onset.

## Related

- [../README.md](../README.md): what chirp-recorder does.
- radiod's man page (ka9q-radio `docs/ka9q-radio-3.md`): `sweep-rate`, `sweep-period`,
  `sweep-duration`, `sweep-start`.
