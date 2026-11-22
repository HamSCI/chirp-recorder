# Golden captures

`tests/test_golden.py` and `test_parity_on_a_golden_capture` in
`tests/test_physics_parity.py` read real captures that stay outside this
repository. They fail, and never skip, when the captures are absent.

## Where they live

Task 17 took one night of 4.463 MHz captures at K3LR and W2NAF, 2026-11-19
22:00Z to 2026-11-20 12:00Z. Each capture holds 30 s of IQ at 64 kS/s from
`codar_grab.py`. They sit on the devbox:

    /home/mjh/hamsci/tasks/chirp-registry/golden/K3LR/    index.txt, 16 captures
    /home/mjh/hamsci/tasks/chirp-registry/golden/W2NAF/   index.txt, 19 captures

The folder's own `README.md` says how Task 17 took them and how to take
another night.

Point the tests elsewhere with `CHIRP_GOLDEN_DIR`:

    CHIRP_GOLDEN_DIR=/path/to/golden uv run pytest -q

The reference CLEAN, `tasks/codar-mf/clean.py`, sits at
`/home/mjh/hamsci/tasks/codar-mf/clean.py`; `CHIRP_CLEAN_PY` overrides it.

## What each capture carries

Each `index.txt` line labels the capture's first sample:

- `t0`: radiod's `rtp_to_utc` plus hf-timestd's authority offset at capture
  time, with the authority's `tier` and `sigma_ns`.
- `in_idx0`, `in_rate`, `in_epoch`: radiod's input-sample index of x[0], from a
  counter map fitted to the channel's own status pairs.

The hf-timestd minute rows that would place `in_idx0` on T6's ruler
(`relabel.py`) got pruned before the pull. So `replay()` labels each capture
from `t0`, and no such row claims T6. The line's `tier` and `sigma_ns`
describe hf-timestd's authority, not this label, and travel only as
engineering.

On this night `t0` scatters from capture to capture against radiod's own
input counter, across 200 ms at K3LR and 420 ms at W2NAF. The label
cancels out of the comparison with `clean.py`. The ground-wave test places
each capture by the counter instead (`replay.counter_labels`: `in_idx0 /
in_rate` plus the run's epoch). That keeps the spacing between captures
exact and leaves one unknown offset per run, which a delay hold does not
need.

## Keep them

Whoever clears raw data on the devbox keeps this folder until codar-sounder
retires fleet-wide.
