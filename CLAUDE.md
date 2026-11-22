# CLAUDE.md

Guidance for Claude Code in this repository.

## What this project is

**chirp-recorder** times CODAR FMCW chirps on IQ for the HamSCI sigmond
suite. It replaces codar-sounder. It asks radiod for a raw IQ channel,
dechirps in Python, and writes sigmondb `arrival` and `capture` rows. A
later stopgap also writes `codar.spots` rows and the JSONL that codar-sounder
writes today.

Time comes from `/run/hf-timestd/t6-anchor.json`. The code never reads the
host clock into a timing field. Every instant holds integer nanoseconds.

This repo holds the skeleton only (version 0.1.0). Later tasks add the
daemon, config, detector and writer.

Repo: https://github.com/HamSCI/chirp-recorder (not pushed yet).
Author: Michael Hauan (AC0G).

## Quick reference

```bash
uv sync --extra dev
uv run pytest
uv run chirp-recorder version --json
sudo ./scripts/install.sh        # production install, Pattern A
bash -n scripts/install.sh       # syntax check
```

## Architecture

- `src/chirp_recorder/cli.py` holds the command line. `replay` runs
  codar_grab captures through the detector (`core/replay.py`); `compare`
  gives the side-by-side verdict against codar-sounder (`core/compare.py`):
  chirp-recorder's rows from its own `rows/` JSONL, codar-sounder's from
  `--codar-rows DIR` or else the sink queue. `tests/golden/README.md` says where the golden
  captures live.
- `src/chirp_recorder/_reporter_id.py` converts reporter-id spellings
  (`to_key`, `to_storage`, `systemd_instance`, `display`). It copies
  codar-sounder's module unchanged and goes when codar-sounder retires.
- Shared DSP lives in hamsci-dsp. No client copies it from another.
- ka9q-python serves all radiod access, with `client_id = "chirp-recorder"`.
  The client never picks a multicast destination.

## Contract

`CONTRACT_VERSION = "0.8"`. Config is per instance only:
`/etc/chirp-recorder/<KEY>.toml`. There is no shared config. The unit passes
`--instance %i`, and `%i` holds the reporter key. The instance name and the
radiod it reads stay separate settings.

`deploy.toml` declares two `[[hs_uploader.pipeline]]` blocks,
`chirp-sigmondb-arrival` and `chirp-sigmondb-capture`, with producer
`chirp-recorder` and tar tags `chirp-arrival` and `chirp-capture`. It declares
no `[[radiod.fragment]]` and no `instances_from`.

## Production paths

- Source: `/opt/git/sigmond/chirp-recorder` (editable install)
- Venv: `/opt/git/sigmond/chirp-recorder/venv`
- Config: `/etc/chirp-recorder/<KEY>.toml`
- Data: `/var/lib/chirp-recorder/<KEY>/`
- Log: `/var/log/chirp-recorder/<KEY>.log`
- Sink: `/var/lib/sigmond/sink.db`
- Service user: `chirprec:chirprec`, a member of the `sigmond` group

## One instance per receiver

A reporter id names a receiver. KEY takes the reporter id with `/` and `=`
turned to `-` (AC0G/B1A becomes AC0G-B1A). One unit,
`chirp-recorder@AC0G-B1A`, serves that receiver. A reporter change moves an
instance and never adds a second one. Nobody retires codar-sounder by hand:
sigmond converts each instance and then retires it.

## Rules

- Run Python in root-run scripts with `-I`. `tests/test_scripts_python_isolated.py` checks it.
- Never write precise station coordinates into docs or tests.
- Every sweep time counts from the sweep's onset at its start frequency, never the centre
  crossing. `docs/SWEEP-TIMING.md` gives the rule and why.
