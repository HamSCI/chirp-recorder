# chirp-recorder

chirp-recorder times CODAR chirps on radio IQ. It replaces codar-sounder.

It asks radiod for a raw IQ channel and dechirps the signal in Python. It
takes time from the T6 anchor file that hf-timestd writes
(`/run/hf-timestd/t6-anchor.json`), never from the host clock. For each
processing interval it writes sigmondb `arrival` and `capture` rows through
the sink. The rows keep the schema in `hamsci_dsp.sigmondb_rows`.

This is version 0.1.0, the repo skeleton. Only `chirp-recorder version`
works so far. The daemon arrives in later tasks.

## Install

```
sudo ./scripts/install.sh
```

The script creates the `chirprec` service user, joins it to the `sigmond`
group, builds the venv at `/opt/git/sigmond/chirp-recorder/venv`, and installs
the systemd unit. Sigmond normally runs it for you.

## Configure

One instance serves one receiver, keyed by its reporter id. The key turns `/`
and `=` into `-`, so AC0G/B1A becomes AC0G-B1A.

- Config: `/etc/chirp-recorder/<KEY>.toml`
- Unit: `chirp-recorder@<KEY>.service`
- Data: `/var/lib/chirp-recorder/<KEY>/`
- Log: `/var/log/chirp-recorder/<KEY>.log`

## Develop

```
uv sync --extra dev
uv run pytest
uv run chirp-recorder version --json
```

MIT license.
