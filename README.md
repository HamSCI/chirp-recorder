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

## The scout

The scout watches radiod's spectrum for CODAR chirp sweeps. It reads spectrum
channels only and never IQ. The timing path above keeps all IQ work.

It writes `sweep` and `scout_dwell` rows to the station sink. hs-uploader
sends them to wd30 through the `chirp-sigmondb-sweep` and
`chirp-sigmondb-scout-dwell` pipelines.

It costs radiod one coarse and one fine spectrum channel. On SIGMA that came to
7.9 % of a core. The unit itself runs at `Nice=10` under `CPUQuota=20%`.

It is off by default. To turn it on, set `enabled = true` in the `[scout]` table
of `/etc/chirp-recorder/<KEY>.toml`, then start `chirp-scout@<KEY>.service`.
The config template lists every `[scout]` key with its default. The log is
`/var/log/chirp-recorder/scout-<KEY>.log`.

## Develop

```
uv sync --extra dev
uv run pytest
uv run chirp-recorder version --json
```

MIT license.
