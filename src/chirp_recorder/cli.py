"""chirp-recorder command line: daemon, inventory, validate, version, status,
stations, config init|edit, replay, compare, scout, scout-replay.

The sigmond unit runs `python -m chirp_recorder.cli daemon --instance %i`.
Every JSON subcommand keeps stdout to the JSON alone; logs go to stderr.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

CONTRACT_VERSION = "0.8"
APPLIED_STATE_MAX_AGE_S = 300.0     # a daemon older than this is not running


def _etc(args) -> Path:
    from chirp_recorder.config import DEFAULT_ETC
    return Path(args.etc) if args.etc else DEFAULT_ETC


def _setup_logging(daemon: bool) -> None:
    level = "INFO" if daemon else "WARNING"
    for k in ("CHIRP_RECORDER_LOG_LEVEL", "CLIENT_LOG_LEVEL"):
        if os.environ.get(k):
            level = os.environ[k].strip().upper()
            break
    if not isinstance(logging.getLevelName(level), int):
        level = "INFO"
    logging.basicConfig(stream=sys.stderr, level=level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _keys(args) -> list[str]:
    """The instances in scope: --instance, else SIGMOND_INSTANCE when its file
    exists, else every file in etc."""
    from chirp_recorder.contract import instance_keys
    etc = _etc(args)
    want = (args.instance or "").strip()
    if not want:
        env = os.environ.get("SIGMOND_INSTANCE", "").strip()
        if env and (etc / f"{_safe_key(env)}.toml").exists():
            want = env
    if want:
        return [_safe_key(want)]
    return instance_keys(etc)


def _safe_key(text: str) -> str:
    from chirp_recorder._reporter_id import to_key
    try:
        return to_key(text)
    except ValueError:
        return text


def _handle_version(args) -> int:
    from chirp_recorder import __version__
    from chirp_recorder.version import GIT_INFO
    payload = {"name": "chirp-recorder", "version": __version__,
               "contract_version": CONTRACT_VERSION, "git": GIT_INFO}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"chirp-recorder {__version__}")
    return 0


def _handle_daemon(args) -> int:
    from chirp_recorder.config import ConfigError, load_instance
    from chirp_recorder.core.daemon import Daemon
    _setup_logging(True)
    key = _safe_key((args.instance or os.environ.get("SIGMOND_INSTANCE", "")).strip())
    if not key:
        print("chirp-recorder: daemon needs --instance (or SIGMOND_INSTANCE)", file=sys.stderr)
        return 2
    try:
        cfg = load_instance(key, _etc(args))
    except ConfigError as exc:
        print(f"chirp-recorder: {exc}", file=sys.stderr)
        return 1
    return int(Daemon(cfg, measure=args.measure_cost).run())


def _handle_inventory(args) -> int:
    from chirp_recorder.contract import build_inventory
    _setup_logging(False)
    print(json.dumps(build_inventory(_keys(args), _etc(args)), indent=2))
    return 0


def _handle_validate(args) -> int:
    from chirp_recorder.contract import build_validate
    _setup_logging(False)
    payload = build_validate(_keys(args), _etc(args))
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 1


def _status_of(key: str, etc: Path) -> dict:
    from chirp_recorder.config import ConfigError, load_instance
    from chirp_recorder.core.daemon import applied_state_path
    try:
        cfg = load_instance(key, etc)
    except ConfigError as exc:
        return {"instance": key, "running": False, "error": str(exc), "channels": []}
    path = applied_state_path(cfg.output_dir)
    try:
        age = time.time() - path.stat().st_mtime      # the host clock judges file age only
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"instance": key, "running": False, "channels": []}
    running = age <= APPLIED_STATE_MAX_AGE_S
    return {"instance": key, "running": running, "age_s": round(age),
            "position_source": doc.get("position_source"),
            "channels": doc.get("channels") or []}


def _handle_status(args) -> int:
    _setup_logging(False)
    states = [_status_of(k, _etc(args)) for k in _keys(args)]
    if args.json:
        print(json.dumps({"instances": states}, indent=2))
        return 0
    if not states:
        print(f"no instance config in {_etc(args)}")
    for st in states:
        if not st["running"]:
            why = f" ({st['error']})" if st.get("error") else ""
            print(f"{st['instance']}: not running{why}")
            continue
        print(f"{st['instance']}: running, state {st['age_s']} s old")
        for c in st["channels"]:
            extra = c.get("refused_reason") or c.get("delay_basis") or ""
            print(f"  {c.get('name')}: {c.get('admission')}"
                  f"{' - ' + str(extra) if extra else ''}, intervals {c.get('intervals', 0)}")
    return 0


def _handle_stations(args) -> int:
    from datetime import datetime, timezone
    from chirp_recorder.config import ConfigError, load_instance
    from chirp_recorder.core.picker import VendoredMenu
    _setup_logging(False)
    keys = _keys(args)
    if len(keys) != 1:
        print("chirp-recorder: stations needs one instance: pass --instance "
              f"({len(keys)} found)", file=sys.stderr)
        return 2
    try:
        cfg = load_instance(keys[0], _etc(args))
    except ConfigError as exc:
        print(f"chirp-recorder: {exc}", file=sys.stderr)
        return 1
    status = cfg.channels[0].status
    menu = VendoredMenu(phase=cfg.picker_phase, reach_table=cfg.reach_km,
                        reach_source=cfg.reach_source)
    chosen = menu.choices(receiver_lat=cfg.receiver_lat, receiver_lon=cfg.receiver_lon,
                          reporter_id=cfg.reporter_id, status=status,
                          hour_utc=datetime.now(timezone.utc).hour)
    out = {"instance": cfg.key, "position_source": cfg.position_source,
           "phase": cfg.picker_phase,
           "choices": [{"name": c.plan.name, "status": c.plan.status,
                        "centre_hz": c.plan.centre_hz,
                        "sample_rate_hz": c.plan.sample_rate_hz,
                        "transmitters": [t for g in c.plan.groups for t in g.tx_ids],
                        "shared_t6": c.shared_t6, "nearest_km": round(c.nearest_km, 1),
                        "reason": c.reason} for c in chosen]}
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        for i, c in enumerate(out["choices"], 1):
            print(f"{i:2d}. {c['name']}  {','.join(c['transmitters'])}  "
                  f"{c['nearest_km']:.0f} km  {c['reason']}")
    return 0


def _handle_config(args) -> int:
    from chirp_recorder import configurator
    _setup_logging(False)
    if args.config_command == "init":
        return configurator.cmd_config_init(args, _etc(args))
    return configurator.cmd_config_edit(args, _etc(args))


def _replay_groups(centre_hz: float, fs: int, gate_us):
    """The seed's groups a channel at centre_hz and fs can hold; --gate-us
    pins every group's gate period (else detect.py fits it per capture)."""
    from dataclasses import replace
    from chirp_recorder.core.catalogue import load_seed
    from chirp_recorder.core.channels import USABLE_FRACTION, groups_for
    half = USABLE_FRACTION * fs / 2
    txs = [t for t in load_seed().values()
           if t.centre_hz - t.sweep_bw_hz / 2 >= centre_hz - half
           and t.centre_hz + t.sweep_bw_hz / 2 <= centre_hz + half]
    groups = list(groups_for(txs))
    if gate_us is not None:
        groups = [replace(g, waveform=replace(g.waveform, gate_period_s=gate_us * 1e-6))
                  for g in groups]
    return groups


def _handle_replay(args) -> int:
    from chirp_recorder.core.replay import replay
    _setup_logging(False)
    groups = _replay_groups(args.centre_hz, args.fs, args.gate_us)
    if not groups:
        print(f"chirp-recorder: no catalogue group fits {args.fs} Hz around "
              f"{args.centre_hz:.0f} Hz", file=sys.stderr)
        return 2
    rows = replay(Path(args.capture_dir), [Path(m) for m in args.minutes or []], groups,
                  fs=args.fs, channel_centre_hz=args.centre_hz, reporter_id=args.reporter_id,
                  label=args.label)
    for r in rows:
        print(json.dumps(r, separators=(",", ":")))
    return 0


def _handle_compare(args) -> int:
    from chirp_recorder.config import ConfigError, load_instance
    from chirp_recorder.core.compare import (compare, read_codar_jsonl_rows, read_jsonl_rows,
                                             read_sink_rows, rung_spacings)
    _setup_logging(False)
    keys = _keys(args)
    if len(keys) != 1:
        print("chirp-recorder: compare needs one instance: pass --instance", file=sys.stderr)
        return 2
    try:
        cfg = load_instance(keys[0], _etc(args))
    except ConfigError as exc:
        print(f"chirp-recorder: {exc}", file=sys.stderr)
        return 2
    window = dict(reporter_id=cfg.reporter_id, since_iso=args.since, until_iso=args.until)
    # chirp-recorder: its own daily JSONL, which nothing drains.
    rows_dir = Path(cfg.output_dir) / "rows"
    chirp = read_jsonl_rows(rows_dir, **window)
    if args.codar_rows:
        # codar-sounder's own daily JSONL (format: compare.read_codar_jsonl_rows)
        codar = read_codar_jsonl_rows(Path(args.codar_rows), **window)
        codar_src = str(args.codar_rows)
    else:
        # The sink queue, which codar-sounder's upload pipeline empties after
        # each send: a window behind the last upload reads short, and the
        # verdict then says "insufficient".
        from hs_uploader.sink.writer import sink_path
        db = sink_path()
        if not db or not Path(db).exists():
            print(f"chirp-recorder: no sink to read ({db or 'no sink path set'}); "
                  "pass --codar-rows DIR for codar-sounder's JSONL", file=sys.stderr)
            return 2
        codar = read_sink_rows(Path(db), producer="codar-sounder", **window)
        codar_src = f"{db} pending_uploads (drained after each upload)"
    sources = {"chirp-recorder": str(rows_dir), "codar-sounder": codar_src}
    # A rung spacing is known only where the group's gate period is (the seed
    # gives the Mid-Atlantic 4.463 MHz group's).
    rung_s = rung_spacings([g for ch in cfg.channels for g in ch.groups])
    v = compare(chirp, codar, rung_s=rung_s, match=args.match, mean_offset=args.mean_offset)
    if args.json:
        print(json.dumps({**v.to_dict(), "sources": sources}, indent=2))
    else:
        print(f"{cfg.key}: {v.status}, match {v.match}  ({len(chirp)} chirp-recorder rows, "
              f"{len(codar)} codar-sounder rows)")
        for r in v.rows:
            print(f"  {r.name:10s} {r.value!s:>12} limit {r.limit!s:>8} "
                  f"{'gates' if r.gating else 'reported'} {'ok' if r.passed else 'FAIL'}")
    return 0 if v.status == "pass" else 1


def _handle_scout(args) -> int:
    import signal
    import threading
    from chirp_recorder.config import ConfigError, load_instance
    from chirp_recorder.scout import runner
    from chirp_recorder.scout.settings import scout_settings
    _setup_logging(True)
    try:
        cfg = load_instance(_safe_key(args.key), _etc(args))
        settings = scout_settings(cfg)
    except ConfigError as exc:
        print(f"chirp-recorder: {exc}", file=sys.stderr)
        return 1
    if not settings.enabled:
        print(f"{cfg.key}: scout disabled ([scout] enabled = false)")
        return 0
    if args.dry_run:
        def writer(table, row):
            print(json.dumps({"table": table, "row": row}, separators=(",", ":")),
                  flush=True)
        close = None
    else:
        writer = runner.make_scout_writer(settings.state_dir)
        close = writer.close
    stop = threading.Event()
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: stop.set())
    try:
        runner.Scout(settings, cfg.reporter_id, writer=writer).run(stop, max_dwells=args.dwells)
    finally:
        if close is not None:
            close()
    return 0


def _handle_scout_replay(args) -> int:
    import numpy as np
    from chirp_recorder.scout.runner import replay
    _setup_logging(False)
    with np.load(args.file) as z:
        power = np.asarray(z["power_db"], dtype=float)
        t_s = np.asarray(z["t_s"], dtype=float)
        f0 = args.f0_hz if args.f0_hz is not None else float(z["f0_hz"]) if "f0_hz" in z else None
        bin_hz = (args.bin_hz if args.bin_hz is not None
                  else float(z["bin_hz"]) if "bin_hz" in z else None)
    if f0 is None or bin_hz is None:
        print("chirp-recorder: the file holds no f0_hz or bin_hz; pass --f0-hz and --bin-hz",
              file=sys.stderr)
        return 2
    for s in replay(power, t_s, f0_hz=f0, bin_hz=bin_hz, view=args.view):
        print(json.dumps({"centre_hz": round(s.centre_hz, 1), "span_hz": round(s.span_hz, 1),
                          "rate_hz_per_s": round(s.rate_hz_per_s, 1),
                          "period_s": None if s.period_s is None else round(s.period_s, 4),
                          "duration_s": None if s.duration_s is None else round(s.duration_s, 4),
                          "n_sweeps": s.n_sweeps, "fit_score": round(s.fit_score, 3),
                          "snr_db": round(s.snr_db, 1), "clipped": s.clipped},
                         separators=(",", ":")))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chirp-recorder")
    parser.add_argument("--etc", help="Per-instance config directory "
                        "(default /etc/chirp-recorder)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("version", help="Version and git provenance")
    p.add_argument("--json", action="store_true", help="Emit JSON")

    p = sub.add_parser("daemon", help="Run one instance")
    p.add_argument("--instance", help="Instance key (AC0G-B4); default SIGMOND_INSTANCE")
    p.add_argument("--measure-cost", action="store_true",
                   help="Admit every channel to measure cost; works only beside "
                        "[capacity] measure = true")

    for name, helptext in (("inventory", "Contract inventory"),
                           ("validate", "Contract validation")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--json", action="store_true", help="Emit JSON (always)")
        p.add_argument("--instance")

    p = sub.add_parser("status", help="What each instance is doing")
    p.add_argument("--json", action="store_true")
    p.add_argument("--instance")

    p = sub.add_parser("stations", help="The picker's ranked choices with reasons")
    p.add_argument("--json", action="store_true")
    p.add_argument("--instance")

    p = sub.add_parser("config", help="Configure an instance")
    cs = p.add_subparsers(dest="config_command", required=True)
    pi = cs.add_parser("init", help="Write a receiver's config")
    pi.add_argument("--non-interactive", action="store_true")
    pi.add_argument("--instance")
    pi.add_argument("--reporter-id")
    pi.add_argument("--radiod-status")
    pi.add_argument("--max-channels", type=int)
    pi.add_argument("--from-codar", metavar="PATH",
                    help="Convert a codar-sounder per-instance file")
    pi.add_argument("--reconfig", action="store_true", help="Overwrite an existing file")
    pe = cs.add_parser("edit", help="Edit a receiver's config")
    pe.add_argument("--non-interactive", action="store_true")
    pe.add_argument("--instance")
    p = sub.add_parser("replay", help="Replay codar_grab captures; prints arrival rows (JSONL)")
    p.add_argument("capture_dir", help="Directory holding index.txt and the .npy captures")
    p.add_argument("--minutes", nargs="*", help="hf-timestd minute rows (JSONL) to relabel by; "
                   "without them each capture keeps its index.txt label")
    p.add_argument("--centre-hz", type=float, default=4_463_000.0)
    p.add_argument("--fs", type=int, default=64_000)
    p.add_argument("--gate-us", type=float, help="Pin every group's gate period (us)")
    p.add_argument("--reporter-id", help="Default: the directory's name")
    p.add_argument("--label", choices=("auto", "counter"), default="auto",
                   help="auto: relabel by --minutes, else each index.txt label; "
                        "counter: radiod's input counter, one epoch per run")

    p = sub.add_parser("compare", help="Side-by-side verdict against codar-sounder")
    p.add_argument("--instance")
    p.add_argument("--since", required=True, help="ISO UTC, inclusive")
    p.add_argument("--until", default="9999-12-31T23:59:59Z", help="ISO UTC, exclusive")
    p.add_argument("--match", choices=("bin", "exact"), default="bin",
                   help="bin (default): within half a codar-sounder range bin + 5 us; "
                        "exact: within 5 us")
    p.add_argument("--codar-rows", metavar="DIR",
                   help="codar-sounder's daily arrival JSONL (arrivals-YYYYMMDD.jsonl, "
                        "e.g. /var/lib/codar-sounder/arrivals); default: the sink queue")
    p.add_argument("--mean-offset", action="store_true",
                   help="Gate on |mean(chirp - codar)| <= 5 us over >= 20 matched clusters")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("scout", help="Watch radiod's spectrum for chirp sweeps")
    p.add_argument("key", help="Instance key (AC0G-B4)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print rows to stdout as JSON lines; write no sink and no JSONL")
    p.add_argument("--dwells", type=int, help="Stop after N fine dwells")

    p = sub.add_parser("scout-replay", help="Run the scout's line search on a saved waterfall")
    p.add_argument("file", help=".npz with power_db (frames x bins), t_s, f0_hz, bin_hz")
    p.add_argument("--bin-hz", type=float, help="Bin width (default: the file's bin_hz)")
    p.add_argument("--f0-hz", type=float, help="Bin 0's frequency (default: the file's f0_hz)")
    p.add_argument("--view", choices=("fine", "coarse"), default="fine",
                   help="Whose caps to apply (default fine)")
    return parser


_HANDLERS = {"version": _handle_version, "daemon": _handle_daemon,
             "inventory": _handle_inventory, "validate": _handle_validate,
             "status": _handle_status, "stations": _handle_stations,
             "config": _handle_config, "replay": _handle_replay,
             "compare": _handle_compare, "scout": _handle_scout,
             "scout-replay": _handle_scout_replay}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return _HANDLERS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
