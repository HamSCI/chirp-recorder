"""chirp-recorder command line: daemon, inventory, validate, version, status,
stations, config init|edit.

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
    return parser


_HANDLERS = {"version": _handle_version, "daemon": _handle_daemon,
             "inventory": _handle_inventory, "validate": _handle_validate,
             "status": _handle_status, "stations": _handle_stations,
             "config": _handle_config}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return _HANDLERS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
