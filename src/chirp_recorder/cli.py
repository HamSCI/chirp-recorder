"""chirp-recorder command line.  This file holds the version subcommand alone;
later tasks add the rest (daemon, inventory, validate, config)."""
from __future__ import annotations

import argparse
import json

CONTRACT_VERSION = "0.8"


def _handle_version(args) -> None:
    from chirp_recorder import __version__
    from chirp_recorder.version import GIT_INFO
    payload = {
        "name": "chirp-recorder",
        "version": __version__,
        "contract_version": CONTRACT_VERSION,
        "git": GIT_INFO,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"chirp-recorder {__version__}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="chirp-recorder")
    sub = parser.add_subparsers(dest="command", required=True)
    p_ver = sub.add_parser("version", help="Version and git provenance")
    p_ver.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args(argv)
    if args.command == "version":
        _handle_version(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
