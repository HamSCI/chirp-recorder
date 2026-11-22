"""`chirp-recorder config init|edit`: first-run configuration per receiver.

One receiver, one file: /etc/chirp-recorder/<KEY>.toml.  The instance name and
the radiod it reads stay two settings: --instance (or SIGMOND_INSTANCE) names
the receiver, --radiod-status (or SIGMOND_RADIOD_STATUS) names its radiod.

Position (pre-flight C28, and the project owner's ruling): timing uses the
receiver's precise position, taken from sigmond's coordination (STATION_LAT and
STATION_LON in the environment, else coordination.env, else coordination.toml
[host]).  A grid square (STATION_GRID) serves only when no coordinates exist,
and the output says so.  With neither, init refuses.  Output never prints a
coordinate.

--from-codar converts a codar-sounder per-instance file: one [[channel]] per
[[radiod]] block, its [processing] physics keys, and its station position.  It
skips the picker so a side-by-side night compares like with like.
"""
from __future__ import annotations

import grp
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from chirp_recorder._reporter_id import display, to_key, to_storage
from chirp_recorder.config import (DEFAULT_ETC, PHYSICS_DEFAULTS, ConfigError,
                                   grid_centre, load_instance)
from chirp_recorder.core.catalogue import load_seed
from chirp_recorder.core.channels import SAMPLE_RATES_HZ, ChannelPlan, channel_errors, groups_for
from chirp_recorder.core.picker import VendoredMenu

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

COORDINATION_ENV = Path("/etc/sigmond/coordination.env")
COORDINATION_TOML = Path("/etc/sigmond/coordination.toml")
DEFAULT_MAX_CHANNELS = 2
SERVICE_GROUP = "chirprec"
log = logging.getLogger(__name__)
_REPO = Path(__file__).resolve().parent.parent.parent
_WIZARD_PATH = _REPO / "scripts" / "config-wizard.sh"
CODAR_PHYSICS_KEYS = tuple(k for k in PHYSICS_DEFAULTS if k != "enabled")
CODAR_DEFAULT_RATE_HZ = 64000      # codar-sounder's own default


class Refused(Exception):
    """config init declines rather than guess."""


# --- position ---------------------------------------------------------------

def _coords(lat, lon) -> Optional[tuple[float, float]]:
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lat) and math.isfinite(lon)
            and -90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    if lat == 0.0 and lon == 0.0:      # the template's placeholder, not a position
        return None
    return lat, lon


def coordination_position(env=None) -> Optional[tuple[float, float]]:
    """The precise (lat, lon) sigmond's coordination holds, or None."""
    env = os.environ if env is None else env
    got = _coords(env.get("STATION_LAT"), env.get("STATION_LON"))
    if got:
        return got
    try:     # an init elevated through sudo may not carry the station's env
        bag = {}
        for line in COORDINATION_ENV.read_text().splitlines():
            k, sep, v = line.strip().partition("=")
            if sep:
                bag[k] = v.strip().strip("'\"")
        got = _coords(bag.get("STATION_LAT"), bag.get("STATION_LON"))
        if got:
            return got
    except OSError:
        pass
    try:
        with open(COORDINATION_TOML, "rb") as f:
            host = tomllib.load(f).get("host") or {}
        return _coords(host.get("lat"), host.get("lon"))
    except (OSError, tomllib.TOMLDecodeError):
        return None


def resolve_position(env, *, codar_station: Optional[dict] = None) -> dict:
    """{'lat','lon','grid','precise':bool,'note':str}.  Raises Refused."""
    grid = (env.get("STATION_GRID") or "").strip()
    cs = codar_station or {}
    # Sigmond coordination is the live authority (GPSDO or the operator); a
    # codar-sounder file can be stale, so it comes second.
    got = coordination_position(env)
    where = "sigmond coordination"
    if not got and cs:
        got = _coords(cs.get("receiver_lat"), cs.get("receiver_lon"))
        where = "the codar-sounder file (no coordinates in sigmond coordination)"
    if cs:
        grid = grid or str(cs.get("grid_square") or "").strip()
    if got:
        return {"lat": got[0], "lon": got[1], "grid": grid, "precise": True,
                "note": f"position: precise, from {where}"}
    if grid:
        try:
            lat, lon = grid_centre(grid)
        except ValueError as e:
            raise Refused(f"STATION_GRID: {e}") from None
        return {"lat": lat, "lon": lon, "grid": grid, "precise": False,
                "note": f"position: grid centre of {grid}.  No precise coordinates "
                        "arrived (STATION_LAT/STATION_LON, sigmond coordination).  "
                        "Ranking uses the grid centre; timing offsets stay withheld "
                        "until [station] receiver_lat and receiver_lon are set."}
    raise Refused("no position: the picker needs the receiver's position.  Set "
                  "STATION_LAT and STATION_LON (sigmond coordination) or STATION_GRID")


# --- TOML writer ------------------------------------------------------------

def _scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        s = repr(v)
        return s if any(c in s for c in ".eE") else s + ".0"
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_scalar(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {_scalar(x)}" for k, x in v.items()) + " }"
    raise TypeError(f"unsupported TOML value {type(v).__name__}")


def render(doc: dict, header: str) -> str:
    lines = [header.rstrip(), ""]
    for sect, body in doc.items():
        if sect == "channel":
            for blk in body:
                lines.append("[[channel]]")
                lines += [f"{k} = {_scalar(v)}" for k, v in blk.items()]
                lines.append("")
        else:
            lines.append(f"[{sect}]")
            lines += [f"{k} = {_scalar(v)}" for k, v in body.items()]
            lines.append("")
    return "\n".join(lines)


# --- the document -----------------------------------------------------------

def _identity(args, env) -> tuple[str, str]:
    """(key, reporter_id).  Refuses to guess either."""
    instance = (getattr(args, "instance", None) or env.get("SIGMOND_INSTANCE", "")).strip()
    if not instance:
        raise Refused("no instance: pass --instance or set SIGMOND_INSTANCE")
    try:
        key = to_key(instance)
    except ValueError:
        raise Refused(f"instance {instance!r} is not a reporter key (AC0G-B4)") from None
    rid = (getattr(args, "reporter_id", None) or "").strip()
    if not rid:
        station = env.get("STATION_REPORTER_ID", "").strip()
        try:
            if station and to_key(station) == key:
                rid = station
        except ValueError:
            pass
    if not rid:
        raise Refused(f"instance {key}: no reporter id for this receiver; pass "
                      "--reporter-id (STATION_REPORTER_ID names another key)")
    try:
        if to_key(rid) != key:
            raise Refused(f"reporter {rid!r} has key {to_key(rid)}, not this instance's {key}")
    except ValueError:
        raise Refused(f"reporter {rid!r} is not a reporter id") from None
    return key, display(to_storage(rid))


def _max_channels(args, env) -> int:
    n = getattr(args, "max_channels", None)
    if n is None:
        raw = env.get("CHIRP_MAX_CHANNELS", "")
        try:
            n = int(raw) if raw.strip() else DEFAULT_MAX_CHANNELS
        except ValueError:
            raise Refused(f"CHIRP_MAX_CHANNELS = {raw!r} is not a whole number") from None
    if n < 1:
        raise Refused("max channels must be at least 1")
    return n


def _channel_block(plan: ChannelPlan) -> dict:
    return {"name": plan.name, "status": plan.status, "centre_hz": plan.centre_hz,
            "sample_rate_hz": plan.sample_rate_hz,
            "transmitters": [t for g in plan.groups for t in g.tx_ids]}


def _picked_channels(status: str, pos: dict, key: str, rid: str, cap: int) -> list[dict]:
    from datetime import datetime, timezone
    menu = VendoredMenu(phase="establish")
    choices = menu.choices(receiver_lat=pos["lat"], receiver_lon=pos["lon"],
                           reporter_id=rid, status=status,
                           hour_utc=datetime.now(timezone.utc).hour)
    if not choices:
        raise Refused("no CODAR transmitter in range of the receiver's position")
    return [_channel_block(c.plan) for c in choices[:cap]]


def read_codar(path: Path) -> dict:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise Refused(f"--from-codar {path}: {e}") from None


def _codar_channels(doc: dict, seed: dict) -> list[dict]:
    """One [[channel]] per [[radiod]] block, carried exactly.  Refuses a block
    whose transmitters its channel cannot hear, naming the block."""
    blocks = doc.get("radiod") or []
    if isinstance(blocks, dict):
        blocks = [blocks]
    if not blocks:
        raise Refused("--from-codar: the file has no [[radiod]] block")
    rate = (doc.get("processing") or {}).get("sample_rate_hz", CODAR_DEFAULT_RATE_HZ)
    if rate not in SAMPLE_RATES_HZ:
        raise Refused(f"--from-codar: [processing] sample_rate_hz = {rate!r} must be "
                      f"one of {SAMPLE_RATES_HZ}")
    out: list[dict] = []
    used: set[str] = set()
    for i, b in enumerate(blocks):
        label = b.get("channel_name") or f"#{i + 1}"
        txs = b.get("transmitter") or []
        txs = [txs] if isinstance(txs, dict) else list(txs)
        ids = [str(t.get("id", "")) for t in txs]
        status = str(b.get("status") or "")
        centre = b.get("center_freq_hz")      # codar-sounder's schema has no such key today
        if centre is None:
            centre = next((t["center_freq_hz"] for t in txs if "center_freq_hz" in t), None)
            log.info("--from-codar: block %s has no centre of its own; using its first "
                     "transmitter's center_freq_hz", label)
        if not ids or centre is None or not status:
            raise Refused(f"--from-codar: [[radiod]] block {label} needs a status, "
                          "a transmitter and its center_freq_hz")
        missing = [t for t in ids if t not in seed]
        if missing:
            raise Refused(f"--from-codar: [[radiod]] block {label}: transmitter(s) "
                          f"{', '.join(missing)} are not in the seed catalogue")
        # A codar-sounder offset counts with the opposite sign from an
        # unreferenced scan buffer; no conversion rescues it, so it is dropped.
        for t in txs:
            if "tdma_offset_samples" in t:
                print(f"chirp-recorder: --from-codar: dropped {t['id']} "
                      f"tdma_offset_samples = {t['tdma_offset_samples']}: codar-sounder "
                      "counts it with the opposite sign and no UTC reference.  Re-derive "
                      "it from chirp-recorder's own rows: (delay_ns - path delay) * "
                      "sample_rate_hz / 1e9, mod one period.", file=sys.stderr)
        name = f"{float(centre) / 1e6:.4f}MHz"
        while name in used:
            name += "b"
        used.add(name)
        plan = ChannelPlan(name, status, float(centre), int(rate),
                           groups_for([seed[t] for t in ids]))
        errs = channel_errors(plan, seed)
        if errs:
            raise Refused(f"--from-codar: [[radiod]] block {label} cannot be recorded: "
                          + "; ".join(errs))
        out.append(_channel_block(plan))
    return out


def build_document(args, env) -> tuple[str, str, dict, dict]:
    """(key, reporter_id, document, position).  Raises Refused."""
    key, rid = _identity(args, env)
    codar = read_codar(Path(args.from_codar)) if getattr(args, "from_codar", None) else None
    cap = _max_channels(args, env)
    pos = resolve_position(env, codar_station=(codar or {}).get("station"))
    status = (getattr(args, "radiod_status", None) or env.get("SIGMOND_RADIOD_STATUS", "")).strip()
    if codar is not None and status:
        print(f"chirp-recorder: --radiod-status {status} ignored: --from-codar carries "
              "each block's own radiod", file=sys.stderr)
    if codar is None:
        if not status:
            raise Refused(f"instance {key}: no radiod for this receiver; pass "
                          "--radiod-status or set SIGMOND_RADIOD_STATUS")
        channels = _picked_channels(status, pos, key, rid, cap)
    else:
        channels = _codar_channels(codar, load_seed())
    station: dict = {}
    call = (env.get("STATION_CALL") or ((codar or {}).get("station") or {}).get("callsign") or "")
    if call and call != "N0CALL":
        station["callsign"] = call
    if pos["grid"]:
        station["grid_square"] = pos["grid"]
    if pos["precise"]:
        station["receiver_lat"], station["receiver_lon"] = pos["lat"], pos["lon"]
    doc: dict = {"instance": {"reporter_id": rid}, "station": station,
                 "processing": {"interval_s": 30},
                 "picker": {"phase": "establish", "max_channels": cap}}
    if codar is not None:
        proc = codar.get("processing") or {}
        phys = {k: proc[k] for k in CODAR_PHYSICS_KEYS if k in proc}
        if isinstance(phys.get("snr_threshold_db"), (int, float)) \
                and not isinstance(phys["snr_threshold_db"], bool):
            # codar-sounder's threshold sits on its magnitude-sum profile;
            # chirp-recorder sums energy (physics.2), which reads about twice
            # the dB for a concentrated peak.  Doubling keeps the operating
            # point, as the default's 5 -> 10 dB did (config.PHYSICS_DEFAULTS).
            old = float(phys["snr_threshold_db"])
            phys["snr_threshold_db"] = 2 * old
            print(f"chirp-recorder: --from-codar: snr_threshold_db {old:g} -> {2 * old:g} dB: "
                  "codar-sounder's profile sums magnitude, chirp-recorder's sums energy, "
                  "which reads about twice the dB for the same peak", file=sys.stderr)
        if phys:
            doc["physics"] = phys
    doc["channel"] = channels
    return key, rid, doc, pos


def write_config(etc: Path, key: str, rid: str, doc: dict, source: str) -> Path:
    """Validate through load_instance in a scratch directory, then install."""
    text = render(doc, f"# chirp-recorder instance {key} (reporter {rid}).\n"
                       f"# Written by `chirp-recorder config init`{source}.")
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / f"{key}.toml").write_text(text, encoding="utf-8")
        try:
            load_instance(key, Path(td))
        except ConfigError as e:
            raise Refused(str(e).replace(td + "/", "")) from None
    target = Path(etc) / f"{key}.toml"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    try:
        tmp.write_text(text, encoding="utf-8")
        _restrict(tmp)
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


def _restrict(path: Path) -> None:
    """The file can hold a precise position: never world-readable.  0640 with
    group chirprec (the daemon's group) when that group exists, else 0600."""
    try:
        gid = grp.getgrnam(SERVICE_GROUP).gr_gid
    except KeyError:
        path.chmod(0o600)
        return
    try:
        os.chown(path, -1, gid)
    except PermissionError:
        path.chmod(0o600)
        return
    path.chmod(0o640)


def cmd_config_init(args, etc: Path = DEFAULT_ETC) -> int:
    env = os.environ
    interactive = not getattr(args, "non_interactive", False)
    if interactive and _wizard_available(args):
        return _exec_wizard(args, "init")
    try:
        if interactive:
            _prompt_missing(args, env)
        key, rid = _identity(args, env)
        target = Path(etc) / f"{key}.toml"
        if target.exists() and not getattr(args, "reconfig", False):
            print(f"chirp-recorder: config exists at {target}; use --reconfig to overwrite")
            return 0
        key, rid, doc, pos = build_document(args, env)
        source = f" from {args.from_codar}" if getattr(args, "from_codar", None) else ""
        path = write_config(etc, key, rid, doc, source)
    except Refused as exc:
        print(f"chirp-recorder: config init refused: {exc}", file=sys.stderr)
        return 1
    names = ", ".join(f"{b['name']} ({','.join(b['transmitters'])})" for b in doc["channel"])
    print(f"chirp-recorder: wrote {path} (reporter {rid})")
    print(f"chirp-recorder: {pos['note']}")
    print(f"chirp-recorder: {len(doc['channel'])} channel(s): {names}")
    return 0


def _prompt_missing(args, env) -> None:
    """Plain prompts for what a non-interactive init would refuse over."""
    def ask(label, default=""):
        try:
            got = input(f"{label}{f' [{default}]' if default else ''}: ").strip()
        except EOFError:        # no terminal: leave it missing, so init refuses
            return default
        return got or default
    if not getattr(args, "reporter_id", None):
        args.reporter_id = ask("Reporter id (e.g. AC0G/B4)", env.get("STATION_REPORTER_ID", ""))
    if not getattr(args, "from_codar", None) and not getattr(args, "radiod_status", None):
        args.radiod_status = ask("Radiod status name (e.g. b4-status.local)",
                                 env.get("SIGMOND_RADIOD_STATUS", ""))
    if getattr(args, "max_channels", None) is None:
        raw = ask("Channels to record", str(DEFAULT_MAX_CHANNELS))
        try:
            args.max_channels = int(raw)
        except ValueError:
            raise Refused(f"channels {raw!r} is not a whole number") from None


def cmd_config_edit(args, etc: Path = DEFAULT_ETC) -> int:
    instance = (getattr(args, "instance", None) or os.environ.get("SIGMOND_INSTANCE", "")).strip()
    try:
        key = to_key(instance)
    except ValueError:
        print("chirp-recorder: config edit needs --instance (or SIGMOND_INSTANCE)",
              file=sys.stderr)
        return 1
    target = Path(etc) / f"{key}.toml"
    if not target.exists():
        print(f"chirp-recorder: no config at {target}; run `chirp-recorder config init` first")
        return 1
    if getattr(args, "non_interactive", False):
        print(target.read_text())
        return 0
    if _wizard_available(args):
        return _exec_wizard(args, "edit")
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
    rc = subprocess.run([editor, str(target)], check=False).returncode
    try:
        load_instance(key, Path(etc))
    except ConfigError as exc:
        print(f"chirp-recorder: the edited file does not load:\n{exc}", file=sys.stderr)
        return rc or 1
    return rc


# --- sigmond wizard (when present) ------------------------------------------

try:
    import sigmond.wizard_dispatch as _sigmond_wd
    assert _sigmond_wd.SIGMOND_WIZARD_DISPATCH_API == "1"
except (ImportError, AssertionError):
    _sigmond_wd = None


def _wizard_available(args=None) -> bool:
    """True when sigmond's wizard gate passes and a wizard script ships.
    chirp-recorder ships none yet, so the plain prompts serve."""
    if _sigmond_wd is None or not _WIZARD_PATH.is_file():
        return False
    return bool(_sigmond_wd.is_wizard_available(args, _WIZARD_PATH))


def _exec_wizard(args, mode: str) -> int:
    result = _sigmond_wd.exec_wizard(
        _WIZARD_PATH, extra_env={"CHIRP_RECORDER_CLI": sys.argv[0]}, parse=None,
        extra_args=[mode])
    if result.error:
        print(f"chirp-recorder: wizard exec failed: {result.error}", file=sys.stderr)
        return 1
    return result.returncode
