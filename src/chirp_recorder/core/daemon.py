"""The daemon: one instance, every channel its config names.

Admission runs first (capacity.admit).  Each admitted channel gets one worker
thread that labels, detects and writes its intervals.  A channel that fails
leaves the others running; its worker restarts the source 30 s later.  The
daemon never sheds a running channel.  A refused channel waits for the next
start or SIGHUP.

Rows go through one RowWriter, the sole owner of the sink connection
(pre-flight C12), physics spots included.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from chirp_recorder import __version__
from chirp_recorder.config import InstanceConfig, physics_settings
from chirp_recorder.core.capacity import admit, load_costs, read_headroom
from chirp_recorder.core.catalogue import load_seed
from chirp_recorder.core.channels import ChannelPlan
from chirp_recorder.core.detect import detect_group
from chirp_recorder.core.rows import rows_for_group
from chirp_recorder.core.source import RawChannelSource
from chirp_recorder.core.timing import (COORDINATION_ENV, Tie, chain_delay_from_env,
                                        radiod_id_for_status)
from hamsci_dsp import sigmondb_rows as sr
from hamsci_dsp.timing import APPLIED_STATE_SCHEMA
from chirp_recorder.physics.stopgap import utc_datetime
from chirp_recorder.core.writer import PhaseStore, RowWriter, make_sink_writers

log = logging.getLogger(__name__)

RESTART_DELAY_S = 30.0
APPLIED_STATE_PERIOD_S = 60.0
TICK_S = 5.0
STALE_FACTOR = 3.0


def sd_notify(message: str) -> None:
    """Minimal sd_notify: no python-systemd dependency."""
    sock_path = os.environ.get("NOTIFY_SOCKET")
    if not sock_path:
        return
    if sock_path.startswith("@"):
        sock_path = "\0" + sock_path[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect(sock_path)
            s.sendall(message.encode("utf-8"))
    except OSError as exc:
        log.warning("sd_notify failed (%s): %s", message, exc)


def applied_state_path(output_dir: Path) -> Path:
    return Path(output_dir) / "applied-state.json"


class ChannelWorker(threading.Thread):
    def __init__(self, plan: ChannelPlan, *, source_factory: Callable, tie, writer,
                 interval_s: float, reporter_id: str, version: str,
                 physics=None, position_source: str = "precise",
                 restart_delay_s: float = RESTART_DELAY_S,
                 clock: Callable[[], float] = time.monotonic):
        super().__init__(name=f"chirp-chan-{plan.name}", daemon=True)
        self.plan = plan
        self.source_factory = source_factory
        self.tie = tie
        self.writer = writer
        self.interval_s = interval_s
        self.reporter_id = reporter_id
        self.version = version
        self.physics = physics
        self.position_source = position_source
        self.restart_delay_s = restart_delay_s
        self._clock = clock
        self._stop_evt = threading.Event()
        self._source = None
        self._spell_logged = False
        self._lock = threading.Lock()
        self.last_interval_at: Optional[float] = None
        self._stats = {"delivered": 0, "failures": 0, "restarts": 0,
                       "rows": 0, "spots": 0, "delay_basis": None, "t_level": None}

    @property
    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def _set(self, **kw) -> None:
        with self._lock:
            self._stats.update(kw)

    def _bump(self, key: str, n: int = 1) -> None:
        with self._lock:
            self._stats[key] += n

    def stop(self) -> None:
        self._stop_evt.set()
        src = self._source
        if src is not None:
            try:
                src.stop()
            except Exception as exc:  # noqa: BLE001
                log.warning("channel %s: source stop failed: %s", self.plan.name, exc)

    def run(self) -> None:
        first = True
        while not self._stop_evt.is_set():
            if not first:
                self._bump("restarts")
            first = False
            src = None
            try:
                src = self.source_factory(self.plan, interval_s=self.interval_s)
                self._source = src
                if self._stop_evt.is_set():       # stop() came before the source was known
                    break
                for iv in src.intervals():
                    if self._stop_evt.is_set():
                        break
                    try:
                        self._handle(iv)
                        self._spell_logged = False
                    except Exception as exc:  # noqa: BLE001 - one bad interval keeps the source
                        self._bump("failures")
                        self._log_failure("interval failed", exc, restart=False)
            except Exception as exc:  # noqa: BLE001 - a channel never takes the daemon down
                self._bump("failures")
                self._log_failure("source failed", exc, restart=True)
            finally:
                self._source = None
                if src is not None:
                    try:
                        src.stop()
                    except Exception:  # noqa: BLE001
                        pass
            if self._stop_evt.wait(self.restart_delay_s):
                break
        if self.physics is not None:
            try:
                self.physics.close()
            except Exception as exc:  # noqa: BLE001
                log.warning("channel %s: physics close failed: %s", self.plan.name, exc)

    def _log_failure(self, what: str, exc: Exception, *, restart: bool) -> None:
        """The traceback once per failure spell, then one line each time."""
        tail = f"; restarting its source in {self.restart_delay_s:.0f} s" if restart else ""
        if not self._spell_logged:
            self._spell_logged = True
            log.error("channel %s %s: %s%s", self.plan.name, what, exc, tail, exc_info=True)
        else:
            log.error("channel %s %s again: %s%s", self.plan.name, what, exc, tail)

    def _handle(self, iv) -> None:
        label = self.tie.label(iv)
        detector_version = f"chirp-recorder/{self.version}"
        duration = len(iv.samples) / self.plan.sample_rate_hz
        n_rows = 0
        for group in self.plan.groups:
            result = detect_group(iv.samples, fs=self.plan.sample_rate_hz,
                                  channel_centre_hz=self.plan.centre_hz, group=group)
            capture, arrivals = rows_for_group(
                result, label=label, reporter_id=self.reporter_id,
                radiod=self.plan.status, channel=self.plan.name, duration_s=duration,
                detector_version=detector_version,
                input_epoch_gps_s=iv.input_epoch_gps_s, interval_s=self.interval_s,
                position_source=self.position_source)
            phases = []
            period_s = group.waveform.period_s
            for i, a in enumerate(result.arrivals):
                if a.slow is None:
                    continue
                _, delay_ns = sr.split_arrival(int(label.utc_ns) + int(a.tau_ns), period_s)
                phases.append(dict(
                    group=arrivals[i]["group_key"],
                    cpi_start_utc=utc_datetime(label.utc_ns), radiod=self.plan.status,
                    sweep_start_ns=int(label.utc_ns), bin=i, period_s=period_s,
                    tau_ns=int(a.tau_ns), delay_ns=int(delay_ns), level_db=a.level_db,
                    slow=a.slow))
            self.writer.put(capture, arrivals, phases)
            n_rows += len(arrivals) + 1
        spots = 0
        if self.physics is not None:
            try:
                out = self.physics.feed(iv, label)
                if out.spots:
                    self.writer.put(None, [], [], spots=out.spots)
                    spots = len(out.spots)
            except Exception as exc:  # noqa: BLE001 - physics never stops detection
                log.warning("channel %s: physics failed: %s", self.plan.name, exc)
        with self._lock:
            self._stats["delivered"] += 1
            self._stats["rows"] += n_rows
            self._stats["spots"] += spots
            self._stats["delay_basis"] = label.delay_basis
            self._stats["t_level"] = getattr(label, "t_level", None)
        self.last_interval_at = self._clock()


class Daemon:
    def __init__(self, cfg: InstanceConfig, *, headroom_reader=read_headroom, costs=None,
                 source_factory=RawChannelSource, notify=sd_notify, measure: bool = False,
                 clock: Callable[[], float] = time.monotonic,
                 restart_delay_s: float = RESTART_DELAY_S,
                 sink_writers=None, env_files: Optional[list] = None):
        self.cfg = cfg
        self.headroom_reader = headroom_reader
        self._costs = costs
        self.source_factory = source_factory
        self.notify = notify
        self.clock = clock
        self.restart_delay_s = restart_delay_s
        self._sink_writers = sink_writers
        self._env_files = env_files if env_files is not None else [
            COORDINATION_ENV, Path("/etc/chirp-recorder/env") / f"{cfg.key}.env"]
        cap = cfg.raw.get("capacity") or {}
        self.measure = bool(measure) and cap.get("measure") is True
        self._measure_asked = bool(measure)
        self._workers: list[ChannelWorker] = []
        self._refused: dict[str, str] = {}      # channel name -> reason
        self._admitted_estimated = False
        self._writer: Optional[RowWriter] = None
        self._seed = None
        self._ready = False
        self._last_state_at: Optional[float] = None
        self._stop_evt = threading.Event()
        self._hup = threading.Event()
        self._lock = threading.Lock()

    # --- accessors ----------------------------------------------------------

    @property
    def workers(self) -> list[ChannelWorker]:
        return list(self._workers)

    def worker(self, name: str) -> ChannelWorker:
        for w in self._workers:
            if w.plan.name == name:
                return w
        raise KeyError(name)

    # --- admission and workers ---------------------------------------------

    def _ensure_writer(self) -> RowWriter:
        if self._writer is None:
            sinks = self._sink_writers if self._sink_writers is not None \
                else make_sink_writers()
            self._writer = RowWriter(
                output_dir=self.cfg.output_dir, sink_writers=sinks,
                phase_store=PhaseStore(Path(self.cfg.output_dir) / "phase",
                                       radiod=self.cfg.key))
        return self._writer

    def _physics_for(self, plan: ChannelPlan):
        if not self.cfg.physics.get("enabled"):
            return None
        from chirp_recorder.physics import PhysicsStopgap
        if self._seed is None:
            self._seed = load_seed()
        st = self.cfg.raw.get("station") or {}
        try:
            return PhysicsStopgap(
                plan, self._seed, settings=physics_settings(self.cfg, plan),
                output_dir=self.cfg.output_dir, reporter_id=self.cfg.reporter_id,
                radiod=plan.status, host_call=str(st.get("callsign") or ""),
                host_grid=self.cfg.grid)
        except Exception as exc:  # noqa: BLE001
            log.warning("channel %s: physics stopgap unavailable: %s", plan.name, exc)
            return None

    def _start_worker(self, plan: ChannelPlan) -> ChannelWorker:
        tie = Tie(radiod_status=plan.status,
                  chain_delay_ns=chain_delay_from_env(radiod_id_for_status(plan.status)))
        w = ChannelWorker(
            plan, source_factory=self.source_factory, tie=tie, writer=self._ensure_writer(),
            interval_s=self.cfg.interval_s, reporter_id=self.cfg.reporter_id,
            version=__version__, physics=self._physics_for(plan),
            position_source=self.cfg.position_source,
            restart_delay_s=self.restart_delay_s, clock=self.clock)
        self._workers.append(w)
        w.start()
        return w

    def _costs_now(self) -> dict:
        if self._costs is None:
            try:
                self._costs = load_costs()
            except Exception as exc:  # noqa: BLE001
                log.warning("costs.toml unreadable (%s); treating every cost as unmeasured", exc)
                self._costs = {}
        return self._costs

    def _admit(self, plans: list[ChannelPlan]):
        return admit(plans, self.headroom_reader(), self._costs_now(), __version__,
                     measure=self.measure)

    @staticmethod
    def _describe(plan: ChannelPlan) -> str:
        groups = "; ".join(f"{g.waveform.centre_hz / 1e6:.4f} MHz: {', '.join(g.tx_ids)}"
                           for g in plan.groups)
        return (f"channel {plan.name}: radiod {plan.status}, centre "
                f"{plan.centre_hz / 1e6:.4f} MHz, {plan.sample_rate_hz} S/s, groups [{groups}]")

    def start_workers(self) -> None:
        if self._measure_asked and not self.measure:
            log.info("--measure-cost ignored: [capacity] measure is not true in the instance file")
        if self.measure:
            log.warning("measuring cost: admission bypassed")
        adm = self._admit(list(self.cfg.channels))
        self._admitted_estimated = adm.estimated
        for plan in adm.admitted:
            log.info(self._describe(plan))
            self._start_worker(plan)
        for plan, why in adm.refused:
            self._refused[plan.name] = why
            log.warning("channel %s refused: %s", plan.name, why)
        if adm.refused:
            log.error("instance %s records %d of %d channels; refused: %s", self.cfg.key,
                      len(adm.admitted), len(self.cfg.channels),
                      ", ".join(p.name for p, _ in adm.refused))
        if not self._workers:
            log.error("instance %s admitted no channel (see the refusals above); "
                      "exiting so systemd retries admission", self.cfg.key)
        self._send_ready()

    def _send_ready(self) -> None:
        if self._workers and not self._ready:
            self._ready = True
            self.notify(f"STATUS=recording {len(self._workers)} of "
                        f"{len(self.cfg.channels)} channels")
            self.notify("READY=1")

    def readmit(self) -> None:
        """Re-run admission for refused channels only.  Running channels stay."""
        if not self._refused:
            return
        running = {w.plan.name for w in self._workers}
        # Ask for everything in config order, so a running channel's cost counts.
        adm = self._admit([p for p in self.cfg.channels
                           if p.name in running or p.name in self._refused])
        for plan in adm.admitted:
            if plan.name in self._refused:
                log.info("channel %s admitted on re-check", plan.name)
                log.info(self._describe(plan))
                del self._refused[plan.name]
                self._start_worker(plan)
        for plan, why in adm.refused:
            if plan.name in self._refused and self._refused[plan.name] != why:
                self._refused[plan.name] = why
                log.warning("channel %s still refused: %s", plan.name, why)
        if self._refused:
            log.error("instance %s records %d of %d channels", self.cfg.key,
                      len(self._workers), len(self.cfg.channels))
        self._send_ready()

    # --- watchdog and applied state ----------------------------------------

    def _delivering(self, now: float) -> bool:
        limit = STALE_FACTOR * self.cfg.interval_s
        for w in self._workers:
            t = w.last_interval_at
            if t is not None and now - t <= limit:
                return True
        return False

    def tick(self, now: float) -> None:
        if self._delivering(now):
            self.notify("WATCHDOG=1")
        if self._last_state_at is None or now - self._last_state_at >= APPLIED_STATE_PERIOD_S:
            self._last_state_at = now
            try:
                self._write_applied_state()
            except OSError as exc:
                log.warning("applied-state write failed: %s", exc)

    def applied_state(self) -> dict:
        chans = []
        running = {w.plan.name: w for w in self._workers}
        for plan in self.cfg.channels:
            w = running.get(plan.name)
            chans.append({
                "name": plan.name, "radiod": plan.status,
                "centre_hz": plan.centre_hz, "sample_rate_hz": plan.sample_rate_hz,
                "transmitters": [t for g in plan.groups for t in g.tx_ids],
                "admission": "admitted" if w is not None else "refused",
                "refused_reason": self._refused.get(plan.name),
                "delay_basis": w.stats["delay_basis"] if w else None,
                "intervals": w.stats["delivered"] if w else 0,
            })
        return {"instance": self.cfg.key, "reporter_id": self.cfg.reporter_id,
                "version": __version__, "position_source": self.cfg.position_source,
                "interval_s": self.cfg.interval_s, "measuring_cost": self.measure,
                "estimated_cost": self._admitted_estimated, "channels": chans}

    def timing_authority_applied(self) -> Optional[dict]:
        """The CLIENT-CONTRACT section 3 block: populated when every channel that
        has labelled an interval used a corrected label (T6 or authority),
        else None (a mixed state reads null, never an average)."""
        labelled = [w for w in self._workers if w.stats["delay_basis"] is not None]
        if not labelled or any(w.stats["delay_basis"] not in ("t6", "authority")
                               for w in labelled):
            return None
        bases = sorted({w.stats["delay_basis"] for w in labelled})
        tiers = sorted({w.stats["t_level"] for w in labelled if w.stats["t_level"]})
        radiods = sorted({w.plan.status for w in labelled})
        return {"source": "hf-timestd", "tier": tiers[0] if len(tiers) == 1 else None,
                "sigma_ns": None, "snapshot_age_s": None,
                "radiod_id": radiods[0] if len(radiods) == 1 else None,
                "delay_basis": bases,
                "channels": {"total": len(self._workers), "anchored": len(labelled),
                             "applied": len(labelled)}}

    def _write_applied_state(self) -> None:
        path = applied_state_path(self.cfg.output_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        now = time.time()    # the host clock only dates the file, for a reader's age test
        payload = {**self.applied_state(),
                   "schema": APPLIED_STATE_SCHEMA, "written_epoch": now,
                   "written_utc": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
                   "timing_authority_applied": self.timing_authority_applied()}
        try:
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def startup_report(self) -> list[str]:
        lines = [self._describe(w.plan) for w in self._workers]
        lines += [f"channel {n} refused: {why}" for n, why in self._refused.items()]
        if self._refused:
            lines.append(f"instance {self.cfg.key} records {len(self._workers)} of "
                         f"{len(self.cfg.channels)} channels")
        return lines

    # --- signals, run, stop -------------------------------------------------

    def _reload_log_level(self) -> None:
        level = None
        for path in self._env_files:
            try:
                for line in Path(path).read_text().splitlines():
                    k, sep, v = line.strip().partition("=")
                    if sep and k in ("CHIRP_RECORDER_LOG_LEVEL", "CLIENT_LOG_LEVEL"):
                        if k == "CHIRP_RECORDER_LOG_LEVEL" or level is None:
                            level = v.strip().strip("'\"").upper()
            except OSError:
                continue
        if level and isinstance(logging.getLevelName(level), int):
            logging.getLogger().setLevel(level)
            log.info("log level now %s", level)

    def on_sighup(self) -> None:
        """Safe to call from a signal handler: it only sets a flag."""
        self._hup.set()

    def _service_hup(self) -> None:
        if self._hup.is_set():
            self._hup.clear()
            self._reload_log_level()
            self.readmit()

    def run(self) -> int:
        signal.signal(signal.SIGTERM, lambda *_: self._stop_evt.set())
        signal.signal(signal.SIGINT, lambda *_: self._stop_evt.set())
        signal.signal(signal.SIGHUP, lambda *_: self.on_sighup())
        try:
            self.start_workers()
            if not self._workers:
                return 1
            while not self._stop_evt.wait(TICK_S):
                self._service_hup()
                self.tick(self.clock())
        finally:
            self.stop()
        return 0

    def stop(self) -> None:
        self._stop_evt.set()
        for w in self._workers:
            w.stop()
        for w in self._workers:
            w.join(timeout=10)
            if w.is_alive():
                log.warning("channel %s did not stop within 10 s", w.plan.name)
        if self._writer is not None:
            self._writer.close()
            self._writer = None
