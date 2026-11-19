"""Daemon tests.  Sources are fakes; detection is stubbed so the tests run in
milliseconds.  Every coordinate is made up."""
import json
import logging
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from chirp_recorder import __version__
from chirp_recorder.config import load_instance
from chirp_recorder.core import daemon as dm
from chirp_recorder.core.capacity import Headroom
from chirp_recorder.core.detect import GroupResult
from chirp_recorder.core.source import Interval, StatusPair
from chirp_recorder.core.daemon import ChannelWorker, Daemon, applied_state_path

GiB = 1 << 30
calm = lambda: Headroom(8, 0.5, 0.0, 0.0, 16 * GiB, 12 * GiB, None)   # noqa: E731
COSTS = {__version__: {64_000: (0.5, 256 << 20), 96_000: (0.7, 300 << 20)}}   # pre-flight C21

TWO = """
[instance]
reporter_id = "AC0G/T15"
[station]
callsign = "AC0G"
grid_square = "EM38ww"
receiver_lat = 38.0
receiver_lon = -95.0
[paths]
output_dir = "{out}"
[processing]
interval_s = 30
[[channel]]
name = "c4463"
status = "omni-status.local"
centre_hz = 4463000
sample_rate_hz = 64000
transmitters = ["AGL1"]
[[channel]]
name = "c13475"
status = "omni-status.local"
centre_hz = 13475000
sample_rate_hz = 64000
transmitters = ["ACI1"]
"""


def make_cfg(tmp_path, extra=""):
    (tmp_path / "AC0G-T15.toml").write_text((TWO + extra).format(out=tmp_path / "out"))
    return load_instance("AC0G-T15", tmp_path)


@pytest.fixture
def two_channel_cfg(tmp_path):
    return make_cfg(tmp_path)


def with_capacity_measure(tmp_path):
    return make_cfg(tmp_path, "\n[capacity]\nmeasure = true\n")


def _iv(i, channel):
    return Interval(samples=np.zeros(64, dtype=np.complex64), first_rtp=i * 64,
                    first_input_index=float(i * 1000), input_samprate=129_600_000,
                    input_epoch_gps_s=1.7e9, channel=channel,
                    status_pair=StatusPair(1_400_000_000_000_000_000, 0, 64_000))


class FakeSources:
    """source_factory stand-ins.  run_for(n) waits until some worker has
    delivered n intervals; now is the clock afterwards."""

    def __init__(self):
        self.created: list = []
        self.workers = lambda: []
        self.now = time.monotonic()

    def _src(self, plan, n, fail):
        outer = self

        class Src:
            stats = {}

            def __init__(self):
                self.stop_evt = threading.Event()
                outer.created.append(plan.name)

            def intervals(self):
                if fail:
                    raise RuntimeError("radiod refused the channel")
                for i in range(n):
                    yield _iv(i, plan.name)
                self.stop_evt.wait()

            def stop(self):
                self.stop_evt.set()
        return Src()

    def ok(self, plan, *, interval_s):
        return self._src(plan, 5, False)

    def silent(self, plan, *, interval_s):
        return self._src(plan, 0, False)

    def fail_on(self, name):
        return lambda plan, *, interval_s: self._src(plan, 5, plan.name == name)

    def run_for(self, n, who=None):
        end = time.monotonic() + 5
        while time.monotonic() < end:
            ws = self.workers()
            if ws and all(w.stats["delivered"] >= n for w in ws if who is None or w.plan.name in who):
                break
            time.sleep(0.01)
        self.now = time.monotonic()


@pytest.fixture
def fake_sources(monkeypatch):
    def fake_detect(samples, *, fs, channel_centre_hz, group):
        return GroupResult(group, 3, -40.0, -45.0, None, [], {})
    monkeypatch.setattr(dm, "detect_group", fake_detect)
    return FakeSources()


@pytest.fixture
def made():
    out = []
    yield out
    for d in out:
        d.stop()


def build(made, fs, cfg, **kw):
    kw.setdefault("headroom_reader", calm)
    kw.setdefault("costs", COSTS)
    kw.setdefault("restart_delay_s", 0.05)
    kw.setdefault("sink_writers", ())
    d = Daemon(cfg, **kw)
    fs.workers = lambda: d.workers
    made.append(d)
    return d


def test_every_configured_channel_runs(two_channel_cfg, fake_sources, made):
    d = build(made, fake_sources, two_channel_cfg, source_factory=fake_sources.ok)
    d.start_workers()
    assert {w.plan.name for w in d.workers} == {"c4463", "c13475"}


def test_one_failing_channel_leaves_the_other(two_channel_cfg, fake_sources, made):
    d = build(made, fake_sources, two_channel_cfg, source_factory=fake_sources.fail_on("c4463"))
    d.start_workers()
    fake_sources.run_for(3, who={"c13475"})
    assert d.worker("c13475").stats["delivered"] >= 3
    assert d.worker("c4463").stats["failures"] >= 1
    assert d.worker("c4463").is_alive()          # it restarts its source


def test_failed_source_is_restarted(two_channel_cfg, fake_sources, made):
    d = build(made, fake_sources, two_channel_cfg, source_factory=fake_sources.fail_on("c4463"))
    d.start_workers()
    end = time.monotonic() + 3
    while d.worker("c4463").stats["restarts"] < 2 and time.monotonic() < end:
        time.sleep(0.01)
    assert d.worker("c4463").stats["restarts"] >= 2


def test_short_admission_says_so(two_channel_cfg, fake_sources, made, caplog):
    caplog.set_level(logging.INFO)
    d = build(made, fake_sources, two_channel_cfg, costs={}, source_factory=fake_sources.ok)
    d.start_workers()
    assert "records 1 of 2 channels" in caplog.text
    errs = [r for r in caplog.records if r.levelno == logging.ERROR]
    warns = [r for r in caplog.records if r.levelno == logging.WARNING
             and "c13475 refused" in r.getMessage()]
    assert len(errs) == 1 and len(warns) == 1
    assert any("c4463" in line and "64000" in line for line in d.startup_report())


def test_watchdog_waits_for_a_delivered_interval(two_channel_cfg, fake_sources, made):
    notes = []
    d = build(made, fake_sources, two_channel_cfg, source_factory=fake_sources.silent,
              notify=notes.append)
    d.tick(now=200.0)
    assert "WATCHDOG=1" not in notes
    d.start_workers()                       # workers exist, none delivers
    time.sleep(0.1)
    d.tick(now=time.monotonic())
    assert "WATCHDOG=1" not in notes


def test_watchdog_pets_when_a_worker_delivers(two_channel_cfg, fake_sources, made):
    notes = []
    d = build(made, fake_sources, two_channel_cfg, source_factory=fake_sources.ok,
              notify=notes.append)
    d.start_workers()
    fake_sources.run_for(1)
    d.tick(now=fake_sources.now)
    assert "WATCHDOG=1" in notes
    notes.clear()
    d.tick(now=fake_sources.now + 3 * 30 + 1)       # delivery now older than 3 x interval_s
    assert "WATCHDOG=1" not in notes


def test_ready_once_workers_start(two_channel_cfg, fake_sources, made):
    notes = []
    d = build(made, fake_sources, two_channel_cfg, source_factory=fake_sources.ok,
              notify=notes.append)
    assert "READY=1" not in notes
    d.start_workers()
    assert notes.count("READY=1") == 1


def test_no_ready_with_no_worker(two_channel_cfg, fake_sources, made):
    notes = []
    hot = lambda: Headroom(8, 0.5, 50.0, 0.0, 16 * GiB, 12 * GiB, None)   # noqa: E731
    d = build(made, fake_sources, two_channel_cfg, headroom_reader=hot,
              source_factory=fake_sources.ok, notify=notes.append)
    d.start_workers()
    assert d.workers == [] and "READY=1" not in notes


def test_measuring_needs_both_switches(tmp_path, fake_sources, made, caplog):
    caplog.set_level(logging.INFO)
    cfg = make_cfg(tmp_path)
    d = build(made, fake_sources, cfg, costs={}, source_factory=fake_sources.ok, measure=True)
    d.start_workers()
    assert len(d.workers) == 1                       # the flag alone changes nothing
    assert "measuring cost" not in caplog.text
    (tmp_path / "m").mkdir()
    cfg2 = with_capacity_measure(tmp_path / "m")
    d2 = build(made, fake_sources, cfg2, costs={}, source_factory=fake_sources.ok, measure=True)
    d2.start_workers()
    assert len(d2.workers) == 2
    warns = [r for r in caplog.records if r.levelno == logging.WARNING
             and "measuring cost: admission bypassed" in r.getMessage()]
    assert len(warns) == 1


def test_config_switch_alone_changes_nothing(tmp_path, fake_sources, made):
    cfg = with_capacity_measure(tmp_path)
    d = build(made, fake_sources, cfg, costs={}, source_factory=fake_sources.ok)
    d.start_workers()
    assert len(d.workers) == 1


def test_worker_writes_rows_with_the_interval_fields(two_channel_cfg, fake_sources, made):
    d = build(made, fake_sources, two_channel_cfg, source_factory=fake_sources.ok)
    d.start_workers()
    fake_sources.run_for(2)
    d.stop()
    files = list((two_channel_cfg.output_dir / "rows").glob("*.jsonl"))
    assert files
    rows = [json.loads(line) for f in files for line in f.read_text().splitlines()]
    cap = [r["row"] if "row" in r else r for r in rows]
    blob = json.dumps(cap)
    assert "input_epoch_gps_s" in blob and "interval_s" in blob and "precise" in blob


def test_physics_spots_go_through_the_row_writer():
    puts = []
    writer = SimpleNamespace(put=lambda *a, **k: puts.append((a, k)))
    spot = {"table": "spots"}
    phys = SimpleNamespace(feed=lambda iv, label: SimpleNamespace(spots=[spot]),
                           close=lambda: None)
    plan = SimpleNamespace(name="c", status="s", centre_hz=4.463e6, sample_rate_hz=64000,
                           groups=())
    label = SimpleNamespace(delay_basis="radiod")
    w = ChannelWorker(plan, source_factory=None, tie=SimpleNamespace(label=lambda iv: label),
                      writer=writer, interval_s=30, reporter_id="X", version="1", physics=phys)
    w._handle(_iv(0, "c"))
    assert any(k.get("spots") == [spot] and a == (None, [], []) for a, k in puts)
    assert w.stats["spots"] == 1


def test_a_physics_failure_leaves_detection(fake_sources):
    writer = SimpleNamespace(put=lambda *a, **k: None)

    def boom(iv, label):
        raise RuntimeError("x")
    phys = SimpleNamespace(feed=boom, close=lambda: None)
    plan = SimpleNamespace(name="c", status="s", centre_hz=4.463e6, sample_rate_hz=64000,
                           groups=())
    w = ChannelWorker(plan, source_factory=None,
                      tie=SimpleNamespace(label=lambda iv: SimpleNamespace(delay_basis="radiod")),
                      writer=writer, interval_s=30, reporter_id="X", version="1", physics=phys)
    w._handle(_iv(0, "c"))
    assert w.stats["delivered"] == 1


def test_applied_state_contents_and_rate_limit(two_channel_cfg, fake_sources, made, monkeypatch):
    d = build(made, fake_sources, two_channel_cfg, costs={}, source_factory=fake_sources.ok)
    d.start_workers()
    fake_sources.run_for(1, who={"c4463"})
    path = applied_state_path(two_channel_cfg.output_dir)
    assert path == two_channel_cfg.output_dir / "applied-state.json"
    t0 = fake_sources.now
    d.tick(now=t0)
    first = json.loads(path.read_text())
    chans = {c["name"]: c for c in first["channels"]}
    assert chans["c4463"]["admission"] == "admitted"
    assert chans["c4463"]["delay_basis"] == "radiod"
    assert chans["c13475"]["admission"] == "refused" and chans["c13475"]["refused_reason"]
    assert first["position_source"] == "precise"
    text = path.read_text()
    assert "38.0" not in text and "-95.0" not in text           # no coordinates
    writes = []
    real = dm.os.replace
    monkeypatch.setattr(dm.os, "replace", lambda a, b: (writes.append(b), real(a, b)))
    d.tick(now=t0 + 30)
    assert writes == []                                          # at most once a minute
    d.tick(now=t0 + 61)
    assert len(writes) == 1
    assert not list(path.parent.glob("*.tmp"))


def test_sighup_readmits_refused_and_sheds_none(two_channel_cfg, fake_sources, made):
    state = {"costs": {}}
    d = build(made, fake_sources, two_channel_cfg, costs=state["costs"],
              source_factory=fake_sources.ok)
    d.start_workers()
    first = d.worker("c4463")
    assert len(d.workers) == 1
    d._costs.update(COSTS)                    # a measured row arrives
    d.on_sighup()
    d._service_hup()
    assert {w.plan.name for w in d.workers} == {"c4463", "c13475"}
    assert d.worker("c4463") is first and first.is_alive()
    # now the machine is hot: nothing running is shed
    d.headroom_reader = lambda: Headroom(8, 0.5, 90.0, 0.0, 16 * GiB, 12 * GiB, None)
    d.on_sighup()
    d._service_hup()
    assert len(d.workers) == 2


def test_sighup_reads_the_log_level(two_channel_cfg, fake_sources, made, tmp_path):
    env = tmp_path / "x.env"
    env.write_text("CLIENT_LOG_LEVEL=ERROR\nCHIRP_RECORDER_LOG_LEVEL=debug\n")
    d = build(made, fake_sources, two_channel_cfg, source_factory=fake_sources.ok,
              env_files=[env])
    root = logging.getLogger()
    old = root.level
    try:
        d._reload_log_level()
        assert root.level == logging.DEBUG
    finally:
        root.setLevel(old)


def test_stop_joins_workers_and_closes_the_writer(two_channel_cfg, fake_sources, made):
    d = build(made, fake_sources, two_channel_cfg, source_factory=fake_sources.ok)
    d.start_workers()
    ws, wr = d.workers, d._writer
    d.stop()
    assert not any(w.is_alive() for w in ws) and not wr.is_alive()
