"""The scout runner: fine dwells, the coarse window, and every stream closed.

Fake streams deliver synthetic frames synchronously and step a fake clock,
so no test touches radiod or the network.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from chirp_recorder.scout.plan import Dwell
from chirp_recorder.scout.published import Published
from chirp_recorder.scout.runner import Scout, make_scout_writer
from chirp_recorder.scout.settings import ScoutSettings

EMPTY = Published((), (), None, "none")
GPS0_NS = 1_795_000_000 * 10**9


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def sweep_frame(rng, *, f0_hz, bin_hz, n_bins, t_s, frame_s, rate, lo, hi, period,
                level_db=20.0):
    """Noise at -100 dB plus every bin one sawtooth sweep crossed since the
    previous frame (radiod integrates between frames)."""
    p = -100.0 + rng.normal(0.0, 1.0, n_bins)
    span = hi - lo
    hit = np.zeros(n_bins, bool)
    for a in np.linspace(t_s - frame_s, t_s, 64):
        ph = (a % period) * abs(rate)
        if ph >= span:
            continue
        f = lo + ph if rate > 0 else hi - ph
        k = int((f - f0_hz) / bin_hz)
        if 0 <= k < n_bins:
            hit[k] = True
    p[hit] = -100.0 + level_db
    return p


class World:
    """Holds the fake streams and decides what each delivers."""

    def __init__(self, settings, clock, script):
        self.s, self.clock, self.script = settings, clock, script
        self.streams: list = []
        self.dwells = 0

    def control(self, status):
        return SimpleNamespace(status=status, close=lambda: None)

    def stream(self, control, frequency_hz, bin_count, resolution_bw, *,
               poll_interval_sec, on_spectrum):
        s = FakeStream(self, frequency_hz, bin_count, resolution_bw, on_spectrum)
        self.streams.append(s)
        return s

    def fine(self):
        return [s for s in self.streams if s.view == "fine"]

    def coarse(self):
        return [s for s in self.streams if s.view == "coarse"]


class FakeStream:
    def __init__(self, world, freq, bin_count, rbw, cb):
        self.w, self.freq, self.bin_count, self.rbw, self.cb = world, freq, bin_count, rbw, cb
        coarse = bin_count == round(world.s.nyquist_hz / world.s.coarse_bin_hz)
        self.view = "coarse" if coarse else "fine"
        self.tuned: list = []
        self.stopped = 0
        self.dead = False

    def _deliver(self):
        self.tuned.append(self.freq)
        frames = [] if self.dead else self.w.script(self, self.freq)
        for dt, power, gps_ns in frames:
            self.w.clock.t += dt
            self.cb(SimpleNamespace(spectrum=SimpleNamespace(bin_power_db=power),
                                    gps_time=gps_ns, frequency=self.freq))
        if self.view == "fine":
            self.w.clock.t += self.w.s.dwell_s + 1.0        # the dwell's time passes

    def start(self):
        self._deliver()

    def set_frequency(self, hz):
        self.freq = hz
        self._deliver()

    def stop(self):
        self.stopped += 1


def fine_settings(**kw):
    return ScoutSettings(enabled=True, status="x-status.local", **kw)


def advance(clock):
    """A wait that moves the fake clock instead of sleeping."""
    def wait(ev, s):
        clock.t += s
    return wait


def make_scout(settings, world, clock, rows=None, **kw):
    rows = [] if rows is None else rows
    kw.setdefault("wait", advance(clock))
    return Scout(settings, "AC0G/T7", control_factory=world.control,
                 stream_factory=world.stream,
                 writer=lambda table, row: rows.append((table, row)),
                 now=lambda: 1.795e9, monotonic=clock,
                 utcnow=lambda: datetime(2026, 11, 28, 12, 0, tzinfo=timezone.utc),
                 published_loader=lambda: EMPTY, **kw)


def codar_frames(n, *, f0_hz, bin_hz, n_bins, frame_s=0.15, seed=1):
    """n frames of a CODAR-like sweep (25.73 kHz/s down, 1 s period,
    4.450-4.4757 MHz) with gps_time stamps; spacing jitters by 22 ms sd,
    as SIGMA's packets do."""
    rng = np.random.default_rng(seed)
    out, t = [], 0.0
    for _ in range(n):
        dt = float(np.clip(rng.normal(frame_s, 0.022), 0.08, 0.25))
        t += dt
        out.append((dt, sweep_frame(rng, f0_hz=f0_hz, bin_hz=bin_hz, n_bins=n_bins,
                                    t_s=t, frame_s=dt, rate=-25730.0,
                                    lo=4.450e6, hi=4.47573e6, period=1.0),
                    GPS0_NS + int(round(t * 1e9))))
    return out


def test_dwell_writes_sweeps_and_one_dwell_row():
    s = fine_settings()
    clock = Clock()
    n_bins = int(s.fine_span_hz / s.fine_bin_hz)
    d = Dwell("fine", 4.86e6, s.fine_span_hz, s.fine_bin_hz, "band")
    f0 = d.centre_hz - n_bins * s.fine_bin_hz / 2
    frames = codar_frames(280, f0_hz=f0, bin_hz=s.fine_bin_hz, n_bins=n_bins)
    world = World(s, clock, lambda st, f: frames)
    scout = make_scout(s, world, clock)
    sweeps, dwell = scout.run_dwell(d)
    offsets = np.cumsum([dt for dt, _, _ in frames])
    assert offsets[-1] < s.dwell_s
    after_settle = int(np.sum(offsets >= 0.5))
    assert dwell is not None and dwell["n_frames"] == after_settle
    assert dwell["view"] == "fine" and dwell["n_streaks"] == 1
    assert dwell["noise_db"] == pytest.approx(-100.0, abs=0.3)
    assert dwell["frame_s"] == pytest.approx(0.15, rel=0.02)
    assert dwell["engineering"]["time_source"] == "gps_time"
    assert len(sweeps) == 1
    r = sweeps[0]
    assert r["rate_hz_per_s"] == pytest.approx(-25730.0, rel=0.01)
    assert r["period_s"] == pytest.approx(1.0, rel=0.01)
    assert r["centre_hz"] == pytest.approx(4.4629e6, abs=2000)
    assert r["matched_key"] == "" and r["engineering"]["time_source"] == "gps_time"
    json.dumps(r)
    json.dumps(dwell)


def test_dwell_without_gps_time_uses_monotonic_and_survives_nan():
    s = fine_settings()
    clock = Clock()
    n_bins = int(s.fine_span_hz / s.fine_bin_hz)
    d = Dwell("fine", 4.86e6, s.fine_span_hz, s.fine_bin_hz, "band")
    f0 = d.centre_hz - n_bins * s.fine_bin_hz / 2
    frames = [(dt, p, None) for dt, p, _ in
              codar_frames(280, f0_hz=f0, bin_hz=s.fine_bin_hz, n_bins=n_bins)]
    frames[10][1][5] = -np.inf
    frames[11][1][:] = np.nan            # no finite bin: skipped and counted
    world = World(s, clock, lambda st, f: frames)
    sweeps, dwell = make_scout(s, world, clock).run_dwell(d)
    assert dwell["engineering"]["time_source"] == "monotonic"
    assert dwell["engineering"]["frames_skipped"] == "1"
    assert len(sweeps) == 1


def test_starved_dwell_writes_nothing(caplog):
    s = fine_settings()
    clock = Clock()
    n_bins = int(s.fine_span_hz / s.fine_bin_hz)
    d = Dwell("fine", 4.86e6, s.fine_span_hz, s.fine_bin_hz, "band")
    f0 = d.centre_hz - n_bins * s.fine_bin_hz / 2
    frames = codar_frames(40, f0_hz=f0, bin_hz=s.fine_bin_hz, n_bins=n_bins)
    world = World(s, clock, lambda st, f: frames)
    with caplog.at_level(logging.WARNING):
        sweeps, dwell = make_scout(s, world, clock).run_dwell(d)
    assert sweeps == [] and dwell is None
    assert "dwell starved" in caplog.text


def test_channel_reopens_after_radiod_restart():
    s = fine_settings(min_frames=20)
    clock = Clock()
    n_bins = int(s.fine_span_hz / s.fine_bin_hz)
    noise = np.full(n_bins, -100.0)

    def script(st, f):
        return [(0.15, noise, None)] * 100

    world = World(s, clock, script)
    scout = make_scout(s, world, clock)
    d = Dwell("fine", 4.86e6, s.fine_span_hz, s.fine_bin_hz, "band")
    assert scout.run_dwell(d)[1] is not None
    first = world.fine()[0]
    first.dead = True                     # radiod restarted: frames stop
    assert scout.run_dwell(d)[1] is None
    assert len(world.fine()) == 1
    assert scout.run_dwell(d)[1] is not None
    assert first.stopped >= 1 and len(world.fine()) == 2
    scout.close()
    assert all(x.stopped for x in world.streams)


def test_fine_stream_opens_once_and_retunes():
    s = fine_settings(min_frames=20)
    clock = Clock()
    n_bins = int(s.fine_span_hz / s.fine_bin_hz)
    world = World(s, clock, lambda st, f: [(0.15, np.full(n_bins, -100.0), None)] * 100)
    scout = make_scout(s, world, clock)
    for c in (4.86e6, 9.33e6):
        scout.run_dwell(Dwell("fine", c, s.fine_span_hz, s.fine_bin_hz, "band"))
    assert len(world.fine()) == 1 and world.fine()[0].tuned == [4.86e6, 9.33e6]
    assert world.fine()[0].bin_count == n_bins and world.fine()[0].rbw == s.fine_bin_hz


def test_stop_closes_every_stream(monkeypatch):
    import chirp_recorder.scout.runner as runner
    monkeypatch.setattr(runner, "RETRY_S", 0.1)
    s = fine_settings(dwell_s=0.3, min_frames=1)
    world = World(s, Clock(), lambda st, f: [])
    raised = []

    def stream(control, frequency_hz, bin_count, resolution_bw, *, poll_interval_sec,
               on_spectrum):
        st = World.stream(world, control, frequency_hz, bin_count, resolution_bw,
                          poll_interval_sec=poll_interval_sec, on_spectrum=on_spectrum)
        if st.view == "fine" and not raised:
            raised.append(st)

            def boom():
                raise RuntimeError("radiod said no")
            st.start = boom
        else:
            st._deliver = lambda: None    # real clock: nothing arrives
        return st

    scout = Scout(s, "AC0G/T7", control_factory=world.control, stream_factory=stream,
                  writer=lambda t, r: None, now=time.time, monotonic=time.monotonic,
                  utcnow=lambda: datetime.now(timezone.utc), published_loader=lambda: EMPTY)
    stop = threading.Event()
    th = threading.Thread(target=scout.run, args=(stop,))
    th.start()
    time.sleep(1.0)
    t0 = time.monotonic()
    stop.set()
    th.join(5.0)
    assert not th.is_alive()
    assert time.monotonic() - t0 <= s.fine_poll_s + 0.3
    assert raised and len(world.coarse()) == 1 and len(world.fine()) >= 2
    assert all(x.stopped for x in world.streams)


def test_coarse_window_flags_the_fine_cycle():
    s = ScoutSettings(enabled=True, status="x", nyquist_hz=16.0e6, coarse_bin_hz=4000.0,
                      coarse_window_s=60.0, targets_mhz=(4.86, 9.33), min_frames=50)
    clock = Clock()
    n_coarse = int(s.nyquist_hz / s.coarse_bin_hz)
    rng = np.random.default_rng(3)

    def script(st, f):
        if st.view == "fine" or st is not world.coarse()[0]:
            return []                 # the coarse sweeps arrive once, on its first start
        frames = []
        for i in range(160):          # 64 s at 0.4 s
            t = (i + 1) * 0.4
            frames.append((0.4, sweep_frame(rng, f0_hz=0.0, bin_hz=s.coarse_bin_hz,
                                            n_bins=n_coarse, t_s=t, frame_s=0.4,
                                            rate=50_000.0, lo=7.1e6, hi=7.5e6,
                                            period=10.0),
                           GPS0_NS + int(t * 1e9)))
        return frames

    rows: list = []
    world = World(s, clock, script)
    scout = make_scout(s, world, clock, rows)
    scout.run(threading.Event(), max_dwells=4)
    assert world.coarse()[0].freq == pytest.approx(8.0e6)
    assert world.coarse()[0].bin_count == n_coarse
    # R19: a 400 kHz sweep is narrower than a fine window, so it only flags;
    # R18: the coarse view writes no dwell rows.
    assert not any(r["view"] == "coarse" for t, r in rows)
    tuned = [f for x in world.fine() for f in x.tuned]
    assert tuned[0] == pytest.approx(4.86e6)
    assert any(abs(f - 7.3e6) < 10e3 for f in tuned[1:])
    assert all(x.stopped for x in world.streams)


def test_coarse_streak_inside_a_target_window_is_not_flagged():
    s = ScoutSettings(enabled=True, status="x", nyquist_hz=16.0e6, coarse_bin_hz=4000.0,
                      coarse_window_s=60.0, targets_mhz=(4.86, 7.0), min_frames=50)
    clock = Clock()
    n_coarse = int(s.nyquist_hz / s.coarse_bin_hz)
    rng = np.random.default_rng(3)

    def script(st, f):
        if st.view == "fine":
            return []
        return [(0.4, sweep_frame(rng, f0_hz=0.0, bin_hz=s.coarse_bin_hz, n_bins=n_coarse,
                                  t_s=(i + 1) * 0.4, frame_s=0.4, rate=50_000.0,
                                  lo=7.1e6, hi=7.5e6, period=10.0), None)
                for i in range(160)]

    world = World(s, clock, script)
    scout = make_scout(s, world, clock)
    scout.run(threading.Event(), max_dwells=4)
    tuned = [f for x in world.fine() for f in x.tuned]
    assert not any(abs(f - 7.3e6) < 10e3 for f in tuned)


def test_scout_writer_keeps_daily_jsonl_and_prunes(tmp_path, monkeypatch):
    import chirp_recorder.scout.runner as runner
    monkeypatch.setattr(runner, "_sink_writers", lambda: None)
    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()
    (rows_dir / "2026-01-01.jsonl").write_text("{}\n")
    w = make_scout_writer(tmp_path, today=lambda: datetime(2026, 11, 28).date())
    w("sweep", {"time": "2026-11-28T12:00:00.000Z", "x": 1})
    w("scout_dwell", {"time": "2026-11-28T12:00:45.000Z", "y": 2})
    w.close()
    lines = (rows_dir / "2026-11-28.jsonl").read_text().splitlines()
    assert [json.loads(x)["table"] for x in lines] == ["sweep", "scout_dwell"]
    assert json.loads(lines[0])["row"]["x"] == 1
    assert not (rows_dir / "2026-01-01.jsonl").exists()


def test_scout_writer_sends_to_the_sinks(tmp_path, monkeypatch):
    import chirp_recorder.scout.runner as runner
    got = {"sweep": [], "scout_dwell": []}

    class W:
        def __init__(self, t):
            self.t = t
            self.closed = False

        def insert(self, rows):
            got[self.t].extend(rows)

        def flush(self):
            pass

        def close(self):
            self.closed = True

    sinks = {"sweep": W("sweep"), "scout_dwell": W("scout_dwell")}
    monkeypatch.setattr(runner, "_sink_writers", lambda: sinks)
    w = make_scout_writer(tmp_path)
    w("sweep", {"time": "2026-11-28T12:00:00.000Z"})
    w("scout_dwell", {"time": "2026-11-28T12:00:00.000Z"})
    w.close()
    assert len(got["sweep"]) == 1 and len(got["scout_dwell"]) == 1
    assert all(x.closed for x in sinks.values())


def test_a_failing_line_search_writes_no_rows(monkeypatch, caplog):
    """A crash in find_streaks (seen: a zero fitted rate on SIGMA's full
    waterfall): the dwell could not look, so it claims nothing (R18).  The
    channel was fine, so the next dwell keeps it."""
    import chirp_recorder.scout.runner as runner

    def boom(*a, **k):
        raise ZeroDivisionError("float division by zero")
    monkeypatch.setattr(runner, "find_streaks", boom)
    s = fine_settings(min_frames=20)
    clock = Clock()
    n_bins = int(s.fine_span_hz / s.fine_bin_hz)
    world = World(s, clock, lambda st, f: [(0.15, np.full(n_bins, -100.0), None)] * 100)
    scout = make_scout(s, world, clock)
    with caplog.at_level(logging.WARNING):
        sweeps, dwell = scout.run_dwell(
            Dwell("fine", 4.86e6, s.fine_span_hz, s.fine_bin_hz, "band"))
    assert sweeps == [] and dwell is None
    assert "line search failed" in caplog.text and "ZeroDivisionError" in caplog.text
    assert not scout._fine_stale


def test_coarse_window_closes_early_when_it_holds_too_many_dashes(caplog):
    s = ScoutSettings(enabled=True, status="x", nyquist_hz=16.0e6, coarse_bin_hz=4000.0,
                      coarse_window_s=3600.0, targets_mhz=(4.86, 9.33), min_frames=50)
    clock = Clock()
    n_coarse = int(s.nyquist_hz / s.coarse_bin_hz)
    rng = np.random.default_rng(3)

    def script(st, f):
        if st.view == "fine" or st is not world.coarse()[0]:
            return []
        return [(0.4, sweep_frame(rng, f0_hz=0.0, bin_hz=s.coarse_bin_hz, n_bins=n_coarse,
                                  t_s=(i + 1) * 0.4, frame_s=0.4, rate=50_000.0,
                                  lo=7.1e6, hi=7.5e6, period=10.0), None)
                for i in range(160)]

    rows: list = []
    world = World(s, clock, script)
    scout = make_scout(s, world, clock, rows, coarse_max_held=50)
    with caplog.at_level(logging.WARNING):
        scout.run(threading.Event(), max_dwells=1)
    assert "coarse window closed early at" in caplog.text
    # The early window ran its search: its narrow sweep flagged the fine
    # cycle (R19), and the coarse view wrote no rows (R18).
    assert scout.cycle._flags and abs(scout.cycle._flags[0][0] - 7.3e6) < 10e3
    assert not any(r["view"] == "coarse" for t, r in rows)
    assert scout._coarse_held() == 0          # the new window holds none of the old dashes


def test_coarse_window_runs_its_hour_below_the_threshold():
    s = ScoutSettings(enabled=True, status="x", nyquist_hz=16.0e6, coarse_bin_hz=4000.0,
                      coarse_window_s=3600.0, targets_mhz=(4.86, 9.33), min_frames=50)
    clock = Clock()
    rows: list = []
    world = World(s, clock, lambda st, f: [])
    make_scout(s, world, clock, rows).run(threading.Event(), max_dwells=2)
    assert not any(t == "scout_dwell" and r["view"] == "coarse" for t, r in rows)


def test_a_failed_coarse_reopen_does_not_end_the_run():
    """The coarse stream goes silent; its reopen raises once; a later pass
    opens it, and run() carries on to its last dwell."""
    s = ScoutSettings(enabled=True, status="x", nyquist_hz=16.0e6, coarse_bin_hz=4000.0,
                      coarse_window_s=60.0, targets_mhz=(4.86, 9.33), min_frames=50)
    clock = Clock()
    world = World(s, clock, lambda st, f: [])
    built = []

    def stream(control, frequency_hz, bin_count, resolution_bw, *, poll_interval_sec,
               on_spectrum):
        st = World.stream(world, control, frequency_hz, bin_count, resolution_bw,
                          poll_interval_sec=poll_interval_sec, on_spectrum=on_spectrum)
        if st.view == "coarse":
            built.append(st)
            if len(built) == 2:
                def boom():
                    raise RuntimeError("radiod not back yet")
                st.start = boom
        return st

    scout = Scout(s, "AC0G/T7", control_factory=world.control, stream_factory=stream,
                  writer=lambda t, r: None, now=lambda: 1.795e9, monotonic=clock,
                  utcnow=lambda: datetime(2026, 11, 28, 12, 0, tzinfo=timezone.utc),
                  published_loader=lambda: EMPTY, wait=advance(clock))
    dwells = 6
    scout.run(threading.Event(), max_dwells=dwells)
    assert len(built) >= 3                          # silent, failed reopen, reopened
    assert sum(len(x.tuned) for x in world.fine()) == dwells
    assert all(x.stopped for x in world.streams)


def test_fine_centres_round_to_whole_bins_and_snapped_frequency_is_kept():
    """A flagged centre is any float; the runner tunes a whole number of fine
    bins, and frames reporting a frequency 1.5 bins off still count."""
    s = fine_settings(min_frames=20)
    clock = Clock()
    n_bins = int(s.fine_span_hz / s.fine_bin_hz)

    def script(st, f):
        return [(0.15, np.full(n_bins, -100.0), None)] * 100

    class Snapped(FakeStream):
        def _deliver(self):
            self.tuned.append(self.freq)
            for dt, power, gps in script(self, self.freq):
                self.w.clock.t += dt
                self.cb(SimpleNamespace(spectrum=SimpleNamespace(bin_power_db=power),
                                        gps_time=gps, frequency=self.freq + 1.5 * self.rbw))
            self.w.clock.t += self.w.s.dwell_s + 1.0

    world = World(s, clock, script)

    def stream(control, frequency_hz, bin_count, resolution_bw, *, poll_interval_sec,
               on_spectrum):
        st = Snapped(world, frequency_hz, bin_count, resolution_bw, on_spectrum)
        world.streams.append(st)
        return st

    scout = Scout(s, "AC0G/T7", control_factory=world.control, stream_factory=stream,
                  writer=lambda t, r: None, now=lambda: 1.795e9, monotonic=clock,
                  utcnow=lambda: datetime(2026, 11, 28, 12, 0, tzinfo=timezone.utc),
                  published_loader=lambda: EMPTY)
    sweeps, dwell = scout.run_dwell(Dwell("fine", 7_298_012.3, s.fine_span_hz,
                                          s.fine_bin_hz, "flag"))
    tuned = world.fine()[0].tuned[0]
    assert tuned / s.fine_bin_hz == round(tuned / s.fine_bin_hz)
    assert abs(tuned - 7_298_012.3) <= s.fine_bin_hz / 2
    assert dwell is not None and dwell["centre_hz"] == tuned
    assert dwell["engineering"]["frames_other_freq"] == "0"


# --- final-review fix wave (R18-R21) ------------------------------------------

def coarse_settings(**kw):
    base = dict(enabled=True, status="x", nyquist_hz=16.0e6, coarse_bin_hz=4000.0,
                coarse_window_s=60.0, targets_mhz=(4.86, 9.33), min_frames=50)
    base.update(kw)
    return ScoutSettings(**base)


def coarse_sweep_frames(s, n, *, lo, hi, rate, period, samprate=None, seed=3):
    rng = np.random.default_rng(seed)
    n_coarse = int(s.nyquist_hz / s.coarse_bin_hz)
    fe = None if samprate is None else SimpleNamespace(input_samprate=samprate)
    out = []
    for i in range(n):
        p = sweep_frame(rng, f0_hz=0.0, bin_hz=s.coarse_bin_hz, n_bins=n_coarse,
                        t_s=(i + 1) * 0.4, frame_s=0.4, rate=rate, lo=lo, hi=hi,
                        period=period)
        out.append((0.4, p, GPS0_NS + int((i + 1) * 0.4e9), fe))
    return out


class CoarseWorld(World):
    """Coarse frames, delivered once on the first coarse start; may carry
    a frontend with input_samprate."""

    def __init__(self, s, clock, frames):
        super().__init__(s, clock, None)
        self.frames = frames

    def stream(self, control, frequency_hz, bin_count, resolution_bw, *,
               poll_interval_sec, on_spectrum):
        st = super().stream(control, frequency_hz, bin_count, resolution_bw,
                            poll_interval_sec=poll_interval_sec, on_spectrum=on_spectrum)
        world = self

        def deliver():
            st.tuned.append(st.freq)
            if st.view == "coarse" and st is world.coarse()[0]:
                for dt, p, g, fe in world.frames:
                    world.clock.t += dt
                    ns = SimpleNamespace(spectrum=SimpleNamespace(bin_power_db=p),
                                         gps_time=g, frequency=st.freq)
                    if fe is not None:
                        ns.frontend = fe
                    st.cb(ns)
            if st.view == "fine":
                world.clock.t += world.s.dwell_s + 1.0
        st._deliver = deliver
        return st


def test_burst_heavy_dwell_under_min_frames_writes_nothing(caplog):
    """min_frames counts frames that could show a dash: bursts do not (R18)."""
    s = fine_settings(min_frames=50)
    clock = Clock()
    n_bins = int(s.fine_span_hz / s.fine_bin_hz)
    rng = np.random.default_rng(5)

    def frame(burst):
        p = -100.0 + rng.normal(0.0, 1.0, n_bins)
        if burst:                       # 150 two-bin runs: over the 100-dash cap
            k = rng.choice(n_bins // 4, 150, replace=False) * 4
            p[k] += 20.0
            p[k + 1] += 20.0
        return p

    frames = [(0.15, frame(i % 5 != 0), None) for i in range(100)]   # 80 % bursts
    world = World(s, clock, lambda st, f: frames)
    with caplog.at_level(logging.WARNING):
        sweeps, dwell = make_scout(s, world, clock).run_dwell(
            Dwell("fine", 4.86e6, s.fine_span_hz, s.fine_bin_hz, "band"))
    assert sweeps == [] and dwell is None
    assert "dwell starved" in caplog.text and "less bursts" in caplog.text


def test_dwell_row_names_its_view_limits():
    s = fine_settings(min_frames=20)
    clock = Clock()
    n_bins = int(s.fine_span_hz / s.fine_bin_hz)
    world = World(s, clock, lambda st, f: [(0.15, np.full(n_bins, -100.0), None)] * 100)
    _, dwell = make_scout(s, world, clock).run_dwell(
        Dwell("fine", 4.86e6, s.fine_span_hz, s.fine_bin_hz, "band"))
    e = dwell["engineering"]
    assert (e["rate_min_hz_s"], e["rate_max_hz_s"], e["min_span_hz"], e["bin_hz"]) == (
        "200.0", "200000.0", "10000.0", "150.0")


def test_wide_coarse_sweep_writes_a_sweep_row_and_no_dwell_row():
    """R19: a sweep at least a fine window wide is the coarse view's to
    report; R18: the coarse view writes no scout_dwell row."""
    s = coarse_settings()
    clock = Clock()
    frames = coarse_sweep_frames(s, 160, lo=11.0e6, hi=13.5e6, rate=2.5e5, period=12.0)
    rows: list = []
    world = CoarseWorld(s, clock, frames)
    scout = make_scout(s, world, clock, rows)
    scout.run(threading.Event(), max_dwells=2)
    coarse = [(t, r) for t, r in rows if r["view"] == "coarse"]
    assert [t for t, _ in coarse] == ["sweep"]
    assert coarse[0][1]["span_hz"] >= s.fine_span_hz
    assert not scout.cycle._flags


def test_coarse_window_below_half_its_frames_writes_nothing(caplog):
    s = coarse_settings()
    clock = Clock()
    # 100 frames (40 s) in a window of 86 s at 0.4 s polls: 215 promised, 108 needed
    frames = coarse_sweep_frames(s, 100, lo=11.0e6, hi=13.5e6, rate=1.0e6, period=5.0)
    rows: list = []
    world = CoarseWorld(s, clock, frames)
    with caplog.at_level(logging.WARNING):
        make_scout(s, world, clock, rows).run(threading.Event(), max_dwells=2)
    assert not any(r["view"] == "coarse" for t, r in rows)
    assert "dwell starved: coarse" in caplog.text and "min_frames 108" in caplog.text


def test_nyquist_mismatch_stops_the_scout():
    from chirp_recorder.scout.runner import ScoutConfigError
    s = coarse_settings()
    clock = Clock()
    frames = coarse_sweep_frames(s, 5, lo=11.0e6, hi=13.5e6, rate=1.0e6, period=5.0,
                                 samprate=129_600_000)
    world = CoarseWorld(s, clock, frames)
    with pytest.raises(ScoutConfigError, match="64800000"):
        make_scout(s, world, clock).run(threading.Event(), max_dwells=3)
    assert world.fine() == [] and all(x.stopped for x in world.streams)


def test_nyquist_matching_or_absent_runs(caplog):
    s = coarse_settings()
    for rate in (32_000_000, None):
        clock = Clock()
        frames = coarse_sweep_frames(s, 5, lo=11.0e6, hi=13.5e6, rate=1.0e6, period=5.0,
                                     samprate=rate)
        world = CoarseWorld(s, clock, frames)
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            make_scout(s, world, clock).run(threading.Event(), max_dwells=1)
        assert len(world.fine()) == 1
        assert ("no input sample rate" in caplog.text) == (rate is None)
        assert caplog.text.count("no input sample rate") <= 1


def test_cli_returns_one_on_a_config_contradiction(tmp_path, monkeypatch):
    import chirp_recorder.scout.runner as runner
    from chirp_recorder.cli import main
    from tests.test_scout_settings import cfg_for
    cfg_for(tmp_path, '\n[scout]\nenabled = true\n')

    class Contradicted:
        def __init__(self, *a, **k):
            pass

        def run(self, stop, max_dwells=None):
            raise runner.ScoutConfigError("[scout] nyquist_hz = 1 Hz, but radiod ...")

    monkeypatch.setattr(runner, "Scout", Contradicted)
    assert main(["--etc", str(tmp_path), "scout", "AC0G-T7", "--dry-run"]) == 1


def test_cli_scout_quiets_ka9q_spectrum_polls(tmp_path, monkeypatch):
    """ka9q.control logs each poll at INFO (~31 000 lines an hour on SIGMA);
    scout mode raises that logger to WARNING unless the run asks for DEBUG."""
    import chirp_recorder.scout.runner as runner
    from chirp_recorder.cli import main
    from tests.test_scout_settings import cfg_for
    cfg_for(tmp_path, '\n[scout]\nenabled = true\n')
    seen = []

    class Probe:
        def __init__(self, *a, **k):
            pass

        def run(self, stop, max_dwells=None):
            seen.append(logging.getLogger("ka9q.control").isEnabledFor(logging.INFO))

    monkeypatch.setattr(runner, "Scout", Probe)
    ka9q = logging.getLogger("ka9q.control")
    monkeypatch.setattr(ka9q, "level", logging.NOTSET)
    monkeypatch.setattr(logging.getLogger(), "level", logging.INFO)
    monkeypatch.delenv("CHIRP_RECORDER_LOG_LEVEL", raising=False)
    monkeypatch.delenv("CLIENT_LOG_LEVEL", raising=False)
    assert main(["--etc", str(tmp_path), "scout", "AC0G-T7", "--dry-run"]) == 0
    ka9q.setLevel(logging.NOTSET)
    logging.getLogger().setLevel(logging.DEBUG)
    assert main(["--etc", str(tmp_path), "scout", "AC0G-T7", "--dry-run"]) == 0
    assert seen == [False, True]


def test_coarse_flags_are_capped_and_deduplicated():
    from hamsci_dsp.chirp.streaks import Streak
    s = coarse_settings()
    clock = Clock()
    world = World(s, clock, lambda st, f: [])
    scout = make_scout(s, world, clock)

    def st(mhz):
        return (Streak(mhz * 1e6, 300e3, 5e4, 10.0, 6.0, 20.0, 5, 0.9, False, (0.0,)), "", {})
    # 7.3 and 7.8 MHz lie within half a fine span: one flag.  Then six more
    # apart from each other and from the targets: the cap stops at 5.
    passed = [st(m) for m in (7.3, 7.8, 1.2, 2.6, 11.0, 12.5, 14.0)]
    scout._coarse_acc = None

    class Acc:
        dwell = Dwell("coarse", 8e6, 16e6, 4000.0, "band")
        counts = {"frames_received": 1}
        n_frames = n_used = 200
    scout._coarse_acc = Acc()
    scout._rows = lambda acc, **kw: ([], {"engineering": {"streaks_found": "7"}}, passed, "ok")
    scout._close_coarse_window()
    centres = [c for c, _ in scout.cycle._flags]
    assert len(centres) == 5
    assert centres == pytest.approx([7.3e6, 1.2e6, 2.6e6, 11.0e6, 12.5e6])


def test_backoff_doubles_to_five_minutes_and_logs_each_step(caplog):
    from chirp_recorder.scout.runner import _Backoff
    clock = Clock(0.0)
    b = _Backoff(clock)
    assert b.due()
    delays = []
    with caplog.at_level(logging.INFO):
        for _ in range(9):
            b.failed(OSError("no route"))
            delays.append(b.delay)
            assert not b.due()
            clock.t = b.next_at
            assert b.due()
        b.ok()
        b.ok()
    assert delays == [5, 10, 20, 40, 80, 160, 300, 300, 300]
    assert caplog.text.count("radiod unreachable") == 9
    assert caplog.text.count("radiod reachable again") == 1


def test_radiod_unreachable_at_start_is_retried_with_backoff():
    s = coarse_settings()
    clock = Clock(0.0)
    world = World(s, clock, lambda st, f: [])
    tries = []

    def control(status):
        tries.append(clock.t)
        if len(tries) <= 3:
            raise OSError("radiod gone")
        return world.control(status)

    scout = Scout(s, "AC0G/T7", control_factory=control, stream_factory=world.stream,
                  writer=lambda t, r: None, now=lambda: 1.795e9, monotonic=clock,
                  utcnow=lambda: datetime(2026, 11, 28, 12, 0, tzinfo=timezone.utc),
                  published_loader=lambda: EMPTY, wait=advance(clock))
    scout.run(threading.Event(), max_dwells=1)
    assert tries == [0.0, 5.0, 15.0, 35.0]
    assert len(world.coarse()) == 1 and len(world.fine()) == 1
