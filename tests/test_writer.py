import json
import sqlite3
import time
from datetime import datetime, timezone

import pytest
from hamsci_dsp import sigmondb_rows as sr

from chirp_recorder.core import writer as wmod
from chirp_recorder.core.writer import PhaseStore, RowWriter, make_sink_writers


def _rows(t="2026-11-19T01:02:03Z"):
    cap = sr.capture_row(time=t[:-1] + ".000Z", reporter_id="X", centre_hz=4.463e6,
                         rate_hz_per_s=-25733.9, duration_s=30.0, detector_version="v")
    arr = sr.arrival_row(time=t, reporter_id="X", centre_hz=4.463e6, delay_ns=5,
                         detector_version="v", engineering={"a": "1"})
    return cap, [arr]


class FakeW:
    def __init__(self, fail=False):
        self.rows, self.fail, self.closed = [], fail, False

    def insert(self, rows):
        if self.fail:
            raise RuntimeError("sink down")
        self.rows.extend(rows)

    def close(self):
        self.closed = True


def test_rows_reach_a_real_sink_and_the_jsonl(tmp_path, monkeypatch):
    db = tmp_path / "sink.db"
    monkeypatch.setenv("HAMSCI_SINK_PATH", str(db))
    sinks = make_sink_writers()
    assert sinks is not None and len(sinks) == 3
    assert [w.table for w in sinks] == ["arrival", "capture", "spots"]
    assert all(w.producer == "chirp-recorder" for w in sinks)
    w = RowWriter(output_dir=tmp_path / "out", sink_writers=sinks, phase_store=None)
    cap, arrs = _rows()
    spot = {"time": "2026-11-19T01:02:03Z", "reporter_id": "X"}
    w.put(cap, arrs, [], spots=[spot])
    w.close()
    con = sqlite3.connect(db)
    names = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    assert "pending_uploads" in names
    rows = con.execute("select * from pending_uploads").fetchall()
    assert len(rows) == 3
    lines = (tmp_path / "out" / "rows" / "2026-11-19.jsonl").read_text().splitlines()
    assert len(lines) == 2 and w.stats["written"] == 2 and w.stats["sink_failures"] == 0
    assert any("spots" in json.dumps(r) for r in rows)


def test_spots_land_through_the_third_writer(tmp_path):
    a, c, s = FakeW(), FakeW(), FakeW()
    w = RowWriter(output_dir=tmp_path, sink_writers=(a, c, s), phase_store=None)
    w.put(None, [], [], spots=[{"x": 1}])
    w.close()
    assert s.rows == [{"x": 1}] and a.rows == [] and c.rows == []
    assert not (tmp_path / "rows").exists() or not list((tmp_path / "rows").glob("*.jsonl"))


def test_sink_failure_counts_and_jsonl_survives(tmp_path):
    a, c, s = FakeW(fail=True), FakeW(fail=True), FakeW()
    w = RowWriter(output_dir=tmp_path, sink_writers=(a, c, s), phase_store=None)
    cap, arrs = _rows()
    w.put(cap, arrs, [])
    w.close()
    assert w.stats["sink_failures"] == 2 and w.stats["written"] == 2
    lines = (tmp_path / "rows" / "2026-11-19.jsonl").read_text().splitlines()
    assert len(lines) == 2


def test_queue_full_counts_and_never_blocks(tmp_path, monkeypatch):
    gate = []

    class Slow(FakeW):
        def insert(self, rows):
            while not gate:
                time.sleep(0.01)

    w = RowWriter(output_dir=tmp_path, sink_writers=(Slow(), Slow(), Slow()),
                  phase_store=None, queue_max=2)
    cap, arrs = _rows()
    t0 = time.monotonic()
    for _ in range(20):
        w.put(cap, arrs, [])
    assert time.monotonic() - t0 < 2.0
    assert w.stats["queue_full"] >= 1
    gate.append(1)
    w.close()


def test_close_flushes_the_queue(tmp_path):
    a, c, s = FakeW(), FakeW(), FakeW()
    w = RowWriter(output_dir=tmp_path, sink_writers=(a, c, s), phase_store=None)
    cap, arrs = _rows()
    for _ in range(50):
        w.put(cap, arrs, [])
    w.close()
    assert len(a.rows) == 50 and len(c.rows) == 50
    assert a.closed and c.closed and s.closed
    assert w.stats["written"] == 100


def test_phases_go_to_the_phase_store(tmp_path):
    import numpy as np
    store = PhaseStore(tmp_path / "phase")
    w = RowWriter(output_dir=tmp_path, sink_writers=None, phase_store=store)
    t = datetime(2026, 11, 19, 1, 0, tzinfo=timezone.utc)
    w.put(None, [], [dict(group="g", cpi_start_utc=t, radiod="r", sweep_start_ns=1, bin=3,
                          period_s=1.0, tau_ns=5, delay_ns=5, level_db=10.0,
                          slow=np.ones(4, complex))])
    w.close()
    assert len(list((tmp_path / "phase").rglob("*.jsonl"))) == 1


def test_phase_store_prunes_days_past_its_keep(tmp_path):
    store = PhaseStore(tmp_path, radiod="r", keep_days=2)
    old = tmp_path / "r" / "g" / "2026-11-01.jsonl"
    old.parent.mkdir(parents=True)
    old.write_text("{}\n")
    store.prune(today=datetime(2026, 11, 12, tzinfo=timezone.utc).date())
    assert not old.exists()


def test_phase_store_series_stay_on_the_station(tmp_path):
    import numpy as np
    store = PhaseStore(tmp_path, keep_days=7)
    t = datetime(2026, 11, 19, 1, 0, tzinfo=timezone.utc)
    ref = store.write(group="4.4630 -25.73", cpi_start_utc=t, radiod="r", sweep_start_ns=1,
                      bin=3, period_s=1.0, tau_ns=5, delay_ns=5, level_db=10.0,
                      slow=np.exp(1j * np.arange(32)))
    f = tmp_path / ref
    rec = json.loads(f.read_text().splitlines()[0])
    assert len(rec["phase_rad"]) == 32 and len(rec["amp_db"]) == 32 and rec["delay_ns"] == 5
