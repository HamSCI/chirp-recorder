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


def test_full_queue_keeps_the_jsonl_and_loses_only_the_sink_copy(tmp_path):
    gate = []

    class Slow(FakeW):
        def insert(self, rows):
            while not gate:
                time.sleep(0.01)
            super().insert(rows)

    a = Slow()
    w = RowWriter(output_dir=tmp_path, sink_writers=(a, Slow(), Slow()),
                  phase_store=None, queue_max=1)
    cap, arrs = _rows()
    for _ in range(10):
        w.put(cap, arrs, [])
    n_full = w.stats["queue_full"]
    assert n_full >= 1
    gate.append(1)
    w.close()
    lines = (tmp_path / "rows" / "2026-11-19.jsonl").read_text().splitlines()
    assert len(lines) == 20 and len(a.rows) == 10 - n_full


def test_a_jsonl_failure_never_skips_the_sink(tmp_path):
    import numpy as np
    a, c, s = FakeW(), FakeW(), FakeW()
    w = RowWriter(output_dir=tmp_path, sink_writers=(a, c, s), phase_store=None)
    cap, arrs = _rows()
    bad = dict(arrs[0], engineering={"x": object()})
    npy = dict(arrs[0], delay_sigma_ns=np.float32(2.5))
    w.put(cap, [bad, npy], [])
    w.close()
    st = w.stats
    assert st["jsonl_failures"] == 1 and st["written"] == 2     # npy row + capture
    assert len(a.rows) == 2 and len(c.rows) == 1


@pytest.mark.parametrize("how", ["none", "raises"])
def test_a_spots_failure_keeps_the_arrival_and_capture_sinks(tmp_path, monkeypatch, caplog, how):
    monkeypatch.setenv("HAMSCI_SINK_PATH", str(tmp_path / "sink.db"))

    def bad():
        if how == "raises":
            raise RuntimeError("spots table broken")
        return None
    monkeypatch.setattr(wmod, "make_spots_writer", bad)
    with caplog.at_level("WARNING"):
        sinks = make_sink_writers()
    assert sinks is not None and sinks[2] is None
    assert [w.table for w in sinks[:2]] == ["arrival", "capture"]
    assert any("codar.spots" in r.getMessage() for r in caplog.records)
    for w in sinks[:2]:
        w.close()


def test_a_failure_after_the_first_writer_closes_it(monkeypatch):
    import hs_uploader.sink as sink
    built = []

    def from_env(**kw):
        if kw["table"] == "capture":
            raise RuntimeError("no capture table")
        built.append(FakeW())
        return built[-1]
    monkeypatch.setattr(sink.Writer, "from_env", staticmethod(from_env))
    assert make_sink_writers() is None
    assert built and all(w.closed for w in built)


def test_rowwriter_skips_spots_when_its_spots_writer_is_none(tmp_path):
    arr, cap = FakeW(), FakeW()
    w = RowWriter(output_dir=tmp_path, sink_writers=(arr, cap, None), phase_store=None)
    c, a = _rows()
    w.put(c, a, [], spots=[{"time": "x"}, {"time": "y"}])
    w.close()
    assert arr.rows and cap.rows
    assert w.stats["spots_skipped"] == 2 and w.stats["sink_failures"] == 0


def test_rows_jsonl_keeps_30_days_by_the_rows_own_day(tmp_path):
    """compare reads this copy, so it keeps a stated window: files more than
    ROWS_KEEP_DAYS before the day being written go at that day's first write.
    The day comes from the rows, never the host clock."""
    from chirp_recorder.core.writer import ROWS_KEEP_DAYS
    assert ROWS_KEEP_DAYS == 30
    d = tmp_path / "rows"
    d.mkdir()
    for day in ("2026-10-19", "2026-10-20", "2026-11-18"):
        (d / f"{day}.jsonl").write_text("{}\n")
    (d / "notes.txt").write_text("kept\n")
    w = RowWriter(output_dir=tmp_path, sink_writers=None, phase_store=None)
    cap, arrs = _rows()                                   # rows dated 2026-11-19
    w.put(cap, arrs, [])
    w.close()
    assert sorted(p.name for p in d.iterdir()) == [
        "2026-10-20.jsonl", "2026-11-18.jsonl", "2026-11-19.jsonl", "notes.txt"]
