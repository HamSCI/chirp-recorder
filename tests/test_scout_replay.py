"""scout-replay on the SIGMA waterfall, and the scout verb when disabled.

The fixture holds 410 frames of SIGMA's fine channel cropped to 4.42-4.52
MHz (scripts/crop_scout_fixture.py makes it); frame times are packet
arrivals.  It carries no position.
"""
from __future__ import annotations

import json
from pathlib import Path

from chirp_recorder.cli import main
from tests.test_scout_settings import cfg_for

FIXTURE = Path(__file__).parent / "data" / "scout_sigma_4460k.npz"
CODAR_RATE_HZ_S = 25_730.0


def test_fixture_is_small():
    assert FIXTURE.stat().st_size < 1_000_000


def test_replay_finds_the_sigma_codar(capsys):
    assert main(["scout-replay", str(FIXTURE)]) == 0
    streaks = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    hits = [s for s in streaks
            if abs(abs(s["rate_hz_per_s"]) - CODAR_RATE_HZ_S) <= 0.02 * CODAR_RATE_HZ_S
            and s["period_s"] is not None and 0.9 <= s["period_s"] <= 1.1]
    assert hits, streaks


def test_disabled_exits_zero(tmp_path, capsys):
    cfg_for(tmp_path)
    assert main(["--etc", str(tmp_path), "scout", "AC0G-T7"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and "disabled" in out[0]


def test_dry_run_writes_no_files(tmp_path, capsys, monkeypatch):
    """--dry-run prints rows and opens no sink or JSONL."""
    import chirp_recorder.scout.runner as runner
    cfg = cfg_for(tmp_path, '\n[scout]\nenabled = true\n')
    built = {}

    class FakeScout:
        def __init__(self, settings, reporter_id, *, writer, **kw):
            built["writer"] = writer

        def run(self, stop, max_dwells=None):
            built["max_dwells"] = max_dwells
            built["writer"]("sweep", {"time": "2026-11-28T12:00:00.000Z"})

        def close(self):
            pass

    monkeypatch.setattr(runner, "Scout", FakeScout)
    monkeypatch.setattr(runner, "make_scout_writer",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("sink built")))
    assert main(["--etc", str(tmp_path), "scout", "AC0G-T7", "--dry-run", "--dwells", "2"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert json.loads(lines[-1]) == {"table": "sweep",
                                     "row": {"time": "2026-11-28T12:00:00.000Z"}}
    assert built["max_dwells"] == 2
    assert not (cfg.output_dir / "scout" / "rows").exists()
