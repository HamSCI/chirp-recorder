from chirp_recorder.core.catalogue import Transmitter, load_seed


def test_seed_loads_hfrnet_list():
    s = load_seed()
    assert len(s) >= 100 and all(t.kind == "codar" for t in s.values())


def test_seed_period_is_inverse_repetition():
    assert {t.period_s for t in load_seed().values()} <= {1.0, 0.5, 0.25}


def test_waveform_passes_gate_period_through():
    t = Transmitter("A", "codar", 4.463e6, -25_733.913, 1.0, 25_734, 40.0, -74.0,
                    gate_period_s=0.004)
    wf = t.waveform()
    assert wf.gate_period_s == 0.004 and wf.centre_hz == 4.463e6
    assert wf.rate_hz_per_s == -25_733.913 and wf.period_s == 1.0


MID_ATLANTIC_4463 = {"AMAG", "ASSA", "BLCK", "BRIG", "CEDR", "HEMP", "HOOK", "LISL", "LOVE",
                     "MRCH", "NANT", "NAUS", "OCRA"}


def test_mid_atlantic_4463_group_carries_its_measured_gate():
    """The 11-03 result (codar-mf/clean.py): 1945.6 us, gate on at the sweep
    start.  Only the Mid-Atlantic sites of the 4.463 MHz -25.73 kHz/s group
    carry it; the West Coast, Caribbean and Alaska sites keep a fitted gate."""
    s = load_seed()
    gated = {t.id for t in s.values() if t.gate_period_s is not None}
    assert gated == MID_ATLANTIC_4463
    assert {s[t].gate_period_s for t in gated} == {1.9456e-3}
    assert all(s[t].gate_period_s is None for t in ("BMLR", "MAN1", "PYFC", "THOM", "WAIN"))


def test_the_committed_seed_is_what_rebuild_seed_writes():
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    src = root.parent / "codar-sounder" / "data" / "codar-stations.toml"
    if not src.exists():
        import pytest
        pytest.skip(f"no codar-sounder checkout beside this repo ({src})")
    out = subprocess.run([sys.executable, "-I", str(root / "scripts" / "rebuild-seed.py"),
                          str(src)], capture_output=True, text=True, check=True).stdout
    assert out == (root / "data" / "chirp-seed.toml").read_text()
