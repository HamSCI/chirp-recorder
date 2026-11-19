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
