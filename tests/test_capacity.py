from chirp_recorder.core.capacity import (Headroom, admit, cost_for, load_costs,
                                          read_headroom)
from chirp_recorder.core.catalogue import Transmitter
from chirp_recorder.core.channels import form_channels


def _plans(n):
    return [form_channels([Transmitter(f"T{i}", "codar", (4.4 + i) * 1e6, -25_733.913,
                                       1.0, 25_734, 40, -74)], "s")[0]
            for i in range(n)]


GiB = 1 << 30
COSTS = {"0.1.0": {64_000: (0.20, 300 << 20)}}
CALM = Headroom(8, 2.0, 1.0, 0.0, 16 * GiB, 10 * GiB, 2 * GiB)


def test_admits_what_fits():
    a = admit(_plans(3), CALM, COSTS, "0.1.0")
    assert len(a.admitted) == 3 and not a.estimated


def test_admit_refuses_under_pressure():
    hot = Headroom(14, 15.0, 22.0, 0.0, 16 * GiB, 10 * GiB, 2 * GiB)
    a = admit(_plans(2), hot, COSTS, "0.1.0")
    assert not a.admitted and "22.0" in a.refused[0][1]
    assert a.refused[0][1] == "cpu pressure 22.0 %"


def test_memory_pressure_refuses():
    hot = Headroom(8, 1.0, 0.0, 7.5, 16 * GiB, 10 * GiB, 2 * GiB)
    a = admit(_plans(1), hot, COSTS, "0.1.0")
    assert not a.admitted and "memory pressure 7.5" in a.refused[0][1]


def test_unit_memory_cap_limits_channels():
    a = admit(_plans(8), CALM, COSTS, "0.1.0")
    assert len(a.admitted) == 5


def test_no_unit_cap_means_no_unit_limit():
    head = Headroom(64, 0.0, 0.0, 0.0, 64 * GiB, 60 * GiB, None)
    assert len(admit(_plans(8), head, COSTS, "0.1.0").admitted) == 8


def test_cpu_headroom_limits_channels():
    head = Headroom(4, 2.5, 0.0, 0.0, 16 * GiB, 10 * GiB, None)
    costs = {"0.1.0": {64_000: (0.5, 1 << 20)}}
    # 2.5 + 0.5 n <= 3.0  ->  one channel
    assert len(admit(_plans(4), head, costs, "0.1.0").admitted) == 1


def test_memory_floor_limits_channels():
    head = Headroom(64, 0.0, 0.0, 0.0, 10 * GiB, 2 * GiB, None)
    costs = {"0.1.0": {64_000: (0.1, 256 << 20)}}
    # 2048 - 256 n >= 1536  ->  two channels
    assert len(admit(_plans(5), head, costs, "0.1.0").admitted) == 2


def test_unmeasured_version_opens_one_channel():
    a = admit(_plans(3), CALM, {}, "0.1.0")
    assert len(a.admitted) == 1 and "no measured cost" in a.refused[0][1]
    assert a.refused[0][1] == "no measured cost for chirp-recorder 0.1.0"
    assert len(a.refused) == 2


def test_older_measurement_marks_estimate():
    assert cost_for("0.2.0", 64_000, COSTS).measured is False
    assert admit(_plans(1), CALM, COSTS, "0.2.0").estimated


def test_newest_older_version_wins_and_newer_ignored():
    costs = {"0.1.0": {64_000: (0.1, 1)}, "0.1.5": {64_000: (0.3, 2)},
             "0.3.0": {64_000: (0.9, 9)}}
    c = cost_for("0.2.0", 64_000, costs)
    assert (c.cores, c.version, c.measured) == (0.3, "0.1.5", False)
    assert cost_for("0.0.9", 64_000, costs) is None


def test_exact_version_is_measured():
    c = cost_for("0.1.0", 64_000, COSTS)
    assert (c.cores, c.mem_bytes, c.measured) == (0.20, 300 << 20, True)


def test_unmeasured_rate_borrows_the_next_rate_up():
    c = cost_for("0.1.0", 64_000, {"0.1.0": {96_000: (0.3, 400 << 20)}})
    assert (c.cores, c.measured) == (0.3, False)


def test_rate_above_every_measured_rate_has_no_cost():
    assert cost_for("0.1.0", 192_000, COSTS) is None


def test_estimated_rate_marks_admission_estimated():
    costs = {"0.1.0": {96_000: (0.3, 100 << 20)}}
    assert admit(_plans(1), CALM, costs, "0.1.0").estimated


def test_measuring_switch_admits_all_and_says_so():
    a = admit(_plans(4), CALM, {}, "0.1.0", measure=True)
    assert len(a.admitted) == 4 and a.estimated and not a.refused


def test_measuring_switch_still_refuses_nothing_even_when_hot():
    hot = Headroom(14, 15.0, 22.0, 0.0, 16 * GiB, 10 * GiB, 2 * GiB)
    assert len(admit(_plans(2), hot, COSTS, "0.1.0", measure=True).admitted) == 2


def test_to_dict_shape():
    d = admit(_plans(2), CALM, {}, "0.1.0").to_dict()
    assert d["admitted"] == [_plans(2)[0].name] and d["refused"][0][0] == _plans(2)[1].name
    assert d["estimated"] is True
    assert isinstance(d["refused"][0], list) and len(d["refused"][0]) == 2
    assert set(d) == {"admitted", "refused", "estimated"}


def test_load_costs_turns_string_rates_into_ints(tmp_path):
    p = tmp_path / "costs.toml"
    p.write_text('[versions."0.1.0".rates."64000"]\ncores = 0.2\nmem_mib = 300\n'
                 'measured_on = "B4"\nmeasured_at = "2026-11-20"\n')
    assert load_costs(p) == {"0.1.0": {64_000: (0.2, 300 << 20)}}


def test_shipped_costs_file_loads_empty():
    assert load_costs() == {}


def _fake_proc(root, cpu_psi="12.50", mem_psi="3.25", omit_psi=False):
    (root / "pressure").mkdir(parents=True)
    if not omit_psi:
        (root / "pressure" / "cpu").write_text(
            f"some avg10=1.00 avg60={cpu_psi} avg300=0.00 total=1\n")
        (root / "pressure" / "memory").write_text(
            f"some avg10=0.00 avg60={mem_psi} avg300=0.00 total=1\n"
            "full avg10=0.00 avg60=9.99 avg300=0.00 total=1\n")
    (root / "meminfo").write_text("MemTotal:       16384000 kB\nMemFree: 1 kB\n"
                                  "MemAvailable:    8192000 kB\n")
    (root / "loadavg").write_text("1.50 1.00 0.50 2/300 1234\n")


def test_read_headroom_fake_proc(tmp_path):
    _fake_proc(tmp_path / "proc")
    cg = tmp_path / "cg"
    cg.mkdir()
    (cg / "memory.max").write_text("2147483648\n")
    h = read_headroom(tmp_path / "proc", cg)
    assert (h.load1, h.cpu_psi_some_avg60, h.mem_psi_some_avg60) == (1.5, 12.5, 3.25)
    assert (h.mem_total, h.mem_available) == (16384000 * 1024, 8192000 * 1024)
    assert h.unit_memory_max == 2 << 30 and h.online_cpus >= 1


def test_read_headroom_max_and_missing_psi(tmp_path):
    _fake_proc(tmp_path / "proc", omit_psi=True)
    cg = tmp_path / "cg"
    cg.mkdir()
    (cg / "memory.max").write_text("max\n")
    h = read_headroom(tmp_path / "proc", cg)
    assert h.unit_memory_max is None
    assert (h.cpu_psi_some_avg60, h.mem_psi_some_avg60) == (0.0, 0.0)


def test_read_headroom_no_cgroup_files(tmp_path):
    _fake_proc(tmp_path / "proc")
    assert read_headroom(tmp_path / "proc", tmp_path / "nowhere").unit_memory_max is None


def test_unmeasured_first_plan_is_estimated_when_calm():
    a = admit(_plans(2), CALM, {}, "0.1.0")
    assert len(a.admitted) == 1 and a.estimated


def test_unmeasured_first_plan_refused_on_loaded_box():
    busy = Headroom(8, 7.5, 1.0, 0.0, 16 * GiB, 10 * GiB, 2 * GiB)
    a = admit(_plans(2), busy, {}, "0.1.0")
    assert not a.admitted and "load" in a.refused[0][1]


def test_placeholder_cost_row_is_skipped(tmp_path):
    p = tmp_path / "costs.toml"
    p.write_text('[versions."0.1.0".rates."64000"]\ncores = 0.0\nmem_mib = 0\n'
                 'measured_on = ""\nmeasured_at = ""\n')
    assert load_costs(p) == {}


def test_missing_psi_logs_warning(tmp_path, caplog):
    _fake_proc(tmp_path / "proc", omit_psi=True)
    with caplog.at_level("WARNING"):
        read_headroom(tmp_path / "proc", tmp_path)
    assert any("pressure" in r.getMessage() and r.levelname == "WARNING"
               for r in caplog.records)


def test_version_reexported():
    import chirp_recorder
    from chirp_recorder import version
    assert version.__version__ == chirp_recorder.__version__
