"""Client contract 0.8 surfaces: inventory, validate, version, status, stations, daemon."""
from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

from cli_support import *  # noqa: F401,F403  (fixtures)

ROOT = Path(__file__).resolve().parents[1]


def test_validate_reports_capacity(tmp_etc, configured, run_cli_out):
    v = json.loads(run_cli_out("validate", "--json", "--instance", "AC0G-B4"))
    assert "capacity" in v and {"admitted", "refused", "estimated"} <= set(v["capacity"])
    assert v["ok"] is True and v["channels"]
    assert all(isinstance(n, str) for n in v["capacity"]["admitted"])


def test_validate_uses_live_headroom(tmp_etc, configured, run_cli_out, monkeypatch):
    from chirp_recorder.core.capacity import Headroom
    import chirp_recorder.contract as contract
    seen = []
    real = contract.admit

    def spy(plans, head, costs, version, **kw):
        seen.append(head)
        return real(plans, head, costs, version, **kw)
    monkeypatch.setattr(contract, "admit", spy)
    monkeypatch.setattr(contract, "read_headroom", lambda: Headroom(
        cpu_psi_some_avg60=99.0, mem_psi_some_avg60=0.0, load1=0.0, online_cpus=4,
        mem_available=8 * 2**30, mem_total=16 * 2**30, unit_memory_max=None))
    v = json.loads(run_cli_out("validate", "--json", "--instance", "AC0G-B4"))
    assert seen and v["capacity"]["admitted"] == []
    assert v["capacity"]["refused"][0][1].startswith("cpu pressure")


def test_validate_bad_config_fails_with_issues(tmp_etc, configured, run_cli, run_cli_out):
    (tmp_etc / "AC0G-B4.toml").write_text("[instance]\nreporter_id = \"AC0G/B4\"\n")
    out = run_cli_out("validate", "--json", "--instance", "AC0G-B4")
    v = json.loads(out)
    assert v["ok"] is False and any(i["severity"] == "fail" for i in v["issues"])
    assert run_cli("validate", "--json", "--instance", "AC0G-B4") != 0


def test_inventory_shape(tmp_etc, configured, run_cli_out, tmp_path):
    inv = json.loads(run_cli_out("inventory", "--json", "--instance", "AC0G-B4"))
    assert inv["client"] == "chirp-recorder" and inv["contract_version"] == "0.8"
    assert inv["config_paths"] == {"AC0G-B4": str(tmp_etc / "AC0G-B4.toml")}
    (inst,) = inv["instances"]
    assert inst["instance"] == "AC0G-B4" and inst["reporter_id"] == "AC0G/B4"
    assert inst["uses_timing_calibration"] is True
    assert inst["provides_timing_calibration"] is False
    assert inst["timing_authority_applied"] is None
    assert inst["radiod_status_dns"] == "b4-status.local"
    assert inst["ka9q_channels"] == len(inst["channels"]) == 2
    assert {"admitted", "refused", "estimated"} <= set(inst["capacity"])
    ch = inst["channels"][0]
    assert ch["groups"][0]["tx_ids"] and "group_key" in ch["groups"][0]
    assert inst["frequencies_hz"] == [int(c["centre_hz"]) for c in inst["channels"]]
    assert inst["data_sinks"][0]["kind"] == "file"
    assert inv["log_paths"]["AC0G-B4"]["process"] == "/var/log/chirp-recorder/AC0G-B4.log"
    assert inst["chain_delay_ns"] == {}


def test_inventory_chain_delay_per_radiod(tmp_path, tmp_etc, configured, run_cli_out,
                                          monkeypatch):
    from chirp_recorder.core import timing
    env = tmp_path / "coordination.env"
    env.write_text("RADIOD_B4_STATUS=b4-status.local\nRADIOD_B4_CHAIN_DELAY_NS=1234\n")
    monkeypatch.setattr(timing, "COORDINATION_ENV", env)
    inv = json.loads(run_cli_out("inventory", "--json"))
    assert inv["instances"][0]["chain_delay_ns"] == {"B4": 1234}


def _point_output(tmp_etc, key="AC0G-B4"):
    """Move the instance's output_dir under tmp (the default is /var/lib)."""
    out_dir = tmp_etc.parent / "data"
    out_dir.mkdir(exist_ok=True)
    path = tmp_etc / f"{key}.toml"
    path.write_text(path.read_text().replace(
        "[processing]", f'[paths]\noutput_dir = "{out_dir}"\n\n[processing]'))
    return out_dir


def test_inventory_reads_timing_authority_applied(tmp_etc, configured, run_cli_out):
    from hamsci_dsp.timing import write_applied_state
    out_dir = _point_output(tmp_etc)
    block = {"mode": "t6", "channels": {"total": 2, "anchored": 2, "applied": 2}}
    write_applied_state(out_dir / "applied-state.json", block)
    inv = json.loads(run_cli_out("inventory", "--json", "--instance", "AC0G-B4"))
    assert inv["instances"][0]["timing_authority_applied"] == block


def test_inventory_lists_a_broken_instance(tmp_etc, configured, run_cli_out):
    (tmp_etc / "AC0G-B1A.toml").write_text("not toml [[")
    inv = json.loads(run_cli_out("inventory", "--json"))
    assert {i["instance"] for i in inv["instances"]} == {"AC0G-B4", "AC0G-B1A"}
    assert any(i["severity"] == "fail" and i["instance"] == "AC0G-B1A" for i in inv["issues"])


def test_stdout_is_clean_json(tmp_etc, configured, run_cli_full):
    for argv in (("inventory", "--json"), ("validate", "--json"), ("version", "--json"),
                 ("stations", "--json", "--instance", "AC0G-B4")):
        rc, out, _ = run_cli_full(*argv)
        assert out.lstrip().startswith("{"), argv
        json.loads(out)


def test_stations_json_ranked_with_reasons(tmp_etc, configured, run_cli_out):
    s = json.loads(run_cli_out("stations", "--json", "--instance", "AC0G-B4"))
    assert s["instance"] == "AC0G-B4" and s["position_source"] == "grid_centre"
    assert s["choices"] and all("reason" in c and c["transmitters"] for c in s["choices"])
    shared = [c["shared_t6"] for c in s["choices"]]
    assert shared == sorted(shared, reverse=True)      # establish phase


def test_stations_needs_one_instance(tmp_etc, configured, run_cli, monkeypatch):
    monkeypatch.setenv("STATION_GRID", "EM38ww")
    assert run_cli("config", "init", "--non-interactive", "--instance", "AC0G-B1A",
                   "--reporter-id", "AC0G/B1A", "--radiod-status", "b1a-status.local") == 0
    assert run_cli("stations", "--json") != 0


def test_status_not_running_and_running(tmp_etc, configured, run_cli_out):
    out = run_cli_out("status")
    assert "AC0G-B4" in out and "not running" in out
    out_dir = _point_output(tmp_etc)
    (out_dir / "applied-state.json").write_text(json.dumps({
        "instance": "AC0G-B4", "channels": [
            {"name": "4.4630MHz", "admission": "admitted", "delay_basis": "t6",
             "intervals": 4},
            {"name": "4.4505MHz", "admission": "refused", "refused_reason": "cpu pressure 22.0 %",
             "delay_basis": None, "intervals": 0}]}))
    out = run_cli_out("status", "--instance", "AC0G-B4")
    assert "4.4630MHz" in out and "admitted" in out and "cpu pressure 22.0 %" in out
    j = json.loads(run_cli_out("status", "--json", "--instance", "AC0G-B4"))
    assert j["instances"][0]["running"] is True


def test_version_subcommand(run_cli_out):
    assert json.loads(run_cli_out("version", "--json"))["contract_version"] == "0.8"


class _FakeDaemon:
    calls = []
    rc = 7

    def __init__(self, cfg, measure=False, **kw):
        _FakeDaemon.calls.append((cfg.key, measure))

    def run(self):
        return _FakeDaemon.rc


def test_daemon_passes_the_exit_code_through(tmp_etc, configured, run_cli, monkeypatch):
    import chirp_recorder.core.daemon as d
    monkeypatch.setattr(d, "Daemon", _FakeDaemon)
    _FakeDaemon.calls.clear()
    assert run_cli("daemon", "--instance", "AC0G-B4") == 7
    _FakeDaemon.rc = 0
    assert run_cli("daemon", "--instance", "AC0G-B4", "--measure-cost") == 0
    assert _FakeDaemon.calls == [("AC0G-B4", False), ("AC0G-B4", True)]


def test_daemon_refuses_a_bad_config(tmp_etc, run_cli):
    assert run_cli("daemon", "--instance", "AC0G-B4") != 0     # no such config


def test_daemon_takes_the_instance_from_the_environment(tmp_etc, configured, run_cli,
                                                        monkeypatch):
    import chirp_recorder.core.daemon as d
    monkeypatch.setattr(d, "Daemon", _FakeDaemon)
    _FakeDaemon.rc, _FakeDaemon.calls = 1, []
    monkeypatch.setenv("SIGMOND_INSTANCE", "AC0G-B4")
    assert run_cli("daemon") == 1 and _FakeDaemon.calls[0][0] == "AC0G-B4"


def test_module_entry_runs_as_the_unit_does(tmp_path):
    out = subprocess.run([sys.executable, "-m", "chirp_recorder.cli", "version", "--json"],
                         capture_output=True, text=True, check=True).stdout
    assert json.loads(out)["name"] == "chirp-recorder"


# --- the receiver_channels hook (Task 6 carry) ---------------------------------

def test_receiver_channels_parser():
    from chirp_recorder.sigmond_tui import parse_receiver_channels
    cfg = {"channel": [{"status": "b4-status.local", "centre_hz": 4463000},
                       {"status": "b4-status.local", "centre_hz": 4450500.0}]}
    status, freqs, enc = parse_receiver_channels(cfg)
    assert status == "b4-status.local" and freqs == {4463000, 4450500} and enc == 4
    assert parse_receiver_channels({}) == ("", set(), 4)


def test_deploy_registers_receiver_channels():
    d = tomllib.loads((ROOT / "deploy.toml").read_text())
    rc = d["client_features"]["receiver_channels"]
    assert rc["parser_file"] == "src/chirp_recorder/sigmond_tui.py"
    assert rc["parser_attr"] == "parse_receiver_channels" and rc["per_instance"] is True
    assert (ROOT / rc["parser_file"]).is_file()


# --- applied-state, end to end (fix round 1) ---------------------------------

def _daemon_with_workers(tmp_etc, bases):
    from types import SimpleNamespace
    from chirp_recorder.config import load_instance
    from chirp_recorder.core.daemon import Daemon
    cfg = load_instance("AC0G-B4", tmp_etc)
    d = Daemon(cfg, headroom_reader=lambda: None, costs={}, notify=lambda m: None)
    d._workers = [SimpleNamespace(
        plan=p, stats={"delay_basis": b, "t_level": "T6" if b == "t6" else None,
                       "delivered": 1})
        for p, b in zip(cfg.channels, bases)]
    return d


def test_daemon_writes_what_validate_reads(tmp_etc, configured, run_cli_out):
    _point_output(tmp_etc)
    d = _daemon_with_workers(tmp_etc, ["t6", "t6"])
    d._write_applied_state()
    v = json.loads(run_cli_out("validate", "--json", "--instance", "AC0G-B4"))
    block = v["timing_authority_applied"]
    assert block["source"] == "hf-timestd" and block["tier"] == "T6"
    assert block["channels"] == {"total": 2, "anchored": 2, "applied": 2}
    own = json.loads((tmp_etc.parent / "data" / "applied-state.json").read_text())
    assert own["instance"] == "AC0G-B4" and own["schema"] == "applied-state/v1"


def test_mixed_labels_read_null(tmp_etc, configured, run_cli_out):
    _point_output(tmp_etc)
    _daemon_with_workers(tmp_etc, ["t6", "radiod"])._write_applied_state()
    inv = json.loads(run_cli_out("inventory", "--json", "--instance", "AC0G-B4"))
    assert inv["instances"][0]["timing_authority_applied"] is None


def test_bad_applied_state_gives_null_and_a_warning(tmp_etc, configured, run_cli_full,
                                                    monkeypatch):
    import hamsci_dsp.timing as ht
    _point_output(tmp_etc)

    def boom(*a, **k):
        raise ValueError("garbled")
    monkeypatch.setattr(ht, "read_applied_state", boom)
    rc, out, err = run_cli_full("inventory", "--json", "--instance", "AC0G-B4")
    assert rc == 0 and json.loads(out)["instances"][0]["timing_authority_applied"] is None
