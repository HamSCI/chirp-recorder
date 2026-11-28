import tomllib, pathlib, subprocess, json
ROOT = pathlib.Path(__file__).resolve().parents[1]

def test_unit_passes_instance_only():
    unit = (ROOT / "systemd/chirp-recorder@.service").read_text()
    assert "daemon --instance %i" in unit and "--radiod-id %i" not in unit and "User=chirprec" in unit
    assert "-/run/hf-timestd" in unit

def test_manifest_pipelines_match_codar_tags():
    d = tomllib.loads((ROOT / "deploy.toml").read_text())
    p = {x["name"]: x for x in d["hs_uploader"]["pipeline"]}
    assert p["chirp-sigmondb-arrival"]["transport"]["tar_tag"] == "chirp-arrival"
    assert p["chirp-sigmondb-capture"]["transport"]["tar_tag"] == "chirp-capture"
    by_producer = {}
    for x in p.values():
        by_producer.setdefault(x["source"]["producer"], set()).add(x["name"])
    assert by_producer == {
        "chirp-recorder": {"chirp-sigmondb-arrival", "chirp-sigmondb-capture"},
        "chirp-scout": {"chirp-sigmondb-sweep", "chirp-sigmondb-scout-dwell"}}
    assert "instances_from" not in d.get("systemd", {}) and "fragment" not in d.get("radiod", {})

def test_version_json():
    out = subprocess.run(["uv", "run", "chirp-recorder", "version", "--json"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    assert json.loads(out)["contract_version"] == "0.8"


def _scout_unit():
    return (ROOT / "systemd/chirp-scout@.service").read_text()


def test_scout_unit_runs_the_scout_verb():
    unit = _scout_unit()
    lines = [ln.strip() for ln in unit.splitlines()]
    exec_start = next(ln for ln in lines if ln.startswith("ExecStart="))
    assert exec_start.endswith("-m chirp_recorder.cli scout %i")
    assert "--instance" not in exec_start
    for want in ("CPUQuota=20%", "MemoryMax=512M", "Nice=10", "Restart=on-failure",
                 "RestartSec=30", "User=chirprec", "ProtectSystem=strict",
                 "EnvironmentFile=-/etc/chirp-recorder/env/%i.env",
                 "StandardOutput=append:/var/log/chirp-recorder/scout-%i.log"):
        assert want in lines, want
    assert "Type=notify" not in lines and "WatchdogSec=180" not in lines
    d = tomllib.loads((ROOT / "deploy.toml").read_text())
    assert "chirp-scout@.service" in d["systemd"]["templated_units"]
    assert any(s.get("dst") == "/etc/systemd/system/chirp-scout@.service"
               and s.get("src") == "systemd/chirp-scout@.service"
               for s in d["install"]["steps"])


def test_scout_pipelines_ship_the_new_tables():
    import re
    d = tomllib.loads((ROOT / "deploy.toml").read_text())
    p = {x["name"]: x for x in d["hs_uploader"]["pipeline"]}
    want = {"chirp-sigmondb-sweep": ("sweep", "sigmondb.sweep", "chirp-sweep"),
            "chirp-sigmondb-scout-dwell": ("scout_dwell", "sigmondb.scout_dwell", "chirp-dwell")}
    for name, (table, primary, tag) in want.items():
        x = p[name]
        assert x["source"]["table"] == table and x["source"]["producer"] == "chirp-scout"
        assert x["transport"]["primary_table_name"] == primary
        assert x["transport"]["tar_tag"] == tag and re.fullmatch(r"[a-z0-9-]+", tag)
        assert x["transport"]["name"] == f"wsprdaemon-tar-sftp:{name}"
        assert "sigmondb-scout" in x["requires_features"]
        assert "sigmondb-tar" in x["requires_features"]
    deps = {x["name"]: x["version"] for x in d["deps"]["pypi"]}
    assert deps["ka9q-python"] == ">=3.29.0" and deps["hamsci-dsp"] == ">=0.11.0"


def test_template_scout_table_parses(tmp_path):
    from chirp_recorder.scout.settings import ScoutSettings, scout_settings
    text = (ROOT / "config/chirp-recorder-config.toml.template").read_text()
    lines = text.splitlines()
    start = lines.index("# [scout]")
    block = []
    for ln in lines[start:]:
        if not ln.startswith("#"):
            break
        block.append(ln[2:] if ln.startswith("# ") else "")
    body = "\n".join(block)
    assert "enabled = false" in body
    from tests.test_scout_settings import cfg_for
    s = scout_settings(cfg_for(tmp_path, "\n" + body + "\n"))
    assert s == ScoutSettings(state_dir=pathlib.Path("/var/lib/chirp-recorder/CALL-ID/scout"))
