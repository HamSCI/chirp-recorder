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
    assert {x["source"]["producer"] for x in p.values()} == {"chirp-recorder"}
    assert "instances_from" not in d.get("systemd", {}) and "fragment" not in d.get("radiod", {})

def test_version_json():
    out = subprocess.run(["uv", "run", "chirp-recorder", "version", "--json"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    assert json.loads(out)["contract_version"] == "0.8"
