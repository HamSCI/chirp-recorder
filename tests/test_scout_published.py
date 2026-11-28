import json
from urllib.parse import parse_qs, urlparse

import pytest

from chirp_recorder.scout import published
from chirp_recorder.scout.published import Published, load_published

CHIRPS = (b'{"group_key":"g1","centre_mhz":4.463,"rate_khz_s":-100.0,'
          b'"period_s":0.5,"span_khz":50.0}\n'
          b'{"group_key":"g2","centre_mhz":9.33,"rate_khz_s":25.0,'
          b'"period_s":null,"span_khz":null}\n')
REJECTS = (b'{"chirp_id":"r1","centre_mhz":13.5,"rate_khz_s":10.0,'
           b'"period_s":1.0,"span_khz":null}\n')


class Fake:
    def __init__(self, answers):
        self.answers, self.urls = answers, []

    def __call__(self, url, timeout):
        self.urls.append(url)
        host = url.split("/?")[0]
        a = self.answers[host]
        if isinstance(a, Exception):
            raise a
        return a[1] if "chirp_reject_local" in url else a[0]


def test_parses_hosts_in_order(tmp_path):
    f = Fake({"https://a": OSError("down"), "https://b": (CHIRPS, REJECTS)})
    p = load_published(tmp_path / "st", ["https://a", "https://b"], "AC0G/T7",
                       fetch=f, now=lambda: 1000.0)
    assert p.source == "https://b" and p.fetched_at == 1000.0
    c = {s.key: s for s in p.chirps}
    assert c["g1"].centre_hz == 4.463e6 and c["g1"].rate_hz_per_s == -1e5
    assert c["g1"].period_s == 0.5 and c["g1"].span_hz == 50e3
    assert c["g2"].period_s is None and c["g2"].span_hz is None
    assert p.rejects[0].key == "r1" and p.rejects[0].span_hz is None
    saved = json.loads((tmp_path / "st" / "published.json").read_text())
    assert saved["source"] == "https://b" and saved["fetched_at"] == 1000.0
    assert saved["chirps"][0]["centre_hz"] == 4.463e6
    assert not list((tmp_path / "st").glob("*.tmp"))


def test_host_answers_only_when_both_queries_parse(tmp_path):
    f = Fake({"https://a": (CHIRPS, b"not json\n"), "https://b": (CHIRPS, REJECTS)})
    p = load_published(tmp_path, ["https://a", "https://b"], "X/1", fetch=f)
    assert p.source == "https://b"


def test_unreachable_hosts_use_cache_then_empty(tmp_path, caplog):
    ok = Fake({"https://a": (CHIRPS, REJECTS)})
    load_published(tmp_path, ["https://a"], "X/1", fetch=ok, now=lambda: 500.0)
    bad = Fake({"https://a": OSError("secret body")})
    with caplog.at_level("WARNING"):
        p = load_published(tmp_path, ["https://a"], "X/1", fetch=bad, now=lambda: 900.0)
    assert p.source == "cache" and p.fetched_at == 500.0
    assert [s.key for s in p.chirps] == ["g1", "g2"] and p.rejects[0].key == "r1"
    assert "https://a" in caplog.text and "OSError" in caplog.text
    assert "secret body" not in caplog.text
    (tmp_path / "published.json").unlink()
    p = load_published(tmp_path, ["https://a"], "X/1", fetch=bad)
    assert p == Published((), (), None, "none")


def test_corrupt_cache_counts_as_none(tmp_path, caplog):
    (tmp_path / "published.json").write_text("{oops")
    bad = Fake({"https://a": OSError("x")})
    p = load_published(tmp_path, ["https://a"], "X/1", fetch=bad)
    assert p == Published((), (), None, "none")


def test_reporter_id_is_a_parameter(tmp_path):
    f = Fake({"https://a": (CHIRPS, REJECTS)})
    load_published(tmp_path, ["https://a"], "AC0G/T7", fetch=f)
    chirp_url, rej_url = f.urls
    assert "param_reporter_id" not in chirp_url
    q = parse_qs(urlparse(rej_url).query)
    assert q["param_reporter_id"] == ["AC0G/T7"]
    assert "{reporter_id:String}" in q["query"][0] and "AC0G" not in q["query"][0]
    assert "AC0G%2FT7" in rej_url


def test_https_get_rejects_other_schemes():
    with pytest.raises(ValueError):
        published._https_get("http://db2.wspr.live/?query=1", 5.0)
    with pytest.raises(ValueError):
        published._https_get("file:///etc/passwd", 5.0)
