import dataclasses
import json
import logging
from types import SimpleNamespace

from chirp_recorder.core.timing import Label, Tie, chain_delay_from_env, normalise_status
from chirp_recorder.core.source import StatusPair
from conftest import ANCHOR_IDX, ANCHOR_UTC, NOW, gps_ns_from_unix_ns, make_interval

IN = 129_600_000


def radiod_utc(iv):
    p = iv.status_pair
    d = (iv.first_rtp - p.rtp_timesnap) & 0xFFFFFFFF
    d = d - 2**32 if d >= 2**31 else d
    return p.gps_time_ns + (315_964_800 - 18) * 10**9 + round(d * 1e9 / p.output_samprate)


def tie(path, **kw):
    kw.setdefault("radiod_status", "b4-status.local")
    return Tie(anchor_path=path, clock_ns=lambda: NOW, **kw)


def test_t6_label_projects_from_the_anchor(anchor_file, interval):
    lab = tie(anchor_file, chain_delay_ns=18_700).label(interval)
    assert lab.delay_basis == "t6" and lab.provenance["origin"] == "t6_anchor"
    assert lab.utc_ns == ANCHOR_UTC + round((interval.first_input_index - ANCHOR_IDX) * 1e9 / IN)
    assert isinstance(lab.utc_ns, int)
    pv = lab.provenance
    assert (pv["u_epoch_ns"], pv["k"], pv["p"], pv["a_level_provenance"],
            pv["counter_epoch_id"]) == (230.0, 1.0, 0.5, "measured", "epoch-7")
    assert "chain_delay_ns" not in lab.engineering
    assert lab.engineering["delay_basis"] == "t6"


def test_new_run_never_uses_old_anchor(anchor_file, interval_from_new_run, fake_auth, caplog):
    with caplog.at_level(logging.INFO):
        lab = tie(anchor_file, authority=fake_auth).label(interval_from_new_run)
    assert lab.delay_basis == "authority" and lab.provenance["u_epoch_ns"] is None
    assert "run" in caplog.text


def test_anchor_for_another_radiod_is_ignored(anchor_file, interval, fake_auth):
    lab = tie(anchor_file, radiod_status="b1b-status.local", authority=fake_auth).label(interval)
    assert lab.delay_basis == "authority"


def test_name_mismatch_logs_info_once_per_channel(anchor_file, interval, fake_auth, caplog):
    t = tie(anchor_file, radiod_status="b1b-status.local", authority=fake_auth)
    with caplog.at_level(logging.INFO):
        t.label(interval)
        t.label(interval)
        t.label(dataclasses.replace(interval, channel="13.4875MHz"))
    recs = [r for r in caplog.records if r.levelno == logging.INFO and "b4-status" in r.getMessage()]
    assert len(recs) == 2
    assert "b1b-status" in recs[0].getMessage() and "4.4630MHz" in recs[0].getMessage()


def test_stale_anchor_falls_back(anchor_file, interval):
    t = Tie(radiod_status="b4-status.local", anchor_path=anchor_file, authority=None,
            clock_ns=lambda: NOW + 121 * 10**9)
    lab = t.label(interval)
    assert lab.delay_basis == "radiod" and lab.provenance["origin"] == ""


def test_stale_anchor_uses_authority(anchor_file, interval, fake_auth):
    t = Tie(radiod_status="b4-status.local", anchor_path=anchor_file, authority=fake_auth,
            clock_ns=lambda: NOW + 121 * 10**9)
    assert t.label(interval).delay_basis == "authority"


def test_future_anchor_refused(anchor_file, interval, fake_auth):
    t = Tie(radiod_status="b4-status.local", anchor_path=anchor_file, authority=fake_auth,
            clock_ns=lambda: NOW - 16 * 10**9)       # file stamped 26 s ahead of this clock
    assert t.label(interval).delay_basis == "authority"


def test_missing_anchor_falls_back(tmp_path, interval):
    lab = tie(tmp_path / "absent.json").label(interval)
    assert lab.delay_basis == "radiod"


def test_label_from_half_sample_index(anchor_file, interval):
    half = dataclasses.replace(interval, first_input_index=interval.first_input_index + 0.5)
    t = tie(anchor_file)
    assert t.label(half).utc_ns - t.label(interval).utc_ns in (3, 4)


def test_status_names_compare_normalised(anchor_file, interval):
    lab = tie(anchor_file, radiod_status="B4-Status.local:5006").label(interval)
    assert lab.delay_basis == "t6"
    assert normalise_status("B4-Status.local") == normalise_status("b4-status") == "b4-status"
    assert normalise_status("b4-status:5006") == "b4-status"


def test_authority_label_flags_its_uncertainty(anchor_file, interval_from_new_run, fake_auth):
    lab = tie(anchor_file, authority=fake_auth, chain_delay_ns=18_700).label(interval_from_new_run)
    assert lab.engineering["authority_sigma_ns"] == fake_auth.sigma_ns
    assert lab.engineering["chain_delay_ns"] == 18_700 and lab.provenance["u_epoch_ns"] is None
    assert lab.engineering["t_level"] == "T3" and lab.t_level == "T3"
    assert lab.engineering["delay_basis"] == "authority"
    assert lab.utc_ns == radiod_utc(interval_from_new_run) + fake_auth.snapshot.rtp_to_utc_offset_ns


def test_unusable_offset_gives_radiod_label(anchor_file, interval):
    snap = SimpleNamespace(offset_usable=False, rtp_to_utc_offset_ns=None,
                           t_level_active=None, sigma_ns=None)
    auth = SimpleNamespace(read=lambda: snap)
    lab = tie(anchor_file, authority=auth, chain_delay_ns=900).label(
        dataclasses.replace(interval, input_epoch_gps_s=0.0))
    assert lab.delay_basis == "radiod" and lab.t_level is None
    assert lab.utc_ns == radiod_utc(interval)
    assert lab.engineering["chain_delay_ns"] == 900
    assert lab.engineering["delay_basis"] == "radiod"
    assert lab.provenance["u_epoch_ns"] is None


def test_authority_read_none_gives_radiod(tmp_path, interval):
    auth = SimpleNamespace(read=lambda: None)
    assert tie(tmp_path / "x", authority=auth).label(interval).delay_basis == "radiod"


def test_rtp_wrap_in_radiod_arithmetic(tmp_path):
    pair = StatusPair(gps_time_ns=gps_ns_from_unix_ns(1_799_999_990 * 10**9), rtp_timesnap=0xFFFFFF00,
                      output_samprate=64_000)
    iv = make_interval(first_rtp=0x100, pair=pair)       # 0x200 = 512 samples past the snapshot
    assert tie(tmp_path / "x").label(iv).utc_ns == 1_799_999_990 * 10**9 + 512 * 1_000_000_000 // 64_000
    pair2 = StatusPair(gps_time_ns=gps_ns_from_unix_ns(1_799_999_990 * 10**9), rtp_timesnap=0x100, output_samprate=64_000)
    iv2 = make_interval(first_rtp=0xFFFFFF00, pair=pair2)  # 512 samples before it
    assert tie(tmp_path / "x").label(iv2).utc_ns == 1_799_999_990 * 10**9 - 512 * 1_000_000_000 // 64_000


def test_integer_ns_precision_in_radiod_arithmetic(tmp_path):
    pair = StatusPair(gps_time_ns=gps_ns_from_unix_ns(1_799_999_990_000_000_001), rtp_timesnap=0, output_samprate=64_000)
    iv = make_interval(first_rtp=1, pair=pair)
    assert tie(tmp_path / "x").label(iv).utc_ns == 1_799_999_990_000_000_001 + 15_625


def test_chain_delay_from_env(tmp_path):
    env = tmp_path / "coordination.env"
    env.write_text("RADIOD_B4_CHAIN_DELAY_NS=18700\nRADIOD_B1B_RX888_CHAIN_DELAY_NS=junk\n")
    assert chain_delay_from_env("b4", env) == 18_700
    assert chain_delay_from_env("b1b-rx888", env) is None
    assert chain_delay_from_env("nope", env) is None
    assert chain_delay_from_env("b4", tmp_path / "absent") is None


def test_label_is_frozen(anchor_file, interval):
    lab = tie(anchor_file).label(interval)
    assert isinstance(lab, Label)


def test_radiod_label_lands_on_the_right_utc_date(tmp_path):
    # radiod reads 1451260818 s of GPS time at 2026-01-01T00:00:00Z (Unix 1767225600).
    pair = StatusPair(gps_time_ns=1_451_260_818 * 10**9, rtp_timesnap=0, output_samprate=64_000)
    lab = tie(tmp_path / "x").label(make_interval(first_rtp=0, pair=pair))
    assert lab.utc_ns == 1_767_225_600 * 10**9


def test_radiod_arithmetic_matches_ka9q_rtp_to_utc(tmp_path):
    from ka9q.discovery import ChannelInfo
    from ka9q.rtp_recorder import rtp_to_utc
    gps = gps_ns_from_unix_ns(1_799_999_990 * 10**9 + 123_456_789)
    pair = StatusPair(gps_time_ns=gps, rtp_timesnap=900_000, output_samprate=64_000)
    ci = ChannelInfo(ssrc=1, preset="iq", sample_rate=64_000, frequency=4.463e6, snr=0.0,
                     multicast_address="239.1.1.1", port=5004, gps_time=gps, rtp_timesnap=900_000)
    for rtp in (1_000_000, 900_000 + 2**30, 900_000 - 2**30):   # both sides of the snapshot
        iv = make_interval(first_rtp=rtp & 0xFFFFFFFF, pair=pair)
        want = rtp_to_utc(rtp & 0xFFFFFFFF, ci, wallclock_hint_sec=1_799_999_990.0)
        assert abs(tie(tmp_path / "x").label(iv).utc_ns / 1e9 - want) < 1e-6


def test_epochs_1p5_s_apart_are_the_same_run(anchor_file, interval):
    from conftest import EPOCH_S
    lab = tie(anchor_file).label(dataclasses.replace(interval, input_epoch_gps_s=EPOCH_S + 1.5))
    assert lab.delay_basis == "t6"


def test_mismatch_logs_on_each_change_of_state(anchor_file, anchor_doc, interval, caplog):
    t = tie(anchor_file, radiod_status="b1b-status.local")
    with caplog.at_level(logging.INFO):
        t.label(interval)
        t.label(interval)
        n1 = len(caplog.records)
        anchor_file.write_text(json.dumps(anchor_doc(radiod_status="b2-status.local")))
        t.label(interval)
        t.label(interval)
    assert n1 == 1 and len(caplog.records) == 2


def test_t6_returning_is_logged(anchor_file, interval, caplog):
    t = tie(anchor_file)
    other = dataclasses.replace(interval, input_epoch_gps_s=0.0)
    with caplog.at_level(logging.INFO):
        t.label(interval); t.label(other); t.label(other); t.label(interval)
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 2 and "another run" in msgs[0] and "applies again" in msgs[1]


def test_authority_read_failure_warns_once(tmp_path, interval, caplog):
    def boom():
        raise OSError("gone")
    t = tie(tmp_path / "x", authority=SimpleNamespace(read=boom))
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            assert t.label(interval).delay_basis == "radiod"
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_normalise_trailing_dot_and_ipv6():
    assert normalise_status("b4-status.local.") == "b4-status"
    assert normalise_status("ff02::1") != normalise_status("ff02::2")
    assert normalise_status("[ff02::1]:5006") == "ff02::1"
    assert normalise_status("192.168.1.5:5006") == "192.168.1.5"
