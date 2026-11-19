"""One interval's sigmondb rows for one transmitter group.

An arrival's delay is its phase within the sweep period, counted from the UTC
epoch: split_arrival(label.utc_ns + tau_ns, period).  The label carries the
timing basis and provenance block.  The rows hold no precise coordinates.
"""
from __future__ import annotations

from hamsci_dsp import sigmondb_rows as sr



def _s(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _g(x: float) -> str:
    return f"{x:.10g}"


def rows_for_group(result, *, label, reporter_id: str, radiod: str, channel: str,
                   duration_s: float, detector_version: str, input_epoch_gps_s: float,
                   interval_s: float, position_source: str,
                   ring_ref: str = "") -> tuple[dict, list[dict]]:
    """(capture_row, [arrival_rows]) for one group's result on one interval."""
    wf = result.group.waveform
    period_s = wf.period_s
    centre_hz = float(wf.centre_hz)
    rate = float(wf.rate_hz_per_s)
    gkey = sr.group_key(centre_hz, rate, period_s, wf.sweep_s)
    gate_s = result.gate.period_s if result.gate is not None else wf.gate_period_s
    eng = {k: _s(v) for k, v in label.engineering.items()}
    eng.update(radiod=radiod, channel=channel,
               input_epoch_gps_s=repr(float(input_epoch_gps_s)),
               interval_s=_g(float(interval_s)),
               sweep_repetition_hz=_g(1.0 / period_s),
               gate_period_us=_g(gate_s * 1e6) if gate_s else "",
               gate_phase_us="0", position_source=position_source)
    prov = dict(label.provenance)

    arrivals = []
    for a in result.arrivals:
        second, delay_ns = sr.split_arrival(int(label.utc_ns) + int(a.tau_ns), period_s)
        arrivals.append(sr.arrival_row(
            time=second, reporter_id=reporter_id, tx_id="", group_key=gkey,
            centre_hz=centre_hz, rate_hz_per_s=rate, delay_ns=delay_ns,
            delay_sigma_ns=a.delay_sigma_ns, freq_hz=centre_hz, level_db=a.level_db,
            margin_db=a.margin_db, half_band_db=a.half_band_db, clean_tag=a.clean_tag,
            doppler_hz=a.doppler_hz, phase_rate=a.phase_rate,
            carrier_phase_rad=a.carrier_phase_rad, s4=a.s4, sigma_phi_rad=a.sigma_phi_rad,
            detector_version=detector_version,
            engineering={**eng, "tau_uncorrected_ns": _s(int(a.tau_uncorrected_ns)),
                         "overlap_flag": _s(bool(a.overlap_flag)),
                         "doppler_ambiguous": _s(bool(a.doppler_ambiguous))},
            **prov))

    tx_ids = result.group.tx_ids
    rej = result.rejected or {}
    capture = sr.capture_row(
        time=sr.utc_iso_ms(label.utc_ns / 1e9), reporter_id=reporter_id,
        centre_hz=centre_hz, rate_hz_per_s=rate, duration_s=float(duration_s),
        n_sweeps=int(result.n_sweeps), expected={t: 1.0 for t in tx_ids},
        null_max_db=float(result.null_max_db), null_median_db=float(result.null_median_db),
        n_expected=len(tx_ids), n_found=len(arrivals), detector_version=detector_version,
        ring_ref=ring_ref,
        engineering={**eng, "rejected_rung": _s(int(rej.get("rung", 0))),
                     "rejected_image": _s(int(rej.get("image", 0))),
                     "rejected_null": _s(int(rej.get("null", 0)))},
        **prov)
    return capture, arrivals
