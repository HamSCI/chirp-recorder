"""sigmond Receiver Channels TUI parser for chirp-recorder.

sigmond loads this file at TUI time through [client_features.receiver_channels]
in deploy.toml.  It is a pure function over a parsed instance config and imports
nothing from chirp_recorder.
"""
from __future__ import annotations

from typing import Optional

try:
    from sigmond.ka9q_encoding import ENCODING_INTS, encoding_to_int
except ImportError:      # sigmond absent (tests, standalone): raw IQ asks for f32
    ENCODING_INTS = {"f32": 4}

    def encoding_to_int(enc):  # type: ignore[misc]
        return ENCODING_INTS.get(str(enc).strip().lower())


def parse_receiver_channels(cfg: dict) -> tuple[str, set[int], Optional[int]]:
    """Return (status_dns, configured_freqs_hz, encoding_int) from an instance config.

    chirp-recorder lays out one [[channel]] block per radiod channel, each with
    its own `status` and `centre_hz`.  The raw-channel source asks radiod for
    complex F32 IQ, so encoding 4 is the default; a block's `encoding` key
    overrides it."""
    blocks = cfg.get("channel") or []
    if isinstance(blocks, dict):
        blocks = [blocks]
    status = ""
    freqs: set[int] = set()
    encoding: Optional[int] = None
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if not status:
            status = str(b.get("status") or "")
        hz = b.get("centre_hz")
        if hz is not None and not isinstance(hz, bool):
            try:
                freqs.add(int(hz))
            except (TypeError, ValueError):
                pass
        if encoding is None and b.get("encoding"):
            encoding = encoding_to_int(b["encoding"])
    if encoding is None:
        encoding = ENCODING_INTS["f32"]
    return status, freqs, encoding
