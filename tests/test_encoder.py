import struct

import numpy as np

from ultra_hdr_converter.encoder import encode_ultrahdr

# Minimal valid JPEG: SOI + APP0 (JFIF stub) + EOI.
_MINIMAL_JPEG = (
    b"\xff\xd8"  # SOI
    b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"  # APP0
    b"\xff\xd9"  # EOI
)

# ISO 21496-1 namespace identifier.
_ISO_NAMESPACE = b"urn:iso:std:iso:ts:21496:-1\x00"

# Unique marker injected into fake gain map payloads for easy searching.
_FAKE_GM_MARKER = b"FAKEGM"


def test_encode_ultrahdr_produces_valid_structure(monkeypatch: object) -> None:
    """The output must start with JPEG SOI, contain XMP gain map metadata,
    contain an MPF APP2 segment, ISO 21496-1 APP2 segments, and end with
    the gain map JPEG."""

    def _fake_jpeg_encode(data: np.ndarray, **_kwargs: object) -> bytes:
        return b"\xff\xd8GAINMAP\xff\xd9"

    monkeypatch.setattr(
        "ultra_hdr_converter.encoder.imagecodecs.jpeg_encode",
        _fake_jpeg_encode,
    )

    gain_map = np.full((4, 4), 128, dtype=np.uint8)
    result = encode_ultrahdr(sdr_jpeg=_MINIMAL_JPEG, gain_map=gain_map)

    # Starts with SOI.
    assert result[:2] == b"\xff\xd8"

    # Contains XMP with gain map namespace.
    assert b"hdrgm:Version" in result
    assert b"hdrgm:GainMapMax" in result

    # Contains MPF APP2.
    assert b"MPF\x00" in result

    # Contains ISO 21496-1 APP2 namespace.
    assert _ISO_NAMESPACE in result

    # The gain map payload survives (ISO segment is injected between SOI
    # and the raw payload, so check for the marker instead of exact bytes).
    assert result.endswith(b"GAINMAP\xff\xd9")


def test_encode_ultrahdr_preserves_original_jpeg(monkeypatch: object) -> None:
    """The original SDR JPEG content must appear in the output."""

    def _fake_jpeg_encode(data: np.ndarray, **_kwargs: object) -> bytes:
        return b"\xff\xd8GM\xff\xd9"

    monkeypatch.setattr(
        "ultra_hdr_converter.encoder.imagecodecs.jpeg_encode",
        _fake_jpeg_encode,
    )

    gain_map = np.zeros((2, 2), dtype=np.uint8)
    result = encode_ultrahdr(sdr_jpeg=_MINIMAL_JPEG, gain_map=gain_map)

    # The original JFIF APP0 payload must survive in the output.
    assert b"JFIF" in result


def test_encode_ultrahdr_mpf_offset_is_consistent(monkeypatch: object) -> None:
    """The MPF gain map offset must point to the actual gain map position."""

    def _fake_jpeg_encode(data: np.ndarray, **_kwargs: object) -> bytes:
        return b"\xff\xd8" + _FAKE_GM_MARKER + b"\xff\xd9"

    monkeypatch.setattr(
        "ultra_hdr_converter.encoder.imagecodecs.jpeg_encode",
        _fake_jpeg_encode,
    )

    gain_map = np.zeros((2, 2), dtype=np.uint8)
    result = encode_ultrahdr(sdr_jpeg=_MINIMAL_JPEG, gain_map=gain_map)

    # Find the gain map SOI: the second SOI in the output (first is the
    # primary JPEG).  The ISO segment is injected after SOI, so search for
    # the ISO namespace that begins the gain map's APP2 payload.
    gm_iso_pos = result.rfind(_ISO_NAMESPACE)
    assert gm_iso_pos > 0

    # Walk back to the SOI that precedes this ISO segment:
    # SOI(2) + APP2 marker(2) + length(2) + namespace... so SOI is before
    # the APP2 marker.  Find the gain map SOI by scanning backwards.
    gm_start = result.rfind(b"\xff\xd8", 0, gm_iso_pos)
    assert gm_start > 0

    # Find the MPF APP2 segment and extract the gain map data offset.
    mpf_pos = result.find(b"MPF\x00")
    assert mpf_pos > 0

    # The MP Header "II" starts right after "MPF\0".
    ii_offset = mpf_pos + 4
    assert result[ii_offset : ii_offset + 2] == b"II"

    # Read the second MP Entry (gain map) data offset (little-endian uint32).
    # MP Entry data starts at: II + tiff_header(8) + IFD(42) + first_entry(16) + offset_field(8).
    # Second entry starts at: II + 8 + 42 + 16 = II + 66.
    # Its data-offset field is at byte 8 within the entry.
    second_entry_offset_pos = ii_offset + 8 + 42 + 16 + 8
    stored_offset = struct.unpack("<I", result[second_entry_offset_pos : second_entry_offset_pos + 4])[0]

    # The stored offset is relative to the "II" position.
    assert ii_offset + stored_offset == gm_start


def test_encode_ultrahdr_iso_segments(monkeypatch: object) -> None:
    """Primary JPEG must have ISO version-only APP2; gain map must have full
    ISO 21496-1 binary metadata APP2."""

    def _fake_jpeg_encode(data: np.ndarray, **_kwargs: object) -> bytes:
        return b"\xff\xd8" + _FAKE_GM_MARKER + b"\xff\xd9"

    monkeypatch.setattr(
        "ultra_hdr_converter.encoder.imagecodecs.jpeg_encode",
        _fake_jpeg_encode,
    )

    gain_map = np.zeros((2, 2), dtype=np.uint8)
    result = encode_ultrahdr(sdr_jpeg=_MINIMAL_JPEG, gain_map=gain_map)

    # Split output into primary and gain map at the second SOI.
    # Find gain map start: last occurrence of SOI before FAKEGM marker.
    gm_marker_pos = result.rfind(_FAKE_GM_MARKER)
    gm_start = result.rfind(b"\xff\xd8", 0, gm_marker_pos)
    primary = result[:gm_start]
    gainmap_jpeg = result[gm_start:]

    # Primary: ISO namespace present with version-only payload (4 zero bytes).
    iso_pos_primary = primary.find(_ISO_NAMESPACE)
    assert iso_pos_primary > 0
    version_data = primary[iso_pos_primary + len(_ISO_NAMESPACE) : iso_pos_primary + len(_ISO_NAMESPACE) + 4]
    assert version_data == b"\x00\x00\x00\x00"

    # Gain map: ISO namespace present with full binary metadata (longer payload).
    iso_pos_gm = gainmap_jpeg.find(_ISO_NAMESPACE)
    assert iso_pos_gm >= 0
    # Full metadata has version(4) + flags(1) + at least 8 rational fields = much more than 4 bytes.
    gm_metadata_start = iso_pos_gm + len(_ISO_NAMESPACE)
    iso_metadata_min_size = 4 + 1 + 8 * 8  # version + flags + 8 rationals (N+D uint32 each)
    assert len(gainmap_jpeg) - gm_metadata_start >= iso_metadata_min_size
