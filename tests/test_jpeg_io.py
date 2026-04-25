import struct

import imagecodecs
import pytest

from ultra_hdr_converter.core.jpeg_io import (
    decode_jpeg,
    extract_icc_profile,
    extract_mpf_gain_map,
    has_mpf_secondary_image,
    has_ultrahdr_metadata,
)
from ultra_hdr_converter.errors import JpegStructureError


def _make_app2_segment(payload: bytes) -> bytes:
    length = len(payload) + 2
    return b"\xff\xe2" + length.to_bytes(2, "big") + payload


def _make_jpeg_with_icc_chunks(chunks: list[tuple[int, int, bytes]]) -> bytes:
    signature = b"ICC_PROFILE\x00"
    segments = []
    for sequence_index, chunk_count, chunk_payload in chunks:
        payload = signature + bytes([sequence_index, chunk_count]) + chunk_payload
        segments.append(_make_app2_segment(payload))
    return b"\xff\xd8" + b"".join(segments) + b"\xff\xd9"


def _make_mpf_jpeg_with_secondary(secondary_data: bytes) -> bytes:
    """Build a minimal JPEG with an MPF segment pointing to the given secondary image."""
    # MP Header
    tiff_header = b"II" + struct.pack("<HI", 0x002A, 8)

    # MP Index IFD with 2 entries: NumberOfImages + MPEntry
    entry_count = 2
    ifd = struct.pack("<H", entry_count)

    # Tag 0xB001 = NumberOfImages, type=LONG, count=1, value=2
    ifd += struct.pack("<HHII", 0xB001, 4, 1, 2)

    # Tag 0xB002 = MPEntry, type=UNDEFINED, count=32
    entries_size = 32  # 2 images * 16 bytes
    mp_entry_data_offset = len(tiff_header) + 2 + entry_count * 12 + 4
    ifd += struct.pack("<HHII", 0xB002, 7, entries_size, mp_entry_data_offset)
    ifd += struct.pack("<I", 0)  # next IFD

    # MP Entries
    # Image 1: primary
    entries = struct.pack("<IIIHH", 0x02000000, 100, 0, 0, 0)
    # Image 2: secondary — offset relative to TIFF header
    data_offset = len(tiff_header) + len(ifd) + 32
    entries += struct.pack("<IIIHH", 0x00000000, len(secondary_data), data_offset, 0, 0)

    payload = b"MPF\x00" + tiff_header + ifd + entries + secondary_data

    return b"\xff\xd8" + _make_app2_segment(payload) + b"\xff\xd9"


# ── ICC profile extraction ────────────────────────────────────────────────────


def test_extract_icc_profile_uses_metadata_when_available(monkeypatch: object) -> None:
    monkeypatch.setattr(
        imagecodecs,
        "jpeg_metadata",
        lambda _payload: {"icc_profile": b"metadata-icc"},
        raising=False,
    )

    assert extract_icc_profile(b"dummy") == b"metadata-icc"


def test_extract_icc_profile_parses_app2_chunks_without_jpeg_metadata(monkeypatch: object) -> None:
    monkeypatch.delattr(imagecodecs, "jpeg_metadata", raising=False)

    jpeg_bytes = _make_jpeg_with_icc_chunks(
        [
            (2, 2, b"world"),
            (1, 2, b"hello "),
        ]
    )

    assert extract_icc_profile(jpeg_bytes) == b"hello world"


def test_extract_icc_profile_returns_none_for_incomplete_icc_chunks(monkeypatch: object) -> None:
    monkeypatch.delattr(imagecodecs, "jpeg_metadata", raising=False)

    jpeg_bytes = _make_jpeg_with_icc_chunks([(2, 2, b"chunk-two-only")])

    assert extract_icc_profile(jpeg_bytes) is None


# ── Ultra HDR metadata detection ──────────────────────────────────────────────


def test_has_ultrahdr_metadata_returns_true_for_iso_21496() -> None:
    payload = b"urn:iso:std:iso:ts:21496:-1\x00\x00\x00"
    jpeg_bytes = b"\xff\xd8" + _make_app2_segment(payload) + b"\xff\xd9"
    assert has_ultrahdr_metadata(jpeg_bytes) is True


def test_has_ultrahdr_metadata_returns_true_for_adobe_xmp() -> None:
    payload = b"http://ns.adobe.com/xap/1.0/\x00<x:xmpmeta>hdrgm:Version=</x:xmpmeta>"
    length = len(payload) + 2
    app1 = b"\xff\xe1" + length.to_bytes(2, "big") + payload
    jpeg_bytes = b"\xff\xd8" + app1 + b"\xff\xd9"
    assert has_ultrahdr_metadata(jpeg_bytes) is True


def test_has_ultrahdr_metadata_returns_true_for_hdrcapacitymin_xmp() -> None:
    """XMP with HDRCapacityMin but without 'hdrgm:' prefix should still be detected."""
    payload = b"http://ns.adobe.com/xap/1.0/\x00<x:xmpmeta>HDRCapacityMin=0.0</x:xmpmeta>"
    length = len(payload) + 2
    app1 = b"\xff\xe1" + length.to_bytes(2, "big") + payload
    jpeg_bytes = b"\xff\xd8" + app1 + b"\xff\xd9"
    assert has_ultrahdr_metadata(jpeg_bytes) is True


def test_has_ultrahdr_metadata_returns_true_for_gainmapmin_xmp() -> None:
    """XMP with GainMapMin but without 'hdrgm:' prefix should still be detected."""
    payload = b"http://ns.adobe.com/xap/1.0/\x00<x:xmpmeta>GainMapMin=0.0</x:xmpmeta>"
    length = len(payload) + 2
    app1 = b"\xff\xe1" + length.to_bytes(2, "big") + payload
    jpeg_bytes = b"\xff\xd8" + app1 + b"\xff\xd9"
    assert has_ultrahdr_metadata(jpeg_bytes) is True


def test_has_ultrahdr_metadata_returns_false_for_standard_jpeg() -> None:
    jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\xff\xd9"
    assert has_ultrahdr_metadata(jpeg_bytes) is False


def test_has_ultrahdr_metadata_returns_false_for_xmp_without_hdrgm() -> None:
    """XMP segment without any gain map attributes should not be detected as Ultra HDR."""
    payload = b"http://ns.adobe.com/xap/1.0/\x00<x:xmpmeta>some other metadata</x:xmpmeta>"
    length = len(payload) + 2
    app1 = b"\xff\xe1" + length.to_bytes(2, "big") + payload
    jpeg_bytes = b"\xff\xd8" + app1 + b"\xff\xd9"
    assert has_ultrahdr_metadata(jpeg_bytes) is False


def test_has_ultrahdr_metadata_returns_false_for_empty_input() -> None:
    assert has_ultrahdr_metadata(b"") is False


def test_has_ultrahdr_metadata_returns_false_for_non_jpeg() -> None:
    assert has_ultrahdr_metadata(b"\x89PNG\r\n\x1a\n") is False


def test_has_ultrahdr_metadata_returns_false_for_truncated_jpeg() -> None:
    assert has_ultrahdr_metadata(b"\xff\xd8\xff") is False


# ── MPF gain map extraction ──────────────────────────────────────────────────


def test_extract_mpf_gain_map_returns_none_if_no_mpf() -> None:
    jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\xff\xd9"
    assert extract_mpf_gain_map(jpeg_bytes) is None


def test_extract_mpf_gain_map_returns_secondary_image() -> None:
    secondary_data = b"\xff\xd8fake\xff\xd9"
    jpeg_bytes = _make_mpf_jpeg_with_secondary(secondary_data)
    extracted = extract_mpf_gain_map(jpeg_bytes)
    assert extracted == secondary_data


def test_extract_mpf_gain_map_returns_none_for_non_jpeg_secondary() -> None:
    """If the secondary image doesn't start with SOI, it should be rejected."""
    non_jpeg_secondary = b"NOT_A_JPEG_AT_ALL"
    jpeg_bytes = _make_mpf_jpeg_with_secondary(non_jpeg_secondary)
    assert extract_mpf_gain_map(jpeg_bytes) is None


def test_extract_mpf_gain_map_returns_none_for_empty_input() -> None:
    assert extract_mpf_gain_map(b"") is None


def test_extract_mpf_gain_map_returns_none_for_non_jpeg() -> None:
    assert extract_mpf_gain_map(b"\x89PNG\r\n\x1a\n") is None


def test_extract_mpf_gain_map_returns_none_for_truncated_segment() -> None:
    """Truncated segment length should not cause a crash."""
    jpeg_bytes = b"\xff\xd8\xff\xe2\x00"
    assert extract_mpf_gain_map(jpeg_bytes) is None


def test_extract_mpf_gain_map_returns_none_for_zero_size_entry() -> None:
    """MPF entry with size=0 should be rejected."""
    tiff_header = b"II" + struct.pack("<HI", 0x002A, 8)
    entry_count = 1
    ifd = struct.pack("<H", entry_count)
    entries_size = 32
    mp_entry_data_offset = len(tiff_header) + 2 + entry_count * 12 + 4
    ifd += struct.pack("<HHII", 0xB002, 7, entries_size, mp_entry_data_offset)
    ifd += struct.pack("<I", 0)

    # Primary entry
    entries = struct.pack("<IIIHH", 0x02000000, 100, 0, 0, 0)
    # Secondary entry with size=0
    entries += struct.pack("<IIIHH", 0x00000000, 0, 100, 0, 0)

    payload = b"MPF\x00" + tiff_header + ifd + entries
    jpeg_bytes = b"\xff\xd8" + _make_app2_segment(payload) + b"\xff\xd9"
    assert extract_mpf_gain_map(jpeg_bytes) is None


def test_extract_mpf_gain_map_returns_none_for_invalid_byte_order() -> None:
    """MPF TIFF header with invalid byte order should be rejected."""
    tiff_header = b"XX" + struct.pack("<HI", 0x002A, 8)  # Invalid BOM
    payload = b"MPF\x00" + tiff_header
    jpeg_bytes = b"\xff\xd8" + _make_app2_segment(payload) + b"\xff\xd9"
    assert extract_mpf_gain_map(jpeg_bytes) is None


def test_extract_mpf_gain_map_returns_none_when_count_is_one() -> None:
    """If NumberOfImages is explicitly 1, there's no secondary image."""
    tiff_header = b"II" + struct.pack("<HI", 0x002A, 8)
    entry_count = 2
    ifd = struct.pack("<H", entry_count)

    # NumberOfImages = 1
    ifd += struct.pack("<HHII", 0xB001, 4, 1, 1)

    entries_size = 16
    mp_entry_data_offset = len(tiff_header) + 2 + entry_count * 12 + 4
    ifd += struct.pack("<HHII", 0xB002, 7, entries_size, mp_entry_data_offset)
    ifd += struct.pack("<I", 0)

    entries = struct.pack("<IIIHH", 0x02000000, 100, 0, 0, 0)
    payload = b"MPF\x00" + tiff_header + ifd + entries
    jpeg_bytes = b"\xff\xd8" + _make_app2_segment(payload) + b"\xff\xd9"
    assert extract_mpf_gain_map(jpeg_bytes) is None


def test_extract_mpf_gain_map_handles_big_endian_tiff() -> None:
    """Verify big-endian (Motorola) byte order is parsed correctly."""
    secondary_data = b"\xff\xd8BIGEND\xff\xd9"

    tiff_header = b"MM" + struct.pack(">HI", 0x002A, 8)
    entry_count = 2
    ifd = struct.pack(">H", entry_count)

    # NumberOfImages = 2
    ifd += struct.pack(">HHII", 0xB001, 4, 1, 2)

    entries_size = 32
    mp_entry_data_offset = len(tiff_header) + 2 + entry_count * 12 + 4
    ifd += struct.pack(">HHII", 0xB002, 7, entries_size, mp_entry_data_offset)
    ifd += struct.pack(">I", 0)

    entries = struct.pack(">IIIHH", 0x02000000, 100, 0, 0, 0)
    data_offset = len(tiff_header) + len(ifd) + 32
    entries += struct.pack(">IIIHH", 0x00000000, len(secondary_data), data_offset, 0, 0)

    payload = b"MPF\x00" + tiff_header + ifd + entries + secondary_data
    jpeg_bytes = b"\xff\xd8" + _make_app2_segment(payload) + b"\xff\xd9"

    extracted = extract_mpf_gain_map(jpeg_bytes)
    assert extracted == secondary_data


# ── has_mpf_secondary_image ──────────────────────────────────────────────────


def test_has_mpf_secondary_image_returns_true_when_present() -> None:
    secondary_data = b"\xff\xd8test\xff\xd9"
    jpeg_bytes = _make_mpf_jpeg_with_secondary(secondary_data)
    assert has_mpf_secondary_image(jpeg_bytes) is True


def test_has_mpf_secondary_image_returns_false_for_plain_jpeg() -> None:
    jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\xff\xd9"
    assert has_mpf_secondary_image(jpeg_bytes) is False


# ── Backward-compat: the old test for extract_mpf_gain_map_returns_secondary_image
# is now superseded by the new helper above, but the original test format is preserved.

def test_extract_mpf_gain_map_returns_secondary_image_legacy_format() -> None:
    """Original test with the initial MPF builder format (1 IFD entry only, no NumberOfImages)."""
    secondary_data = b"\xff\xd8fake\xff\xd9"

    tiff_header = b"II" + struct.pack("<HI", 0x002A, 8)
    entry_count = 1
    ifd = struct.pack("<H", entry_count)

    entries_size = 32
    mp_entry_data_offset = len(tiff_header) + 2 + entry_count * 12 + 4
    ifd += struct.pack("<HHII", 0xB002, 7, entries_size, mp_entry_data_offset)
    ifd += struct.pack("<I", 0)

    entries = struct.pack("<IIIHH", 0x02000000, 100, 0, 0, 0)
    data_offset = len(tiff_header) + len(ifd) + 32
    entries += struct.pack("<IIIHH", 0x00000000, len(secondary_data), data_offset, 0, 0)

    payload = b"MPF\x00" + tiff_header + ifd + entries + secondary_data
    jpeg_bytes = b"\xff\xd8" + _make_app2_segment(payload) + b"\xff\xd9"

    extracted = extract_mpf_gain_map(jpeg_bytes)
    assert extracted == secondary_data


def test_decode_jpeg_raises_structure_error_on_invalid_data() -> None:
    """Invalid JPEG bytes should raise JpegStructureError."""
    with pytest.raises(JpegStructureError, match="JPEG decoding failed"):
        decode_jpeg(b"not a jpeg at all")
