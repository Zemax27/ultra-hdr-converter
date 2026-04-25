import struct

import numpy as np
import pytest

from ultra_hdr_converter.core.ultrahdr_encoder import (
    _find_injection_point,
    _inject_after_soi,
    _strip_mpf_segments,
    encode_ultrahdr,
)
from ultra_hdr_converter.errors import JpegStructureError

_MINIMAL_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"

_ISO_NAMESPACE = b"urn:iso:std:iso:ts:21496:-1\x00"

_FAKE_GM_MARKER = b"FAKEGM"


def test_encode_ultrahdr_produces_valid_structure(monkeypatch: object) -> None:
    def _fake_jpeg_encode(data: np.ndarray, **_kwargs: object) -> bytes:
        return b"\xff\xd8GAINMAP\xff\xd9"

    monkeypatch.setattr(
        "ultra_hdr_converter.core.ultrahdr_encoder.imagecodecs.jpeg_encode",
        _fake_jpeg_encode,
    )

    gain_map = np.full((4, 4), 128, dtype=np.uint8)
    result = encode_ultrahdr(sdr_jpeg=_MINIMAL_JPEG, gain_map=gain_map)

    assert result[:2] == b"\xff\xd8"
    assert b"hdrgm:Version" in result
    assert b"hdrgm:GainMapMax" in result
    assert b"MPF\x00" in result
    assert _ISO_NAMESPACE in result
    assert result.endswith(b"GAINMAP\xff\xd9")


def test_encode_ultrahdr_preserves_original_jpeg(monkeypatch: object) -> None:
    def _fake_jpeg_encode(data: np.ndarray, **_kwargs: object) -> bytes:
        return b"\xff\xd8GM\xff\xd9"

    monkeypatch.setattr(
        "ultra_hdr_converter.core.ultrahdr_encoder.imagecodecs.jpeg_encode",
        _fake_jpeg_encode,
    )

    gain_map = np.zeros((2, 2), dtype=np.uint8)
    result = encode_ultrahdr(sdr_jpeg=_MINIMAL_JPEG, gain_map=gain_map)

    assert b"JFIF" in result


def test_encode_ultrahdr_mpf_offset_is_consistent(monkeypatch: object) -> None:
    def _fake_jpeg_encode(data: np.ndarray, **_kwargs: object) -> bytes:
        return b"\xff\xd8" + _FAKE_GM_MARKER + b"\xff\xd9"

    monkeypatch.setattr(
        "ultra_hdr_converter.core.ultrahdr_encoder.imagecodecs.jpeg_encode",
        _fake_jpeg_encode,
    )

    gain_map = np.zeros((2, 2), dtype=np.uint8)
    result = encode_ultrahdr(sdr_jpeg=_MINIMAL_JPEG, gain_map=gain_map)

    gm_iso_pos = result.rfind(_ISO_NAMESPACE)
    assert gm_iso_pos > 0

    gm_start = result.rfind(b"\xff\xd8", 0, gm_iso_pos)
    assert gm_start > 0

    mpf_pos = result.find(b"MPF\x00")
    assert mpf_pos > 0

    ii_offset = mpf_pos + 4
    assert result[ii_offset : ii_offset + 2] == b"II"

    second_entry_offset_pos = ii_offset + 8 + 42 + 16 + 8
    stored_offset = struct.unpack("<I", result[second_entry_offset_pos : second_entry_offset_pos + 4])[0]

    assert ii_offset + stored_offset == gm_start


def test_encode_ultrahdr_iso_segments(monkeypatch: object) -> None:
    def _fake_jpeg_encode(data: np.ndarray, **_kwargs: object) -> bytes:
        return b"\xff\xd8" + _FAKE_GM_MARKER + b"\xff\xd9"

    monkeypatch.setattr(
        "ultra_hdr_converter.core.ultrahdr_encoder.imagecodecs.jpeg_encode",
        _fake_jpeg_encode,
    )

    gain_map = np.zeros((2, 2), dtype=np.uint8)
    result = encode_ultrahdr(sdr_jpeg=_MINIMAL_JPEG, gain_map=gain_map)

    gm_marker_pos = result.rfind(_FAKE_GM_MARKER)
    gm_start = result.rfind(b"\xff\xd8", 0, gm_marker_pos)
    primary = result[:gm_start]
    gainmap_jpeg = result[gm_start:]

    iso_pos_primary = primary.find(_ISO_NAMESPACE)
    assert iso_pos_primary > 0
    version_data = primary[iso_pos_primary + len(_ISO_NAMESPACE) : iso_pos_primary + len(_ISO_NAMESPACE) + 4]
    assert version_data == b"\x00\x00\x00\x00"

    iso_pos_gm = gainmap_jpeg.find(_ISO_NAMESPACE)
    assert iso_pos_gm >= 0
    gm_metadata_start = iso_pos_gm + len(_ISO_NAMESPACE)
    iso_metadata_min_size = 4 + 1 + 8 * 8
    assert len(gainmap_jpeg) - gm_metadata_start >= iso_metadata_min_size


def test_inject_after_soi_raises_on_invalid_jpeg() -> None:
    with pytest.raises(JpegStructureError, match="SOI marker"):
        _inject_after_soi(b"\x00\x00", b"\x00")


def test_find_injection_point_raises_on_invalid_jpeg() -> None:
    with pytest.raises(JpegStructureError, match="SOI marker"):
        _find_injection_point(b"\x00\x00")


def test_strip_mpf_segments_raises_on_invalid_jpeg() -> None:
    with pytest.raises(JpegStructureError, match="SOI marker"):
        _strip_mpf_segments(b"\x00\x00")
