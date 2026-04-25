import imagecodecs

from ultra_hdr_converter.core.jpeg_io import extract_icc_profile, has_ultrahdr_metadata, extract_mpf_gain_map


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


def test_has_ultrahdr_metadata_returns_false_for_standard_jpeg() -> None:
    jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\xff\xd9"
    assert has_ultrahdr_metadata(jpeg_bytes) is False


def test_extract_mpf_gain_map_returns_none_if_no_mpf() -> None:
    jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\xff\xd9"
    assert extract_mpf_gain_map(jpeg_bytes) is None


def test_extract_mpf_gain_map_returns_secondary_image() -> None:
    import struct
    
    # We will forge a minimal MPF payload containing 2 images.
    # The primary image size won't really matter since we just need the parser to find the second entry.
    # Secondary image data will be just some fake bytes.
    secondary_data = b"\xff\xd8fake\xff\xd9"
    
    # MP Header
    tiff_header_offset = 0
    tiff_header = b"II" + struct.pack("<HI", 0x002A, 8)
    
    # MP Index IFD
    entry_count = 1
    ifd = struct.pack("<H", entry_count)
    
    # MPEntry Tag: 0xB002
    entries_size = 32 # 2 images * 16 bytes
    mp_entry_data_offset = len(tiff_header) + 2 + entry_count * 12 + 4
    ifd += struct.pack("<HHII", 0xB002, 7, entries_size, mp_entry_data_offset)
    ifd += struct.pack("<I", 0) # next ifd
    
    # Entries
    # Image 1
    entries = struct.pack("<IIIHH", 0x02000000, 100, 0, 0, 0)
    # Image 2
    data_offset = len(tiff_header) + len(ifd) + 32
    entries += struct.pack("<IIIHH", 0x00000000, len(secondary_data), data_offset, 0, 0)
    
    payload = b"MPF\x00" + tiff_header + ifd + entries + secondary_data
    
    jpeg_bytes = b"\xff\xd8" + _make_app2_segment(payload) + b"\xff\xd9"
    
    extracted = extract_mpf_gain_map(jpeg_bytes)
    assert extracted == secondary_data
