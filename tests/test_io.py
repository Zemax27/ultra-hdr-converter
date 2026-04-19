import imagecodecs

from ultra_hdr_converter.io import extract_icc_profile


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
