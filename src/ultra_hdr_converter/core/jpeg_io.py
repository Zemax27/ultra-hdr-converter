"""I/O and decoding helpers for SDR and gain map assets."""

from __future__ import annotations

from pathlib import Path

import imagecodecs
import numpy as np

JPEG_SOI = b"\xff\xd8"
JPEG_MIN_BYTES = 4
MARKER_PREFIX = 0xFF
START_OF_SCAN_MARKER = 0xDA
END_OF_IMAGE_MARKER = 0xD9
APP2_MARKER = 0xE2
MARKER_STUFFED_ZERO = 0x00
MARKER_TEM = 0x01
RST_MARKER_MIN = 0xD0
RST_MARKER_MAX = 0xD7
SEGMENT_LENGTH_BYTES = 2
MIN_SEGMENT_LENGTH = 2
ICC_SIGNATURE = b"ICC_PROFILE\x00"
ICC_HEADER_BYTES = 2
ICC_MIN_SEQUENCE = 1


def read_bytes(path: Path | str) -> bytes:
    """Read file contents as bytes."""
    return Path(path).read_bytes()


def write_bytes(path: Path | str, data: bytes) -> None:
    """Write bytes to disk, creating parent folders if needed."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)


def decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes into a NumPy array."""
    return np.asarray(imagecodecs.jpeg_decode(jpeg_bytes))


def _skip_marker_prefixes(jpeg_bytes: bytes, offset: int) -> int:
    """Advance offset past repeated 0xFF marker prefix bytes."""
    while offset < len(jpeg_bytes) and jpeg_bytes[offset] == MARKER_PREFIX:
        offset += 1
    return offset


def _is_standalone_marker(marker: int) -> bool:
    """Return True for markers that are not followed by a payload length field."""
    if marker in {MARKER_STUFFED_ZERO, MARKER_TEM}:
        return True
    return RST_MARKER_MIN <= marker <= RST_MARKER_MAX


def _iter_jpeg_segments(jpeg_bytes: bytes) -> list[tuple[int, bytes]]:
    """Iterate marker/payload segments until SOS or EOI."""
    if len(jpeg_bytes) < JPEG_MIN_BYTES or jpeg_bytes[:SEGMENT_LENGTH_BYTES] != JPEG_SOI:
        return []

    offset = SEGMENT_LENGTH_BYTES
    segments: list[tuple[int, bytes]] = []

    while offset + 1 < len(jpeg_bytes):
        if jpeg_bytes[offset] != MARKER_PREFIX:
            offset += 1
            continue

        offset = _skip_marker_prefixes(jpeg_bytes, offset)
        if offset >= len(jpeg_bytes):
            break

        marker = jpeg_bytes[offset]
        offset += 1

        if marker in {START_OF_SCAN_MARKER, END_OF_IMAGE_MARKER}:
            break
        if _is_standalone_marker(marker):
            continue

        if offset + SEGMENT_LENGTH_BYTES > len(jpeg_bytes):
            break

        segment_length = int.from_bytes(jpeg_bytes[offset : offset + SEGMENT_LENGTH_BYTES], "big")
        offset += SEGMENT_LENGTH_BYTES
        if segment_length < MIN_SEGMENT_LENGTH:
            break

        payload_length = segment_length - MIN_SEGMENT_LENGTH
        if offset + payload_length > len(jpeg_bytes):
            break

        payload = jpeg_bytes[offset : offset + payload_length]
        offset += payload_length
        segments.append((marker, payload))

    return segments


def _parse_icc_app2_payload(payload: bytes) -> tuple[int, int, bytes] | None:
    """Parse ICC APP2 payload header and return sequence metadata plus chunk bytes."""
    signature_length = len(ICC_SIGNATURE)
    minimum_payload = signature_length + ICC_HEADER_BYTES
    if len(payload) < minimum_payload or not payload.startswith(ICC_SIGNATURE):
        return None

    sequence_index = int(payload[signature_length])
    chunk_count = int(payload[signature_length + 1])
    if sequence_index < ICC_MIN_SEQUENCE or chunk_count < ICC_MIN_SEQUENCE:
        return None

    chunk = payload[minimum_payload:]
    return sequence_index, chunk_count, chunk


def _extract_icc_from_jpeg_app2(jpeg_bytes: bytes) -> bytes | None:
    """Extract ICC profile from JPEG APP2 markers according to the ICC JPEG convention."""
    chunks: dict[int, bytes] = {}
    expected_count: int | None = None

    for marker, payload in _iter_jpeg_segments(jpeg_bytes):
        if marker != APP2_MARKER:
            continue

        parsed = _parse_icc_app2_payload(payload)
        if parsed is None:
            continue

        sequence_index, count, chunk = parsed

        if expected_count is None:
            expected_count = count
        elif expected_count != count:
            return None

        chunks[sequence_index] = chunk

    if not chunks:
        return None

    if expected_count is None:
        expected_count = len(chunks)

    if any(index not in chunks for index in range(1, expected_count + 1)):
        return None

    return b"".join(chunks[index] for index in range(1, expected_count + 1))


def extract_icc_profile(jpeg_bytes: bytes) -> bytes | None:
    """Extract embedded ICC profile bytes from JPEG payload, if present."""
    jpeg_metadata = getattr(imagecodecs, "jpeg_metadata", None)
    if callable(jpeg_metadata):
        metadata = jpeg_metadata(jpeg_bytes)
        if isinstance(metadata, dict):
            icc_profile = metadata.get("icc_profile")
            if isinstance(icc_profile, (bytes, bytearray, memoryview)):
                return bytes(icc_profile)

    return _extract_icc_from_jpeg_app2(jpeg_bytes)


def load_gain_map(path: Path | str) -> np.ndarray:
    """Load a gain map from .npy or standard image codecs."""
    source = Path(path)
    if source.suffix.lower() == ".npy":
        return np.asarray(np.load(source))
    return np.asarray(imagecodecs.imread(str(source)))
