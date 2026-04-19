"""I/O and decoding helpers for SDR and gain map assets."""

from __future__ import annotations

from pathlib import Path

import imagecodecs
import numpy as np


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


def _extract_icc_from_jpeg_app2(jpeg_bytes: bytes) -> bytes | None:
    """Extract ICC profile from JPEG APP2 markers according to the ICC JPEG convention."""
    if len(jpeg_bytes) < 4 or jpeg_bytes[:2] != b"\xff\xd8":
        return None

    offset = 2
    chunks: dict[int, bytes] = {}
    expected_count: int | None = None

    while offset + 1 < len(jpeg_bytes):
        if jpeg_bytes[offset] != 0xFF:
            offset += 1
            continue

        while offset < len(jpeg_bytes) and jpeg_bytes[offset] == 0xFF:
            offset += 1

        if offset >= len(jpeg_bytes):
            break

        marker = jpeg_bytes[offset]
        offset += 1

        if marker == 0xDA:  # Start of Scan
            break
        if marker == 0xD9:  # End of Image
            break
        if marker == 0x00 or 0xD0 <= marker <= 0xD7 or marker == 0x01:
            continue

        if offset + 2 > len(jpeg_bytes):
            break

        segment_length = int.from_bytes(jpeg_bytes[offset : offset + 2], "big")
        offset += 2
        if segment_length < 2:
            break

        payload_length = segment_length - 2
        if offset + payload_length > len(jpeg_bytes):
            break

        payload = jpeg_bytes[offset : offset + payload_length]
        offset += payload_length

        if marker != 0xE2:
            continue

        signature = b"ICC_PROFILE\x00"
        if len(payload) < len(signature) + 2 or not payload.startswith(signature):
            continue

        sequence_index = int(payload[len(signature)])
        count = int(payload[len(signature) + 1])
        if sequence_index < 1 or count < 1:
            continue

        if expected_count is None:
            expected_count = count
        elif expected_count != count:
            return None

        chunk = payload[len(signature) + 2 :]
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
