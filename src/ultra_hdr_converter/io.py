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


def extract_icc_profile(jpeg_bytes: bytes) -> bytes | None:
    """Extract embedded ICC profile bytes from JPEG metadata, if present."""
    metadata = imagecodecs.jpeg_metadata(jpeg_bytes)
    if not isinstance(metadata, dict):
        return None

    icc_profile = metadata.get("icc_profile")
    if isinstance(icc_profile, (bytes, bytearray, memoryview)):
        return bytes(icc_profile)
    return None


def load_gain_map(path: Path | str) -> np.ndarray:
    """Load a gain map from .npy or standard image codecs."""
    source = Path(path)
    if source.suffix.lower() == ".npy":
        return np.load(source)
    return np.asarray(imagecodecs.imread(str(source)))
