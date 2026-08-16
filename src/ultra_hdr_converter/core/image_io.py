"""Format-agnostic image I/O for the Ultra HDR pipeline.

This is the single dispatch point between the shared pipeline and the
per-container modules (:mod:`~ultra_hdr_converter.core.jpeg_io`,
:mod:`~ultra_hdr_converter.core.avif_io`).  Callers work in terms of
:class:`~ultra_hdr_converter.core.formats.ImageFormat` and never import a
container module directly, so adding a format means extending the dispatch
tables here.
"""

from __future__ import annotations

from pathlib import Path

import imagecodecs
import numpy as np

from ultra_hdr_converter.core import avif_io, jpeg_io
from ultra_hdr_converter.core.formats import ImageFormat, detect_format
from ultra_hdr_converter.errors import UnsupportedFormatError

NUMPY_SUFFIX = ".npy"


def read_bytes(path: Path | str) -> bytes:
    """Read file contents as bytes."""
    return Path(path).read_bytes()


def write_bytes(path: Path | str, data: bytes) -> None:
    """Write bytes to disk, creating parent folders if needed."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)


def load_gain_map(path: Path | str) -> np.ndarray:
    """Load a gain map from ``.npy`` or any image format imagecodecs can read."""
    source = Path(path)
    if source.suffix.lower() == NUMPY_SUFFIX:
        return np.asarray(np.load(source))
    return np.asarray(imagecodecs.imread(str(source)))


def decode_image(data: bytes, image_format: ImageFormat | None = None) -> np.ndarray:
    """Decode image bytes into a NumPy array.

    Args:
        data: Complete image file bytes.
        image_format: Container format; detected from the bytes when omitted.

    Returns:
        Decoded pixel array of shape (H, W), (H, W, 3) or (H, W, 4).

    Raises:
        UnsupportedFormatError: If the container format is not supported.
        ImageStructureError: If the bytes cannot be decoded.
    """
    match image_format or detect_format(data):
        case ImageFormat.JPEG:
            return jpeg_io.decode_jpeg(data)
        case ImageFormat.AVIF:
            return avif_io.decode_avif(data)
        case unsupported:
            raise UnsupportedFormatError(f"Cannot decode {unsupported.value} images.")


def extract_icc_profile(data: bytes, image_format: ImageFormat | None = None) -> bytes | None:
    """Extract the embedded ICC profile, if the file carries one.

    Args:
        data: Complete image file bytes.
        image_format: Container format; detected from the bytes when omitted.

    Returns:
        Raw ICC profile bytes, or ``None`` when absent.

    Raises:
        UnsupportedFormatError: If the container format is not supported.
    """
    match image_format or detect_format(data):
        case ImageFormat.JPEG:
            return jpeg_io.extract_icc_profile(data)
        case ImageFormat.AVIF:
            return avif_io.extract_icc_profile(data)
        case unsupported:
            raise UnsupportedFormatError(f"Cannot read ICC profiles from {unsupported.value} images.")


def has_ultrahdr_metadata(data: bytes, image_format: ImageFormat | None = None) -> bool:
    """Check whether the file already carries gain map metadata.

    Detects ISO 21496-1 metadata in either carriage: the APP2/XMP segments of an
    Ultra HDR JPEG, or the ``tmap`` derived image item of an AVIF file.

    Args:
        data: Complete image file bytes.
        image_format: Container format; detected from the bytes when omitted.

    Returns:
        True when the image is already gain map encoded.

    Raises:
        UnsupportedFormatError: If the container format is not supported.
    """
    match image_format or detect_format(data):
        case ImageFormat.JPEG:
            return jpeg_io.has_gain_map_metadata(data)
        case ImageFormat.AVIF:
            return avif_io.has_gain_map_metadata(data)
        case unsupported:
            raise UnsupportedFormatError(f"Cannot inspect {unsupported.value} images for gain map metadata.")


def extract_embedded_gain_map(data: bytes, image_format: ImageFormat | None = None) -> np.ndarray | None:
    """Extract a gain map that is embedded as an auxiliary image.

    Only JPEG files carry a gain map that this pipeline can adopt: an MPF
    secondary image without the accompanying XMP/ISO metadata. AVIF gain maps
    are always accompanied by a ``tmap`` item, so such files are reported as
    already encoded rather than reprocessed.

    Args:
        data: Complete image file bytes.
        image_format: Container format; detected from the bytes when omitted.

    Returns:
        The decoded gain map array, or ``None`` when none is embedded.

    Raises:
        UnsupportedFormatError: If the container format is not supported.
    """
    match image_format or detect_format(data):
        case ImageFormat.JPEG:
            mpf_bytes = jpeg_io.extract_mpf_gain_map(data)
            return None if mpf_bytes is None else jpeg_io.decode_jpeg(mpf_bytes)
        case ImageFormat.AVIF:
            return None
        case unsupported:
            raise UnsupportedFormatError(f"Cannot extract gain maps from {unsupported.value} images.")


def has_embedded_gain_map(data: bytes, image_format: ImageFormat | None = None) -> bool:
    """Check for an auxiliary gain map image that lacks gain map metadata."""
    match image_format or detect_format(data):
        case ImageFormat.JPEG:
            return jpeg_io.has_mpf_secondary_image(data)
        case ImageFormat.AVIF:
            return False
        case unsupported:
            raise UnsupportedFormatError(f"Cannot inspect {unsupported.value} images for embedded gain maps.")
