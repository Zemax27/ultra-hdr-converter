"""Container format identification for the Ultra HDR pipeline.

The pipeline is container-agnostic: the gain map generation core is shared and
only the packaging step differs per format.  This module is the single place
that knows which containers exist, how to recognise them and which file name
suffixes map to them.  Adding a new container means adding one enum member and
its suffix/magic entries here.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from ultra_hdr_converter.errors import UnsupportedFormatError

# Minimum number of bytes required to identify any supported container.
MAGIC_PROBE_BYTES = 12

_JPEG_SOI = b"\xff\xd8\xff"
_ISOBMFF_FTYP = b"ftyp"
_AVIF_BRANDS = frozenset({b"avif", b"avis", b"mif1", b"miaf", b"MA1A", b"MA1B"})


class ImageFormat(Enum):
    """Supported image container formats."""

    JPEG = "jpeg"
    AVIF = "avif"

    @property
    def default_suffix(self) -> str:
        """Canonical file name suffix used when writing this format."""
        return _DEFAULT_SUFFIXES[self]

    @property
    def suffixes(self) -> frozenset[str]:
        """All lowercase file name suffixes recognised for this format."""
        return _SUFFIXES[self]


_DEFAULT_SUFFIXES: dict[ImageFormat, str] = {
    ImageFormat.JPEG: ".jpg",
    ImageFormat.AVIF: ".avif",
}

_SUFFIXES: dict[ImageFormat, frozenset[str]] = {
    ImageFormat.JPEG: frozenset({".jpg", ".jpeg"}),
    ImageFormat.AVIF: frozenset({".avif"}),
}

FORMAT_BY_SUFFIX: dict[str, ImageFormat] = {
    suffix: image_format for image_format, suffixes in _SUFFIXES.items() for suffix in suffixes
}

SUPPORTED_SUFFIXES: frozenset[str] = frozenset(FORMAT_BY_SUFFIX)

# Container assumed when nothing else identifies one — for example when naming
# the output for a file whose suffix is unrecognised.  JPEG is chosen because it
# is the most broadly readable of the supported containers.
DEFAULT_FORMAT = ImageFormat.JPEG


def format_from_suffix(path: Path | str) -> ImageFormat | None:
    """Identify the container format from a file name suffix.

    Args:
        path: File path whose suffix is inspected.

    Returns:
        The matching format, or ``None`` when the suffix is not recognised.
    """
    return FORMAT_BY_SUFFIX.get(Path(path).suffix.lower())


def is_supported_path(path: Path | str) -> bool:
    """Return True when the file name suffix maps to a supported container."""
    return format_from_suffix(path) is not None


def output_format_for_input(input_path: Path | str, requested: ImageFormat | None) -> ImageFormat:
    """Decide the container an input file will be converted into.

    Used by the CLI and GUI to name the default output file before any bytes are
    read. Conversions are format-preserving unless a container was requested.

    Args:
        input_path: Path of the source image.
        requested: Explicitly requested output format, or ``None`` for auto.

    Returns:
        The container the output should use.
    """
    return requested or format_from_suffix(input_path) or DEFAULT_FORMAT


def detect_format(data: bytes) -> ImageFormat:
    """Identify the container format from the leading bytes of a file.

    Byte sniffing is authoritative: it is used in preference to the file name
    suffix so that mislabelled files fail with an actionable message instead of
    being handed to the wrong decoder.

    Args:
        data: Raw file contents (only the first few bytes are inspected).

    Returns:
        The detected container format.

    Raises:
        UnsupportedFormatError: If the bytes do not match a supported container.
    """
    if data.startswith(_JPEG_SOI):
        return ImageFormat.JPEG

    if len(data) >= MAGIC_PROBE_BYTES and data[4:8] == _ISOBMFF_FTYP:
        if data[8:12] in _AVIF_BRANDS:
            return ImageFormat.AVIF
        raise UnsupportedFormatError(
            f"Unsupported ISOBMFF brand {data[8:12]!r}; only AVIF files are supported."
        )

    raise UnsupportedFormatError(
        f"Unrecognised image container (leading bytes {data[:MAGIC_PROBE_BYTES]!r}). "
        f"Supported formats: {', '.join(sorted(fmt.value for fmt in ImageFormat))}."
    )


def resolve_output_format(
    output_path: Path | str,
    requested: ImageFormat | None,
    input_format: ImageFormat,
) -> ImageFormat:
    """Decide which container to write.

    Priority: an explicitly requested format wins, then the output file name
    suffix, then the input format (so conversions are format-preserving by
    default).

    Args:
        output_path: Destination path for the converted file.
        requested: Explicitly requested output format, or ``None`` for auto.
        input_format: Format of the source file, used as the final fallback.

    Returns:
        The container format to write.
    """
    if requested is not None:
        return requested
    return format_from_suffix(output_path) or input_format
