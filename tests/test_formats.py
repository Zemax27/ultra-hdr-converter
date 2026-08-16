from pathlib import Path

import pytest

from ultra_hdr_converter.core.formats import (
    DEFAULT_FORMAT,
    SUPPORTED_SUFFIXES,
    ImageFormat,
    detect_format,
    format_from_suffix,
    is_supported_path,
    output_format_for_input,
    resolve_output_format,
)
from ultra_hdr_converter.errors import UnsupportedFormatError

JPEG_HEADER = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01"
AVIF_HEADER = b"\x00\x00\x00\x20ftypavif\x00\x00\x00\x00"
HEIC_HEADER = b"\x00\x00\x00\x20ftypheic\x00\x00\x00\x00"
PNG_HEADER = b"\x89PNG\r\n\x1a\n\x00\x00\x00\r"


def test_detect_format_identifies_jpeg() -> None:
    assert detect_format(JPEG_HEADER) is ImageFormat.JPEG


def test_detect_format_identifies_avif() -> None:
    assert detect_format(AVIF_HEADER) is ImageFormat.AVIF


def test_detect_format_accepts_mif1_branded_avif() -> None:
    """AVIF files written by other encoders may lead with the generic HEIF brand."""
    assert detect_format(b"\x00\x00\x00\x20ftypmif1\x00\x00\x00\x00") is ImageFormat.AVIF


def test_detect_format_rejects_other_isobmff_brands() -> None:
    with pytest.raises(UnsupportedFormatError, match="brand"):
        detect_format(HEIC_HEADER)


def test_detect_format_rejects_unknown_container() -> None:
    with pytest.raises(UnsupportedFormatError, match="Unrecognised"):
        detect_format(PNG_HEADER)


def test_detect_format_rejects_empty_input() -> None:
    with pytest.raises(UnsupportedFormatError):
        detect_format(b"")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("photo.jpg", ImageFormat.JPEG),
        ("photo.JPEG", ImageFormat.JPEG),
        ("photo.avif", ImageFormat.AVIF),
        ("photo.AVIF", ImageFormat.AVIF),
        ("photo.png", None),
        ("photo", None),
    ],
)
def test_format_from_suffix(name: str, expected: ImageFormat | None) -> None:
    assert format_from_suffix(Path(name)) is expected


def test_supported_suffixes_cover_every_format() -> None:
    assert SUPPORTED_SUFFIXES == {".jpg", ".jpeg", ".avif"}
    for image_format in ImageFormat:
        assert image_format.default_suffix in SUPPORTED_SUFFIXES


def test_is_supported_path() -> None:
    assert is_supported_path("a.jpg") is True
    assert is_supported_path("a.avif") is True
    assert is_supported_path("a.tiff") is False


def test_resolve_output_format_prefers_explicit_request() -> None:
    assert resolve_output_format("out.jpg", ImageFormat.AVIF, ImageFormat.JPEG) is ImageFormat.AVIF


def test_resolve_output_format_falls_back_to_output_suffix() -> None:
    assert resolve_output_format("out.avif", None, ImageFormat.JPEG) is ImageFormat.AVIF


def test_resolve_output_format_falls_back_to_input_format() -> None:
    """An extensionless output keeps the input container instead of guessing."""
    assert resolve_output_format("out", None, ImageFormat.AVIF) is ImageFormat.AVIF


@pytest.mark.parametrize(
    ("input_name", "requested", "expected"),
    [
        ("photo.jpg", None, ImageFormat.JPEG),
        ("photo.avif", None, ImageFormat.AVIF),
        ("photo.jpg", ImageFormat.AVIF, ImageFormat.AVIF),
        ("photo.avif", ImageFormat.JPEG, ImageFormat.JPEG),
        ("photo.unknown", None, DEFAULT_FORMAT),
    ],
)
def test_output_format_for_input(input_name: str, requested: ImageFormat | None, expected: ImageFormat) -> None:
    """Conversions preserve the input container unless one is explicitly requested."""
    assert output_format_for_input(Path(input_name), requested) is expected
