"""Sample-depth handling across the pipeline.

Decoders return a ``uint16`` array for anything deeper than 8 bits, but the
samples only occupy the low bits — a 10-bit image peaks at 1023, not 65535.
Treating that array as full-range makes the image look almost black, which
silently produces an all-zero gain map. These tests pin that behaviour down.
"""

import imagecodecs
import numpy as np
import pytest

from ultra_hdr_converter.core.color import (
    SDR_BIT_DEPTH,
    max_sample_value,
    rescale_sample_depth,
    to_full_range,
)
from ultra_hdr_converter.core.color_cms import extract_xyz_luminance
from ultra_hdr_converter.core.image_io import probe_bit_depth
from ultra_hdr_converter.core.jpeg_io import encode_jpeg
from ultra_hdr_converter.errors import ColorTransformError, JpegStructureError

DEPTH_10 = 10
DEPTH_12 = 12
MAX_10_BIT = 1023
MAX_12_BIT = 4095
MAX_16_BIT = 65535
MAX_8_BIT = 255
LUMINANCE_TOLERANCE = 0.02
# A 10-bit image read as full-range uint16 measures ~64x too dark.
UNEXPANDED_LUMINANCE_CEILING = 0.05


@pytest.fixture(scope="module")
def highlight_image() -> np.ndarray:
    """An 8-bit image with a bright localised highlight."""
    height, width = 64, 80
    yy, xx = np.mgrid[0:height, 0:width]
    image = np.zeros((height, width, 3), np.float32)
    image[..., 0] = 0.3
    image[..., 1] = 0.35
    image[..., 2] = 0.45
    image += np.exp(-(((xx - 60) ** 2 + (yy - 16) ** 2) / 200.0))[..., None] * 0.9
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


# ---- Sample rescaling --------------------------------------------------------


@pytest.mark.parametrize(
    ("bit_depth", "expected"),
    [(SDR_BIT_DEPTH, MAX_8_BIT), (DEPTH_10, MAX_10_BIT), (DEPTH_12, MAX_12_BIT), (16, MAX_16_BIT)],
)
def test_max_sample_value(bit_depth: int, expected: int) -> None:
    assert max_sample_value(bit_depth) == expected


def test_max_sample_value_rejects_out_of_range_depth() -> None:
    with pytest.raises(ColorTransformError, match="Unsupported sample bit depth"):
        max_sample_value(24)


def test_rescale_maps_full_scale_white_to_full_scale_white() -> None:
    """A 10-bit 1023 must become 255, not be truncated to 255 by luck."""
    samples = np.array([0, 512, MAX_10_BIT], dtype=np.uint16)
    rescaled = rescale_sample_depth(samples, DEPTH_10, SDR_BIT_DEPTH)

    assert rescaled.dtype == np.uint8
    assert rescaled[0] == 0
    assert rescaled[-1] == MAX_8_BIT


def test_rescale_is_a_no_op_when_depths_match() -> None:
    samples = np.array([0, 128, 255], dtype=np.uint8)
    assert np.array_equal(rescale_sample_depth(samples, SDR_BIT_DEPTH, SDR_BIT_DEPTH), samples)


def test_rescale_rejects_float_input() -> None:
    with pytest.raises(ColorTransformError, match="integer array"):
        rescale_sample_depth(np.zeros((2, 2), np.float32), DEPTH_10, SDR_BIT_DEPTH)


def test_to_full_range_expands_ten_bit_samples_to_uint16_range() -> None:
    samples = np.array([0, MAX_10_BIT], dtype=np.uint16)
    expanded = to_full_range(samples, DEPTH_10)

    assert expanded.dtype == np.uint16
    assert expanded[0] == 0
    assert expanded[-1] == MAX_16_BIT


def test_to_full_range_leaves_eight_bit_samples_alone() -> None:
    samples = np.array([0, 77, 255], dtype=np.uint8)
    assert np.array_equal(to_full_range(samples, SDR_BIT_DEPTH), samples)


# ---- Luminance -------------------------------------------------------------


def test_ten_bit_luminance_matches_eight_bit_after_expansion(highlight_image: np.ndarray) -> None:
    """The same picture must measure the same brightness at either depth."""
    image10 = rescale_sample_depth(highlight_image, SDR_BIT_DEPTH, DEPTH_10)

    luminance8 = extract_xyz_luminance(highlight_image, None)
    luminance10 = extract_xyz_luminance(to_full_range(image10, DEPTH_10), None)

    assert np.abs(luminance8 - luminance10).max() < LUMINANCE_TOLERANCE


def test_unexpanded_ten_bit_luminance_is_wrong(highlight_image: np.ndarray) -> None:
    """Regression guard: without expansion a 10-bit image reads as near black."""
    image10 = rescale_sample_depth(highlight_image, SDR_BIT_DEPTH, DEPTH_10)

    naive = extract_xyz_luminance(image10, None)

    assert naive.max() < UNEXPANDED_LUMINANCE_CEILING


# ---- Bit depth probing -------------------------------------------------------


def test_probe_bit_depth_reads_jpeg_frame_precision(highlight_image: np.ndarray) -> None:
    assert probe_bit_depth(bytes(imagecodecs.jpeg_encode(highlight_image, level=90))) == SDR_BIT_DEPTH


def test_probe_bit_depth_reads_avif_codec_configuration(highlight_image: np.ndarray) -> None:
    image10 = rescale_sample_depth(highlight_image, SDR_BIT_DEPTH, DEPTH_10)
    encoded = bytes(imagecodecs.avif_encode(image10, level=80, bitspersample=DEPTH_10))

    assert probe_bit_depth(encoded) == DEPTH_10


# ---- JPEG encoding guard -----------------------------------------------------


def test_encode_jpeg_rejects_high_bit_depth_arrays() -> None:
    """A uint16 array would silently produce an unreadable 12-bit JPEG."""
    with pytest.raises(JpegStructureError, match="8-bit array"):
        encode_jpeg(np.zeros((8, 8, 3), np.uint16))
