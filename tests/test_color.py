import numpy as np
import pytest

from ultra_hdr_converter.core.color import extract_y_channel, luminance_from_grayscale
from ultra_hdr_converter.errors import ColorTransformError


def test_extract_y_channel_returns_y_from_xyz():
    xyz = np.zeros((4, 4, 3), dtype=np.float32)
    xyz[..., 1] = 0.75
    result = extract_y_channel(xyz)
    assert result.shape == (4, 4)
    assert result.dtype == np.float32
    assert np.allclose(result, 0.75)


def test_extract_y_channel_rejects_2d_input():
    xyz = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ColorTransformError, match="XYZ array must be shape"):
        extract_y_channel(xyz)


def test_extract_y_channel_rejects_insufficient_channels():
    xyz = np.zeros((4, 4, 2), dtype=np.float32)
    with pytest.raises(ColorTransformError, match="XYZ array must be shape"):
        extract_y_channel(xyz)


def test_luminance_from_grayscale_casts_dtype():
    gray = np.full((3, 3), 128, dtype=np.uint8)
    result = luminance_from_grayscale(gray)
    assert result.shape == (3, 3)
    assert result.dtype == np.float32
    assert np.allclose(result, 128.0)


def test_luminance_from_grayscale_preserves_float32():
    gray = np.full((3, 3), 0.5, dtype=np.float32)
    result = luminance_from_grayscale(gray)
    assert result.dtype == np.float32
    assert np.allclose(result, 0.5)
