import numpy as np
import pytest

from ultra_hdr_converter.core.gain_map import (
    GainMapConfig,
    _guided_filter_grayscale,
    generate_gain_map,
    validate_gain_map,
)
from ultra_hdr_converter.errors import GainMapConfigError, GainMapDimensionError


def test_validate_gain_map_casts_to_uint8() -> None:
    gain = np.full((4, 4), 301.2, dtype=np.float32)
    validated = validate_gain_map(gain)

    assert validated.dtype == np.uint8
    assert validated.max() == np.iinfo(np.uint8).max


def test_validate_gain_map_rejects_invalid_ndim() -> None:
    gain = np.zeros((2,), dtype=np.uint8)

    with pytest.raises(GainMapDimensionError, match="2D or 3D"):
        validate_gain_map(gain)


def test_validate_gain_map_rejects_invalid_channels() -> None:
    gain = np.zeros((4, 4, 5), dtype=np.uint8)

    with pytest.raises(GainMapDimensionError, match="1 or 3"):
        validate_gain_map(gain)


def test_generate_gain_map_returns_expected_shape_dtype() -> None:
    luminance = np.linspace(0.0, 1.0, 120, dtype=np.float32).reshape(12, 10)
    config = GainMapConfig(guided_radius=2, guided_eps=0.05)

    gain = generate_gain_map(luminance, config=config)

    assert gain.shape == (12, 10)
    assert gain.dtype == np.uint8


def test_generate_gain_map_nonzero_for_highlights() -> None:
    luminance = np.full((8, 8), 0.8, dtype=np.float32)
    config = GainMapConfig(highlight_threshold=0.5, guided_radius=1)

    gain = generate_gain_map(luminance, config=config)

    assert gain.max() > 0, "Gain map should have nonzero values for above-threshold luminance"


def test_generate_gain_map_rejects_invalid_config() -> None:
    luminance = np.ones((8, 8), dtype=np.float32)
    invalid = GainMapConfig(highlight_threshold=0.0)

    with pytest.raises(GainMapConfigError, match="highlight_threshold"):
        generate_gain_map(luminance, config=invalid)


def test_generate_gain_map_rejects_negative_gamma() -> None:
    luminance = np.ones((8, 8), dtype=np.float32)
    invalid = GainMapConfig(expansion_gamma=-1.0)

    with pytest.raises(GainMapConfigError, match="expansion_gamma"):
        generate_gain_map(luminance, config=invalid)


def test_generate_gain_map_rejects_3d_luminance() -> None:
    luminance = np.ones((4, 4, 3), dtype=np.float32)

    with pytest.raises(GainMapDimensionError, match="2-D"):
        generate_gain_map(luminance)


def test_generate_gain_map_uniform_luminance_returns_zeros() -> None:
    luminance = np.full((8, 8), 0.5, dtype=np.float32)
    config = GainMapConfig(highlight_threshold=0.9, guided_radius=1)

    gain = generate_gain_map(luminance, config=config)

    assert gain.dtype == np.uint8
    assert gain.shape == (8, 8)


def test_guided_filter_rejects_shape_mismatch() -> None:
    guide = np.ones((4, 4), dtype=np.float32)
    src = np.ones((4, 3), dtype=np.float32)

    with pytest.raises(GainMapDimensionError, match="same shape"):
        _guided_filter_grayscale(guide, src, radius=1, eps=1e-3)


def test_guided_filter_rejects_non_2d() -> None:
    guide = np.ones((4, 4, 3), dtype=np.float32)
    src = np.ones((4, 4, 3), dtype=np.float32)

    with pytest.raises(GainMapDimensionError, match="2D"):
        _guided_filter_grayscale(guide, src, radius=1, eps=1e-3)
