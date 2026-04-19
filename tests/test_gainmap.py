import numpy as np
import pytest

from ultra_hdr_converter.gainmap import (
    RadianceMapConfig,
    generate_log2_gain_map,
    generate_radiance_gain_map,
    validate_gain_map,
)


def test_validate_gain_map_casts_to_uint8() -> None:
    gain = np.full((4, 4), 301.2, dtype=np.float32)
    validated = validate_gain_map(gain, (4, 4, 3))

    assert validated.dtype == np.uint8
    assert validated.max() == 255


def test_validate_gain_map_rejects_mismatched_shape() -> None:
    gain = np.zeros((2, 2), dtype=np.uint8)

    with pytest.raises(ValueError):
        validate_gain_map(gain, (3, 3, 3))


def test_generate_log2_gain_map_returns_expected_shape_dtype() -> None:
    linear = np.ones((8, 8, 3), dtype=np.float32)
    gain = generate_log2_gain_map(linear)

    assert gain.shape == (8, 8)
    assert gain.dtype == np.uint8


def test_generate_radiance_gain_map_returns_expected_shape_dtype() -> None:
    linear = np.ones((12, 10, 3), dtype=np.float32)
    config = RadianceMapConfig(resize_factor=0.5, guided_radius=2, guided_eps=0.05)

    gain = generate_radiance_gain_map(linear, config=config)

    assert gain.shape == (12, 10)
    assert gain.dtype == np.uint8


def test_generate_radiance_gain_map_rejects_invalid_config() -> None:
    linear = np.ones((8, 8, 3), dtype=np.float32)
    invalid = RadianceMapConfig(resize_factor=0.0)

    with pytest.raises(ValueError):
        generate_radiance_gain_map(linear, config=invalid)
