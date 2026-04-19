import numpy as np
import pytest

from ultra_hdr_converter.gainmap import (
    GainMapConfig,
    generate_gain_map,
    validate_gain_map,
)


def test_validate_gain_map_casts_to_uint8() -> None:
    gain = np.full((4, 4), 301.2, dtype=np.float32)
    validated = validate_gain_map(gain)

    assert validated.dtype == np.uint8
    assert validated.max() == np.iinfo(np.uint8).max


def test_validate_gain_map_rejects_invalid_ndim() -> None:
    gain = np.zeros((2,), dtype=np.uint8)

    with pytest.raises(ValueError):
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

    with pytest.raises(ValueError):
        generate_gain_map(luminance, config=invalid)
