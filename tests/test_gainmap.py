import numpy as np
import pytest

from ultra_hdr_converter.gainmap import generate_log2_gain_map, validate_gain_map


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
