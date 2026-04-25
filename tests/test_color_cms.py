import numpy as np
import pytest

from ultra_hdr_converter.core.color_cms import extract_xyz_luminance, linearize_from_icc
from ultra_hdr_converter.errors import ColorTransformError

EXPECTED_NDIM = 2


def test_linearize_from_icc_falls_back_to_srgb_on_invalid_profile(monkeypatch: object) -> None:
    sdr = np.zeros((2, 2, 3), dtype=np.uint8)
    profile_calls: list[tuple[object, object]] = []
    transform_calls: list[tuple[object, object]] = []

    def _fake_profile(profile: object, **kwargs: object) -> object:
        profile_calls.append((profile, tuple(sorted(kwargs.items()))))
        if profile == b"icc":
            raise ValueError("bad icc")
        return f"profile:{profile}:{kwargs}"

    def _fake_transform(
        _array: np.ndarray,
        src: object,
        dst: object,
        **_kwargs: object,
    ) -> np.ndarray:
        transform_calls.append((src, dst))
        return np.ones((2, 2, 3), dtype=np.float32)

    monkeypatch.setattr("ultra_hdr_converter.core.color_cms.imagecodecs.cms_profile", _fake_profile)
    monkeypatch.setattr("ultra_hdr_converter.core.color_cms.imagecodecs.cms_transform", _fake_transform)

    output = linearize_from_icc(sdr, b"icc", outdtype=np.float32)

    assert output.dtype == np.float32
    assert output.shape == (2, 2, 3)
    assert profile_calls[0][0] == b"icc"
    assert profile_calls[1][0] == "srgb"
    assert transform_calls[0][0] != "srgb"
    assert transform_calls[0][1] != "srgb"


def test_extract_xyz_luminance_returns_2d_float(monkeypatch: object) -> None:
    sdr = np.full((3, 3, 3), 128, dtype=np.uint8)

    def _fake_profile(profile: object, **kwargs: object) -> object:
        return f"profile:{profile}:{kwargs}"

    def _fake_transform(
        _array: np.ndarray,
        _src: object,
        _dst: object,
        **_kwargs: object,
    ) -> np.ndarray:
        result = np.empty((3, 3, 3), dtype=np.float32)
        result[..., 0] = 0.5
        result[..., 1] = 0.7
        result[..., 2] = 0.3
        return result

    monkeypatch.setattr("ultra_hdr_converter.core.color_cms.imagecodecs.cms_profile", _fake_profile)
    monkeypatch.setattr("ultra_hdr_converter.core.color_cms.imagecodecs.cms_transform", _fake_transform)

    luminance = extract_xyz_luminance(sdr, icc_profile=None)

    assert luminance.ndim == EXPECTED_NDIM
    assert luminance.shape == (3, 3)
    assert luminance.dtype == np.float32
    assert np.allclose(luminance, 0.7, atol=1e-6)


def test_extract_xyz_luminance_passthrough_grayscale() -> None:
    gray = np.full((4, 4), 128, dtype=np.uint8)
    luminance = extract_xyz_luminance(gray, icc_profile=None)
    assert luminance.ndim == EXPECTED_NDIM
    assert luminance.shape == (4, 4)
    assert luminance.dtype == np.float32


def test_linearize_from_icc_raises_when_cms_unavailable(monkeypatch: object) -> None:
    sdr = np.zeros((2, 2, 3), dtype=np.uint8)

    def _raise_cms_unavailable() -> None:
        raise ColorTransformError("imagecodecs cms extension is unavailable.")

    monkeypatch.setattr(
        "ultra_hdr_converter.core.color_cms.ensure_cms_available",
        _raise_cms_unavailable,
    )

    with pytest.raises(ColorTransformError, match="cms extension"):
        linearize_from_icc(sdr, None)
