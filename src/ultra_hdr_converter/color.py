"""Color management and linearization helpers."""

from __future__ import annotations

from typing import Any, cast

import imagecodecs
import numpy as np
from numpy.typing import DTypeLike

GRAYSCALE_NDIM = 2
COLOR_NDIM = 3
SINGLE_CHANNEL = 1


def _ensure_cms_available() -> None:
    if not hasattr(imagecodecs, "cms_profile") or not hasattr(imagecodecs, "cms_transform"):
        raise RuntimeError("imagecodecs cms extension is unavailable. Install full imagecodecs with cms support.")


def _transform_profiles(
    sdr_array: np.ndarray,
    src_profile: Any,
    dst_profile: Any,
    outdtype: DTypeLike,
) -> np.ndarray:
    """Run CMS transform between two profiles and normalize dtype behavior across API versions."""
    color_space = (
        "gray"
        if sdr_array.ndim == GRAYSCALE_NDIM or (sdr_array.ndim == COLOR_NDIM and sdr_array.shape[2] == SINGLE_CHANNEL)
        else "rgb"
    )

    attempts: list[dict[str, Any]] = [
        {"colorspace": color_space, "outcolorspace": color_space, "outdtype": outdtype},
        {"colorspace": color_space, "outcolorspace": color_space},
        {"outdtype": outdtype},
        {},
    ]

    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            transformed = cast(Any, imagecodecs.cms_transform)(
                sdr_array,
                src_profile,
                dst_profile,
                **kwargs,
            )
            return np.asarray(transformed, dtype=outdtype)
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error

    raise RuntimeError("cms_transform failed unexpectedly without a captured error.")


def _build_linear_profile(profile: bytes | str) -> Any:
    """Build a linearized ICC profile with support for multiple imagecodecs API generations."""
    try:
        return cast(Any, imagecodecs.cms_profile)(profile, linear=True)
    except TypeError:
        pass

    try:
        return cast(Any, imagecodecs.cms_profile)(profile, gamma=1.0)
    except TypeError:
        pass

    return cast(Any, imagecodecs.cms_profile)(profile, transferfunction="linear")


def _transform_srgb(sdr_array: np.ndarray, outdtype: DTypeLike) -> np.ndarray:
    """Linearize SDR pixels with standard sRGB transfer assumptions."""
    try:
        src_profile = cast(Any, imagecodecs.cms_profile)("srgb")
        dst_profile = _build_linear_profile("srgb")
        return _transform_profiles(sdr_array, src_profile, dst_profile, outdtype)
    except Exception:
        # Legacy fallback path for builds that only support linear=True in cms_transform.
        kwargs: dict[str, Any] = {"outdtype": outdtype}
        try:
            transformed = cast(Any, imagecodecs.cms_transform)(
                sdr_array,
                "srgb",
                "srgb",
                linear=True,
                **kwargs,
            )
            return np.asarray(transformed)
        except TypeError:
            linear = cast(Any, imagecodecs.cms_transform)(
                sdr_array,
                "srgb",
                "srgb",
                linear=True,
            )
            return np.asarray(linear, dtype=outdtype)


def linearize_from_icc(
    sdr_array: np.ndarray,
    icc_profile: bytes | None,
    outdtype: DTypeLike = np.float32,
) -> np.ndarray:
    """
    Convert SDR pixels to linear space using embedded ICC profile when available.

    Falls back to sRGB assumptions if profile is missing.
    """
    _ensure_cms_available()

    if icc_profile:
        try:
            src_profile = cast(Any, imagecodecs.cms_profile)(icc_profile)
            dst_profile = _build_linear_profile(icc_profile)
            return _transform_profiles(sdr_array, src_profile, dst_profile, outdtype)
        except Exception:
            # Some JPEGs carry ICC payloads that libcms rejects; use deterministic sRGB fallback.
            return _transform_srgb(sdr_array, outdtype)

    return _transform_srgb(sdr_array, outdtype)


# sRGB to CIE XYZ conversion matrix, Y row (ITU-R BT.709 primaries).
# Applied to linear sRGB values to obtain CIE 1931 photopic luminance.
_SRGB_TO_XYZ_Y = np.array([0.2126729, 0.7151522, 0.0721750], dtype=np.float32)


def _linearize_to_srgb(sdr_array: np.ndarray, icc_profile: bytes | None, outdtype: DTypeLike) -> np.ndarray:
    """Convert SDR pixels to linear sRGB regardless of source ICC profile."""
    _ensure_cms_available()

    dst_profile = _build_linear_profile("srgb")

    if icc_profile:
        try:
            src_profile = cast(Any, imagecodecs.cms_profile)(icc_profile)
            return _transform_profiles(sdr_array, src_profile, dst_profile, outdtype)
        except Exception:
            pass

    # Fallback: assume source is sRGB.
    src_profile = cast(Any, imagecodecs.cms_profile)("srgb")
    return _transform_profiles(sdr_array, src_profile, dst_profile, outdtype)


def extract_xyz_luminance(
    sdr_array: np.ndarray,
    icc_profile: bytes | None,
    outdtype: DTypeLike = np.float32,
) -> np.ndarray:
    """
    Convert SDR pixels to CIE XYZ via linear sRGB and return the Y channel.

    The source ICC profile is used for chromatic adaptation to sRGB primaries
    before applying the standard sRGB-to-XYZ matrix. Falls back to sRGB
    assumptions when the profile is missing or rejected by CMS.

    Args:
        sdr_array: Non-linear SDR image (uint8, H x W x 3).
        icc_profile: Embedded ICC profile bytes, or None.
        outdtype: Output dtype for the luminance array.

    Returns:
        2-D float array of CIE Y (luminance) values.
    """
    linear_srgb = _linearize_to_srgb(sdr_array, icc_profile, outdtype=np.float32)

    if linear_srgb.ndim == COLOR_NDIM and linear_srgb.shape[2] >= COLOR_NDIM:
        luminance = (
            linear_srgb[..., 0] * _SRGB_TO_XYZ_Y[0]
            + linear_srgb[..., 1] * _SRGB_TO_XYZ_Y[1]
            + linear_srgb[..., 2] * _SRGB_TO_XYZ_Y[2]
        )
    elif linear_srgb.ndim == GRAYSCALE_NDIM:
        luminance = linear_srgb
    else:
        raise ValueError("SDR array must be shape (H, W) or (H, W, C>=3).")

    return np.asarray(luminance, dtype=outdtype)
