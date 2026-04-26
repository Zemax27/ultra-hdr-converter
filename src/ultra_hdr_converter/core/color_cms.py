"""ICC color management transforms using imagecodecs CMS (impure shell)."""

from __future__ import annotations

from typing import Any, cast

import imagecodecs
import numpy as np
from numpy.typing import DTypeLike

from ultra_hdr_converter.core.color import (
    COLOR_NDIM,
    GRAYSCALE_NDIM,
    SINGLE_CHANNEL,
    extract_y_channel,
    luminance_from_grayscale,
)
from ultra_hdr_converter.errors import ColorTransformError


def ensure_cms_available() -> None:
    """Verify that imagecodecs CMS extension is present and usable.

    Raises:
        ColorTransformError: If the CMS extension is not available.
    """
    if not hasattr(imagecodecs, "cms_profile") or not hasattr(imagecodecs, "cms_transform"):
        raise ColorTransformError(
            "imagecodecs cms extension is unavailable. Install full imagecodecs with cms support."
        )


def transform_profiles(
    sdr_array: np.ndarray,
    src_profile: Any,
    dst_profile: Any,
    outdtype: DTypeLike,
) -> np.ndarray:
    """Run CMS transform between two profiles and normalize dtype behavior across API versions.

    Args:
        sdr_array: SDR pixel array of shape (H, W) or (H, W, C).
        src_profile: Source ICC profile handle from ``imagecodecs.cms_profile``.
        dst_profile: Destination ICC profile handle from ``imagecodecs.cms_profile``.
        outdtype: Desired output dtype for the transformed array.

    Returns:
        Transformed pixel array with the requested dtype.
    """
    color_space = (
        "gray"
        if sdr_array.ndim == GRAYSCALE_NDIM or (sdr_array.ndim == COLOR_NDIM and sdr_array.shape[2] == SINGLE_CHANNEL)
        else "rgb"
    )

    try:
        transformed = cast(Any, imagecodecs.cms_transform)(
            sdr_array,
            src_profile,
            dst_profile,
            colorspace=color_space,
            outcolorspace=color_space,
            outdtype=outdtype,
        )
    except TypeError:
        transformed = cast(Any, imagecodecs.cms_transform)(
            sdr_array,
            src_profile,
            dst_profile,
        )
    return np.asarray(transformed, dtype=outdtype)


def build_linear_profile(profile: bytes | str) -> Any:
    """Build a linearized ICC profile with support for multiple imagecodecs API generations.

    Args:
        profile: Source profile identifier or raw bytes.

    Returns:
        Linearized ICC profile handle from ``imagecodecs.cms_profile``.

    Raises:
        ColorTransformError: If none of the API variants succeed.
    """
    try:
        return cast(Any, imagecodecs.cms_profile)(profile, linear=True)
    except TypeError:
        pass

    try:
        return cast(Any, imagecodecs.cms_profile)(profile, gamma=1.0)
    except TypeError:
        pass

    try:
        return cast(Any, imagecodecs.cms_profile)(profile, transferfunction="linear")
    except TypeError:
        raise ColorTransformError(
            f"Cannot build linear profile for {profile!r}: unsupported imagecodecs API."
        ) from None


def transform_srgb(sdr_array: np.ndarray, outdtype: DTypeLike) -> np.ndarray:
    """Linearize SDR pixels with standard sRGB transfer assumptions.

    Args:
        sdr_array: SDR pixel array of shape (H, W) or (H, W, C), dtype uint8.
        outdtype: Desired output dtype for the linearized array.

    Returns:
        Linearized pixel array with the requested dtype.

    Raises:
        ColorTransformError: If all CMS transform attempts fail.
    """
    try:
        src_profile = cast(Any, imagecodecs.cms_profile)("srgb")
        dst_profile = build_linear_profile("srgb")
        return transform_profiles(sdr_array, src_profile, dst_profile, outdtype)
    except (TypeError, ValueError) as exc:
        legacy = _legacy_srgb_linearize(sdr_array, outdtype)
        if legacy is not None:
            return legacy
        raise ColorTransformError("sRGB linearization failed through all code paths.") from exc


def _legacy_srgb_linearize(sdr_array: np.ndarray, outdtype: DTypeLike) -> np.ndarray | None:
    """Legacy fallback path for builds that only support linear=True in cms_transform."""
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
        pass

    try:
        linear = cast(Any, imagecodecs.cms_transform)(
            sdr_array,
            "srgb",
            "srgb",
            linear=True,
        )
        return np.asarray(linear, dtype=outdtype)
    except TypeError:
        return None


def linearize_from_icc(
    sdr_array: np.ndarray,
    icc_profile: bytes | None,
    outdtype: DTypeLike = np.float32,
) -> np.ndarray:
    """Convert SDR pixels to linear space using embedded ICC profile when available.

    Falls back to sRGB assumptions if profile is missing or rejected by CMS.

    Args:
        sdr_array: Non-linear SDR image of shape (H, W, C), dtype uint8.
        icc_profile: Embedded ICC profile bytes, or None.
        outdtype: Output dtype for the linearized array.

    Returns:
        Linearized pixel array with the requested dtype.

    Raises:
        ColorTransformError: If CMS is unavailable or all transform paths fail.
    """
    ensure_cms_available()

    if icc_profile:
        try:
            src_profile = cast(Any, imagecodecs.cms_profile)(icc_profile)
            dst_profile = build_linear_profile(icc_profile)
            return transform_profiles(sdr_array, src_profile, dst_profile, outdtype)
        except (TypeError, ValueError, RuntimeError):
            return transform_srgb(sdr_array, outdtype)

    return transform_srgb(sdr_array, outdtype)


def transform_to_xyz(
    sdr_array: np.ndarray,
    icc_profile: bytes | None,
    outdtype: DTypeLike = np.float32,
) -> np.ndarray:
    """Convert SDR pixels directly to CIE XYZ using CMS.

    Transforms through the ICC Profile Connection Space (CIE XYZ D50),
    avoiding intermediate sRGB conversion and potential gamut clipping.
    Falls back to sRGB assumptions when the profile is missing or rejected.

    Args:
        sdr_array: Non-linear SDR image of shape (H, W, C), dtype uint8.
        icc_profile: Embedded ICC profile bytes, or None.
        outdtype: Output dtype for the XYZ array.

    Returns:
        CIE XYZ array of shape (H, W, 3) with the requested dtype. Values are
        normalized to the range [0.0, 1.0] where 1.0 represents the reference
        white of the Profile Connection Space (D50).

    Raises:
        ColorTransformError: If CMS is unavailable or all transform paths fail.
    """
    ensure_cms_available()

    xyz_profile = cast(Any, imagecodecs.cms_profile)("xyz")

    def _cms_to_xyz(src_profile: Any) -> np.ndarray:
        try:
            transformed = cast(Any, imagecodecs.cms_transform)(
                sdr_array,
                src_profile,
                xyz_profile,
                colorspace="rgb",
                outcolorspace="xyz",
                outdtype=outdtype,
            )
        except TypeError:
            transformed = cast(Any, imagecodecs.cms_transform)(
                sdr_array,
                src_profile,
                xyz_profile,
            )
        return np.asarray(transformed, dtype=outdtype)

    if icc_profile:
        try:
            src_profile = cast(Any, imagecodecs.cms_profile)(icc_profile)
            return _cms_to_xyz(src_profile)
        except (TypeError, ValueError, RuntimeError):
            pass

    src_profile = cast(Any, imagecodecs.cms_profile)("srgb")
    return _cms_to_xyz(src_profile)


def extract_xyz_luminance(
    sdr_array: np.ndarray,
    icc_profile: bytes | None,
    outdtype: DTypeLike = np.float32,
) -> np.ndarray:
    """Convert SDR pixels to CIE XYZ and return the Y (luminance) channel.

    Uses the ICC Profile Connection Space (CIE XYZ D50) for a direct
    conversion that correctly handles any source colourspace without
    intermediate sRGB gamut clipping. Falls back to sRGB assumptions
    when the profile is missing or rejected by CMS.

    The returned luminance values are normalized to the range [0.0, 1.0],
    where 1.0 corresponds to the reference white of the profile connection
    space.

    Args:
        sdr_array: Non-linear SDR image of shape (H, W) or (H, W, 3), dtype uint8.
        icc_profile: Embedded ICC profile bytes, or None.
        outdtype: Output dtype for the luminance array.

    Returns:
        2-D float array of CIE Y (luminance) values with shape (H, W) and values
        in the normalized range [0.0, 1.0].

    Raises:
        ColorTransformError: If the SDR array shape is invalid or CMS fails.
    """
    if sdr_array.ndim == GRAYSCALE_NDIM:
        return luminance_from_grayscale(sdr_array, outdtype=outdtype)

    xyz = transform_to_xyz(sdr_array, icc_profile, outdtype=np.float32)
    return extract_y_channel(xyz, outdtype=outdtype)
