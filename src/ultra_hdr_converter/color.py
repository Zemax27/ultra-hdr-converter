"""Color management and linearization helpers."""

from __future__ import annotations

from typing import Any, cast

import imagecodecs
import numpy as np
from numpy.typing import DTypeLike


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
    color_space = "gray" if sdr_array.ndim == 2 or (sdr_array.ndim == 3 and sdr_array.shape[2] == 1) else "rgb"

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
