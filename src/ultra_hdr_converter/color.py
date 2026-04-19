"""Color management and linearization helpers."""

from __future__ import annotations

from typing import Any

import imagecodecs
import numpy as np
from numpy.typing import DTypeLike


def _ensure_cms_available() -> None:
    if not hasattr(imagecodecs, "cms_profile") or not hasattr(imagecodecs, "cms_transform"):
        raise RuntimeError("imagecodecs cms extension is unavailable. Install full imagecodecs with cms support.")


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

    kwargs: dict[str, Any] = {"outdtype": outdtype}
    if icc_profile:
        src_profile = imagecodecs.cms_profile(icc_profile)
        dst_profile = imagecodecs.cms_profile(icc_profile, linear=True)
        try:
            return np.asarray(imagecodecs.cms_transform(sdr_array, src_profile, dst_profile, **kwargs))
        except TypeError:
            linear = imagecodecs.cms_transform(sdr_array, src_profile, dst_profile)
            return np.asarray(linear, dtype=outdtype)

    try:
        return np.asarray(imagecodecs.cms_transform(sdr_array, "srgb", "srgb", linear=True, **kwargs))
    except TypeError:
        linear = imagecodecs.cms_transform(sdr_array, "srgb", "srgb", linear=True)
        return np.asarray(linear, dtype=outdtype)
