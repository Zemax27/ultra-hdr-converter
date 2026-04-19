"""Ultra HDR encoder wrapper."""

from __future__ import annotations

import imagecodecs
import numpy as np


def _ensure_ultrahdr_available() -> None:
    if not hasattr(imagecodecs, "ultrahdr_encode"):
        raise RuntimeError(
            "imagecodecs ultrahdr extension is unavailable. Install full imagecodecs with ultrahdr support."
        )


def _encode_with_gain_map_api(
    sdr_base: np.ndarray,
    gain_map: np.ndarray,
    icc_profile: bytes | None,
) -> bytes:
    """Try legacy Ultra HDR API variants that accept explicit gain map arguments."""
    keyword_attempts = ["gainmap", "gain_map"]

    for gain_keyword in keyword_attempts:
        kwargs: dict[str, object] = {gain_keyword: gain_map}
        if icc_profile is not None:
            kwargs["metadata"] = {"icc_profile": icc_profile}

        try:
            return bytes(imagecodecs.ultrahdr_encode(sdr_base, **kwargs))
        except TypeError as exc:
            message = str(exc)

            if "metadata" in kwargs and "unexpected keyword argument 'metadata'" in message:
                kwargs.pop("metadata")
                try:
                    return bytes(imagecodecs.ultrahdr_encode(sdr_base, **kwargs))
                except TypeError as inner_exc:
                    if "unexpected keyword argument" not in str(inner_exc):
                        raise
                    continue

            if "unexpected keyword argument" in message:
                continue
            raise

    raise TypeError("ultrahdr_encode gain map keyword arguments are unsupported in this build.")


def _compose_rgba_half_from_linear(linear_sdr: np.ndarray, gain_map: np.ndarray) -> np.ndarray:
    """Build RGBA float16 HDR data from linear SDR and gain map for newer Ultra HDR API."""
    linear = np.asarray(linear_sdr, dtype=np.float32)
    if linear.ndim == 2:
        linear = np.repeat(linear[..., None], 3, axis=2)
    elif linear.ndim == 3 and linear.shape[2] >= 3:
        linear = linear[..., :3]
    else:
        raise ValueError("linear_sdr must be shape (H, W) or (H, W, C>=3).")

    gain = np.asarray(gain_map, dtype=np.float32)
    if gain.ndim == 3:
        gain = gain.mean(axis=2)

    if gain.shape != linear.shape[:2]:
        raise ValueError("gain_map dimensions must match linear_sdr dimensions.")

    boost = np.exp2((gain / 255.0) * 2.0)
    hdr_rgb = np.clip(linear * boost[..., None], 0.0, 65504.0)

    rgba = np.ones((linear.shape[0], linear.shape[1], 4), dtype=np.float16)
    rgba[..., :3] = hdr_rgb.astype(np.float16)
    return rgba


def encode_ultrahdr(
    sdr_base: np.ndarray,
    gain_map: np.ndarray,
    icc_profile: bytes | None,
    linear_sdr: np.ndarray | None = None,
) -> bytes:
    """
    Encode SDR base and gain map into a single Ultra HDR JPEG payload.

    Uses explicit gain map arguments when available, otherwise falls back to
    RGBA-halffloat encoding for newer imagecodecs Ultra HDR API variants.
    """
    _ensure_ultrahdr_available()

    try:
        return _encode_with_gain_map_api(sdr_base=sdr_base, gain_map=gain_map, icc_profile=icc_profile)
    except TypeError:
        pass

    if linear_sdr is None:
        raise RuntimeError(
            "Installed imagecodecs.ultrahdr_encode does not support explicit gain map arguments, "
            "and no linear_sdr fallback input was provided."
        )

    rgba_half = _compose_rgba_half_from_linear(linear_sdr=linear_sdr, gain_map=gain_map)
    return bytes(imagecodecs.ultrahdr_encode(rgba_half))
