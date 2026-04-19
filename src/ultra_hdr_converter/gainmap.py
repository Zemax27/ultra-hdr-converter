"""Gain map validation and generation helpers."""

from __future__ import annotations

import numpy as np


def validate_gain_map(gain_map: np.ndarray, sdr_shape: tuple[int, ...]) -> np.ndarray:
    """Validate and normalize gain map to uint8 with expected dimensions."""
    gain = np.asarray(gain_map)

    if gain.ndim not in (2, 3):
        raise ValueError("Gain map must be 2D or 3D array.")

    if gain.shape[0] != sdr_shape[0] or gain.shape[1] != sdr_shape[1]:
        raise ValueError("Gain map height and width must match SDR base image.")

    if gain.ndim == 3 and gain.shape[2] not in (1, 3):
        raise ValueError("Gain map channels must be 1 or 3 when using a 3D array.")

    if gain.dtype != np.uint8:
        gain = np.clip(gain, 0, 255).astype(np.uint8)

    return gain


def generate_log2_gain_map(linear_sdr: np.ndarray, max_stops: float = 6.0) -> np.ndarray:
    """
    Create a simple single-channel gain map from linear luminance.

    This is a deterministic baseline map for experimentation, not a full HDR tone mapping model.
    """
    linear = np.asarray(linear_sdr, dtype=np.float32)
    if linear.ndim == 3 and linear.shape[2] >= 3:
        luminance = (
            linear[..., 0] * 0.2126 + linear[..., 1] * 0.7152 + linear[..., 2] * 0.0722
        )
    elif linear.ndim == 2:
        luminance = linear
    else:
        raise ValueError("Linear SDR array must be shape (H, W) or (H, W, C>=3).")

    safe_luma = np.maximum(luminance, 1e-6)
    gain_stops = np.log2(safe_luma)
    gain_stops = np.clip(gain_stops, 0.0, max_stops)

    normalized = gain_stops / max_stops
    return np.rint(normalized * 255.0).astype(np.uint8)
