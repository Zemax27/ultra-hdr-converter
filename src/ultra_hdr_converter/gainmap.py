"""Gain map validation and generation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

try:
    from cv2 import ximgproc as _ximgproc  # type: ignore[import-untyped]

    _HAS_XIMGPROC = True
except ImportError:
    _ximgproc = None
    _HAS_XIMGPROC = False

FloatArray = NDArray[np.float32]
UInt8Array = NDArray[np.uint8]

GRAYSCALE_NDIM = 2
COLOR_NDIM = 3
SINGLE_CHANNEL = 1
PERCENT_MAX = 100.0
NORMALIZATION_EPSILON = 1e-8


@dataclass(frozen=True)
class RadianceMapConfig:
    """Configuration for reflectance-aware radiance gain map generation."""

    resize_factor: float = 0.5
    guided_radius: int = 100
    guided_eps: float = 0.5
    clip_percentile_high: float = 99.5
    clip_percentile_low: float = 50.0


def validate_gain_map(gain_map: np.ndarray, sdr_shape: tuple[int, ...]) -> UInt8Array:
    """Validate and normalize gain map to uint8 with expected dimensions."""
    gain = np.asarray(gain_map)

    if gain.ndim not in (GRAYSCALE_NDIM, COLOR_NDIM):
        raise ValueError("Gain map must be 2D or 3D array.")

    if gain.shape[0] != sdr_shape[0] or gain.shape[1] != sdr_shape[1]:
        raise ValueError("Gain map height and width must match SDR base image.")

    if gain.ndim == COLOR_NDIM and gain.shape[2] not in (SINGLE_CHANNEL, COLOR_NDIM):
        raise ValueError("Gain map channels must be 1 or 3 when using a 3D array.")

    if gain.dtype != np.uint8:
        gain = np.clip(gain, 0, 255).astype(np.uint8)

    return np.asarray(gain, dtype=np.uint8)


def generate_log2_gain_map(luminance: np.ndarray, max_stops: float = 6.0) -> UInt8Array:
    """
    Create a simple single-channel gain map from CIE Y luminance.

    This is a deterministic baseline map for experimentation, not a full HDR tone mapping model.

    Args:
        luminance: 2-D array of linear luminance values (CIE Y from XYZ).
        max_stops: Maximum exposure stops to encode.

    Returns:
        Single-channel uint8 gain map.
    """
    luma = np.asarray(luminance, dtype=np.float32)
    if luma.ndim != GRAYSCALE_NDIM:
        raise ValueError("Luminance array must be 2-D (H, W).")

    safe_luma = np.maximum(luma, 1e-6)
    gain_stops = np.log2(safe_luma)
    gain_stops = np.clip(gain_stops, 0.0, max_stops)

    normalized = gain_stops / max_stops
    gain_map = np.rint(normalized * 255.0).astype(np.uint8)
    return np.asarray(gain_map, dtype=np.uint8)


def _resize_bilinear(image: np.ndarray, height: int, width: int) -> FloatArray:
    """Resize a single-channel image using bilinear interpolation."""
    if image.ndim != GRAYSCALE_NDIM:
        raise ValueError("Bilinear resize expects a 2D array.")

    src_h, src_w = image.shape
    if src_h == height and src_w == width:
        return image.astype(np.float32, copy=True)
    if height < 1 or width < 1:
        raise ValueError("Resize dimensions must be positive.")

    y = np.linspace(0, src_h - 1, height, dtype=np.float32)
    x = np.linspace(0, src_w - 1, width, dtype=np.float32)

    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)

    wy = y - y0
    wx = x - x0

    top_left = image[y0[:, None], x0[None, :]]
    top_right = image[y0[:, None], x1[None, :]]
    bottom_left = image[y1[:, None], x0[None, :]]
    bottom_right = image[y1[:, None], x1[None, :]]

    top = top_left * (1.0 - wx)[None, :] + top_right * wx[None, :]
    bottom = bottom_left * (1.0 - wx)[None, :] + bottom_right * wx[None, :]

    resized = top * (1.0 - wy)[:, None] + bottom * wy[:, None]
    return np.asarray(resized, dtype=np.float32)


def _box_filter(image: np.ndarray, radius: int) -> FloatArray:
    """Compute local window sums using a padded integral image."""
    if radius < 1:
        return image.astype(np.float32, copy=True)

    window = 2 * radius + 1
    padded = np.pad(image.astype(np.float32), ((radius, radius), (radius, radius)), mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)

    filtered = (
        integral[window:, window:]
        - integral[:-window, window:]
        - integral[window:, :-window]
        + integral[:-window, :-window]
    )
    return np.asarray(filtered, dtype=np.float32)


def _guided_filter_grayscale(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int,
    eps: float,
) -> FloatArray:
    """Run a grayscale guided filter approximation in pure NumPy."""
    if guide.shape != src.shape:
        raise ValueError("Guide and source arrays must share the same shape.")
    if guide.ndim != GRAYSCALE_NDIM:
        raise ValueError("Guided filter expects single-channel 2D arrays.")

    guide = guide.astype(np.float32, copy=False)
    src = src.astype(np.float32, copy=False)

    count = _box_filter(np.ones_like(guide, dtype=np.float32), radius)
    mean_guide = _box_filter(guide, radius) / count
    mean_src = _box_filter(src, radius) / count
    corr_guide = _box_filter(guide * guide, radius) / count
    corr_guide_src = _box_filter(guide * src, radius) / count

    variance_guide = corr_guide - mean_guide * mean_guide
    covariance = corr_guide_src - mean_guide * mean_src

    a = covariance / (variance_guide + eps)
    b = mean_src - a * mean_guide

    mean_a = _box_filter(a, radius) / count
    mean_b = _box_filter(b, radius) / count

    filtered = mean_a * guide + mean_b
    return np.asarray(filtered, dtype=np.float32)


def _robust_normalize(image: np.ndarray, low: float, high: float) -> UInt8Array:
    """Map illumination values to uint8 using percentile clipping in log-space."""
    log_image = np.log(np.maximum(image, 1e-6)).astype(np.float32)
    flat = log_image.ravel()

    low_value = float(np.percentile(flat, low))
    high_value = float(np.percentile(flat, high))
    if high_value - low_value < NORMALIZATION_EPSILON:
        return np.zeros_like(image, dtype=np.uint8)

    clipped = np.clip(log_image, low_value, high_value)
    normalized = (clipped - low_value) / (high_value - low_value)
    normalized_map = np.rint(normalized * 255.0).astype(np.uint8)
    return np.asarray(normalized_map, dtype=np.uint8)


def generate_radiance_gain_map(
    luminance: np.ndarray,
    config: RadianceMapConfig | None = None,
) -> UInt8Array:
    """
    Generate a reflectance-aware single-channel gain map from CIE Y luminance.

    Args:
        luminance: 2-D array of linear luminance values (CIE Y from XYZ).
        config: Optional radiance generation configuration.

    Returns:
        Single-channel uint8 gain map.
    """
    cfg = config or RadianceMapConfig()

    if not 0.0 < cfg.resize_factor <= 1.0:
        raise ValueError("Radiance resize_factor must be in the range (0.0, 1.0].")
    if cfg.guided_radius < 1:
        raise ValueError("Radiance guided_radius must be >= 1.")
    if cfg.guided_eps <= 0.0:
        raise ValueError("Radiance guided_eps must be > 0.")
    if not 0.0 <= cfg.clip_percentile_low < cfg.clip_percentile_high <= PERCENT_MAX:
        raise ValueError("Radiance clip percentiles must satisfy 0 <= low < high <= 100.")

    luma = np.asarray(luminance, dtype=np.float32)
    if luma.ndim != GRAYSCALE_NDIM:
        raise ValueError("Luminance array must be 2-D (H, W).")

    source_height, source_width = luma.shape
    if cfg.resize_factor < 1.0:
        target_height = max(1, int(round(source_height * cfg.resize_factor)))
        target_width = max(1, int(round(source_width * cfg.resize_factor)))
        luminance_for_filter = _resize_bilinear(luma, target_height, target_width)
    else:
        luminance_for_filter = luma

    log_luminance = np.log(np.maximum(luminance_for_filter, 1e-6)).astype(np.float32)

    # Edge-preserving smoothing via guided filter.
    # Use OpenCV ximgproc when available for speed; fall back to pure NumPy.
    if _HAS_XIMGPROC:
        smoothed = _ximgproc.guidedFilter(
            guide=log_luminance,
            src=log_luminance,
            radius=cfg.guided_radius,
            eps=cfg.guided_eps,
        )
    else:
        smoothed = _guided_filter_grayscale(log_luminance, log_luminance, cfg.guided_radius, cfg.guided_eps)

    illumination_small = np.exp(smoothed)

    if illumination_small.shape != (source_height, source_width):
        illumination = _resize_bilinear(illumination_small, source_height, source_width)
    else:
        illumination = illumination_small

    return _robust_normalize(illumination, cfg.clip_percentile_low, cfg.clip_percentile_high)
