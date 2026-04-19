"""Gain map validation and generation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

try:
    import cv2 as _cv2  # type: ignore[import-untyped]

    _HAS_CV2 = True
except ImportError:
    _cv2 = None
    _HAS_CV2 = False

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
    guided_eps: float = 1e-3
    clip_percentile_high: float = 99.5
    clip_percentile_low: float = 50.0


def validate_gain_map(gain_map: np.ndarray) -> UInt8Array:
    """Validate and normalize gain map to uint8."""
    gain = np.asarray(gain_map)

    if gain.ndim not in (GRAYSCALE_NDIM, COLOR_NDIM):
        raise ValueError("Gain map must be 2D or 3D array.")

    if gain.ndim == COLOR_NDIM and gain.shape[2] not in (SINGLE_CHANNEL, COLOR_NDIM):
        raise ValueError("Gain map channels must be 1 or 3 when using a 3D array.")

    if gain.dtype != np.uint8:
        gain = np.clip(gain, 0, 255).astype(np.uint8)

    return gain


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

    np.maximum(luma, 1e-6, out=luma)
    np.log2(luma, out=luma)
    np.clip(luma, 0.0, max_stops, out=luma)
    luma *= 255.0 / max_stops
    return np.rint(luma).astype(np.uint8)


def _resize_bilinear(image: np.ndarray, height: int, width: int) -> FloatArray:
    """Resize a single-channel image using bilinear interpolation."""
    if image.ndim != GRAYSCALE_NDIM:
        raise ValueError("Bilinear resize expects a 2D array.")

    src_h, src_w = image.shape
    if src_h == height and src_w == width:
        return np.asarray(image, dtype=np.float32)
    if height < 1 or width < 1:
        raise ValueError("Resize dimensions must be positive.")

    if _HAS_CV2:
        resized = _cv2.resize(np.asarray(image, dtype=np.float32), (width, height), interpolation=_cv2.INTER_LINEAR)
        return np.asarray(resized, dtype=np.float32)

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


def _box_filter_mean(image: np.ndarray, radius: int) -> FloatArray:
    """Compute local window means using a padded integral image."""
    if radius < 1:
        return np.array(image, dtype=np.float32, copy=True)

    window = 2 * radius + 1
    padded = np.pad(np.asarray(image, dtype=np.float32), ((radius, radius), (radius, radius)), mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant")
    integral.cumsum(axis=0, out=integral)
    integral.cumsum(axis=1, out=integral)

    filtered = (
        integral[window:, window:]
        - integral[:-window, window:]
        - integral[window:, :-window]
        + integral[:-window, :-window]
    )
    filtered *= 1.0 / (window * window)
    return filtered


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

    guide = np.asarray(guide, dtype=np.float32)
    src = np.asarray(src, dtype=np.float32)
    self_guided = guide is src

    mean_guide = _box_filter_mean(guide, radius)
    # Compute variance in-place: var = mean(guide^2) - mean(guide)^2
    variance_guide = _box_filter_mean(guide * guide, radius)
    variance_guide -= mean_guide * mean_guide

    if self_guided:
        # When guide == src, covariance equals variance; skip redundant box_filter_mean calls.
        # a = var / (var + eps), computed in-place.
        a = variance_guide.copy()
        a /= variance_guide + eps
        # b = mean_guide * (1 - a), computed in-place.
        b = np.subtract(1.0, a)
        b *= mean_guide
    else:
        mean_src = _box_filter_mean(src, radius)
        covariance = _box_filter_mean(guide * src, radius)
        covariance -= mean_guide * mean_src
        a = covariance
        a /= variance_guide + eps
        b = mean_src - a * mean_guide

    mean_a = _box_filter_mean(a, radius)
    mean_b = _box_filter_mean(b, radius)

    # filtered = mean_a * guide + mean_b, in-place.
    mean_a *= guide
    mean_a += mean_b
    return mean_a


def _robust_normalize(image: np.ndarray, low: float, high: float) -> UInt8Array:
    """Map illumination values to uint8 using percentile clipping in log-space."""
    result = np.maximum(image, 1e-6, out=np.empty_like(image, dtype=np.float32))
    np.log(result, out=result)

    low_value, high_value = np.percentile(result, [low, high])
    spread = high_value - low_value
    if spread < NORMALIZATION_EPSILON:
        return np.zeros(image.shape, dtype=np.uint8)

    np.clip(result, low_value, high_value, out=result)
    result -= low_value
    result *= 255.0 / spread
    return np.rint(result).astype(np.uint8)


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

    # Compute log-luminance in-place to avoid an extra allocation.
    log_luminance = np.maximum(luminance_for_filter, 1e-6, out=np.empty_like(luminance_for_filter))
    np.log(log_luminance, out=log_luminance)

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
