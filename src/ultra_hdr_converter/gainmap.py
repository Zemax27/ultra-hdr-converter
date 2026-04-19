"""Gain map generation and validation helpers."""

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
NORMALIZATION_EPSILON = 1e-6


@dataclass(frozen=True)
class GainMapConfig:
    """Configuration for highlight-targeted gain map generation."""

    highlight_threshold: float = 0.5
    expansion_gamma: float = 2.2
    max_boost_factor: float = 4.0
    guided_radius: int = 20
    guided_eps: float = 1e-3
    bloom_weight: float = 0.15


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


# ---- Internal helpers --------------------------------------------------------


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> FloatArray:
    """Hermite interpolation for smooth highlight masking."""
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-6), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


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
    """Run a grayscale guided filter in pure NumPy."""
    if guide.shape != src.shape:
        raise ValueError("Guide and source arrays must share the same shape.")
    if guide.ndim != GRAYSCALE_NDIM:
        raise ValueError("Guided filter expects single-channel 2D arrays.")

    guide = np.asarray(guide, dtype=np.float32)
    src = np.asarray(src, dtype=np.float32)

    mean_guide = _box_filter_mean(guide, radius)
    variance_guide = _box_filter_mean(guide * guide, radius)
    variance_guide -= mean_guide * mean_guide

    mean_src = _box_filter_mean(src, radius)
    covariance = _box_filter_mean(guide * src, radius)
    covariance -= mean_guide * mean_src
    a = covariance
    a /= variance_guide + eps
    b = mean_src - a * mean_guide

    mean_a = _box_filter_mean(a, radius)
    mean_b = _box_filter_mean(b, radius)

    mean_a *= guide
    mean_a += mean_b
    return mean_a


def _gaussian_blur(image: np.ndarray, sigma: float) -> FloatArray:
    """Gaussian blur using cv2 when available, else 3-pass box filter approximation."""
    if _HAS_CV2:
        return np.asarray(_cv2.GaussianBlur(np.asarray(image, dtype=np.float32), (0, 0), sigmaX=sigma), dtype=np.float32)

    # Three-pass box filter approximates a Gaussian with effective sigma ~ radius.
    radius = max(1, int(round(sigma)))
    result = _box_filter_mean(image, radius)
    result = _box_filter_mean(result, radius)
    return _box_filter_mean(result, radius)


# ---- Public API --------------------------------------------------------------


def generate_gain_map(
    luminance: np.ndarray,
    config: GainMapConfig | None = None,
) -> UInt8Array:
    """
    Generate a highlight-targeted single-channel gain map from CIE Y luminance.

    Uses inverse tone mapping to expand compressed highlights while leaving
    midtones and shadows untouched.

    Args:
        luminance: 2-D array of linear luminance values (CIE Y from XYZ).
        config: Optional generation configuration.

    Returns:
        Single-channel uint8 gain map.
    """
    cfg = config or GainMapConfig()

    if not 0.0 < cfg.highlight_threshold < 1.0:
        raise ValueError("highlight_threshold must be in the range (0.0, 1.0).")
    if cfg.expansion_gamma <= 0.0:
        raise ValueError("expansion_gamma must be > 0.")
    if cfg.max_boost_factor <= 0.0:
        raise ValueError("max_boost_factor must be > 0.")
    if cfg.guided_radius < 1:
        raise ValueError("guided_radius must be >= 1.")
    if cfg.guided_eps <= 0.0:
        raise ValueError("guided_eps must be > 0.")
    if not 0.0 <= cfg.bloom_weight <= 1.0:
        raise ValueError("bloom_weight must be in the range [0.0, 1.0].")

    luma = np.asarray(luminance, dtype=np.float32)
    if luma.ndim != GRAYSCALE_NDIM:
        raise ValueError("Luminance array must be 2-D (H, W).")

    # 1. Soft highlight mask via smoothstep.
    highlight_mask = _smoothstep(cfg.highlight_threshold, 1.0, luma)

    # 2. Non-linear highlight expansion (inverse tone mapping).
    stretched = np.power(luma, cfg.expansion_gamma, dtype=np.float32) * cfg.max_boost_factor
    y_hdr = luma + stretched * highlight_mask

    # 3. Log2 gain ratio between synthetic HDR and SDR.
    gain_raw = np.log2((y_hdr + 1e-6) / (luma + 1e-6))

    # 4. Edge-aware smoothing guided by SDR luminance.
    if _HAS_XIMGPROC:
        gain_guided = np.asarray(
            _ximgproc.guidedFilter(
                guide=np.asarray(luma, dtype=np.float32),
                src=np.asarray(gain_raw, dtype=np.float32),
                radius=cfg.guided_radius,
                eps=cfg.guided_eps,
            ),
            dtype=np.float32,
        )
    else:
        gain_guided = _guided_filter_grayscale(luma, gain_raw, cfg.guided_radius, cfg.guided_eps)

    # 5. Aesthetic bloom blending.
    if cfg.bloom_weight > 0.0:
        bloom = _gaussian_blur(gain_guided, sigma=cfg.guided_radius * 0.5)
        gain_final = gain_guided * (1.0 - cfg.bloom_weight) + bloom * cfg.bloom_weight
    else:
        gain_final = gain_guided

    # 6. Normalize to uint8.
    gain_min = float(gain_final.min())
    gain_max = float(gain_final.max())
    if gain_max - gain_min < NORMALIZATION_EPSILON:
        return np.zeros(luma.shape, dtype=np.uint8)

    gain_final -= gain_min
    gain_final *= 255.0 / (gain_max - gain_min)
    return np.rint(gain_final).astype(np.uint8)
