"""Gain map generation and validation helpers (pure core)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from ultra_hdr_converter.errors import GainMapConfigError, GainMapDimensionError

try:
    import cv2 as _cv2  # type: ignore[import-not-found]

    _HAS_CV2 = True
except ImportError:
    _cv2 = None
    _HAS_CV2 = False

try:
    from cv2 import ximgproc as _ximgproc

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
    """Configuration for highlight-targeted gain map generation.

    Attributes:
        highlight_threshold: Linear luminance value where HDR boost begins (0.0-1.0).
        expansion_gamma: Exponent for non-linear highlight stretch (> 0).
        max_boost_factor: Maximum HDR multiplier for the brightest pixels (> 0).
        guided_radius: Guided filter radius for edge-aware smoothing (>= 1).
        guided_eps: Guided filter epsilon (> 0).
        bloom_weight: Weight of aesthetic bloom effect (0.0-1.0).
    """

    highlight_threshold: float = 0.5
    expansion_gamma: float = 2.2
    max_boost_factor: float = 3.0
    guided_radius: int = 20
    guided_eps: float = 1e-3
    bloom_weight: float = 0.15


def validate_gain_map(gain_map: np.ndarray) -> UInt8Array:
    """Validate and normalize gain map to uint8.

    Args:
        gain_map: Raw gain map array of shape (H, W) or (H, W, 1) or (H, W, 3).

    Returns:
        uint8 gain map of shape (H, W) or (H, W, C).

    Raises:
        GainMapDimensionError: If the array has invalid dimensions or channel count.
    """
    gain = np.asarray(gain_map)

    if gain.ndim not in (GRAYSCALE_NDIM, COLOR_NDIM):
        raise GainMapDimensionError(f"Gain map must be 2D or 3D array, got {gain.ndim}D.")

    if gain.ndim == COLOR_NDIM and gain.shape[2] not in (SINGLE_CHANNEL, COLOR_NDIM):
        raise GainMapDimensionError(f"Gain map channels must be 1 or 3 when using a 3D array, got {gain.shape[2]}.")

    if gain.dtype != np.uint8:
        gain = np.clip(gain, 0, 255).astype(np.uint8)

    return gain


# ---- Internal helpers --------------------------------------------------------


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> FloatArray:
    """Hermite interpolation for smooth highlight masking.

    Args:
        edge0: Lower edge of the smooth transition.
        edge1: Upper edge of the smooth transition.
        x: Input array of shape (H, W), dtype float32.

    Returns:
        Smooth-stepped array of shape (H, W), dtype float32, values in [0, 1].
    """
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-6), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _box_filter_mean(image: np.ndarray, radius: int) -> FloatArray:
    """Compute local window means using a padded integral image.

    Args:
        image: 2-D input array of shape (H, W), dtype float32.
        radius: Half-size of the square window (>= 1).

    Returns:
        Smoothed array of shape (H, W), dtype float32.
    """
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
    """Run a grayscale guided filter in pure NumPy.

    Args:
        guide: Guide image of shape (H, W), dtype float32.
        src: Source image of shape (H, W), dtype float32.
        radius: Filter radius in pixels (>= 1).
        eps: Regularization epsilon (> 0).

    Returns:
        Filtered output of shape (H, W), dtype float32.

    Raises:
        GainMapDimensionError: If guide and source shapes differ or are not 2-D.
    """
    if guide.shape != src.shape:
        raise GainMapDimensionError("Guide and source arrays must share the same shape.")
    if guide.ndim != GRAYSCALE_NDIM:
        raise GainMapDimensionError("Guided filter expects single-channel 2D arrays.")

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
    """Gaussian blur using cv2 when available, else 3-pass box filter approximation.

    Args:
        image: 2-D input array of shape (H, W), dtype float32.
        sigma: Gaussian standard deviation in pixels.

    Returns:
        Blurred array of shape (H, W), dtype float32.
    """
    if _HAS_CV2:
        return np.asarray(
            _cv2.GaussianBlur(np.asarray(image, dtype=np.float32), (0, 0), sigmaX=sigma), dtype=np.float32
        )

    radius = max(1, int(round(sigma)))
    result = _box_filter_mean(image, radius)
    result = _box_filter_mean(result, radius)
    return _box_filter_mean(result, radius)


# ---- Public API --------------------------------------------------------------


def _validate_config(cfg: GainMapConfig) -> None:
    """Raise GainMapConfigError if any configuration parameter is out of range.

    Args:
        cfg: Gain map generation configuration.

    Raises:
        GainMapConfigError: If any parameter is invalid.
    """
    if not 0.0 < cfg.highlight_threshold < 1.0:
        raise GainMapConfigError("highlight_threshold must be in the range (0.0, 1.0).")
    if cfg.expansion_gamma <= 0.0:
        raise GainMapConfigError("expansion_gamma must be > 0.")
    if cfg.max_boost_factor <= 0.0:
        raise GainMapConfigError("max_boost_factor must be > 0.")
    if cfg.guided_radius < 1:
        raise GainMapConfigError("guided_radius must be >= 1.")
    if cfg.guided_eps <= 0.0:
        raise GainMapConfigError("guided_eps must be > 0.")
    if not 0.0 <= cfg.bloom_weight <= 1.0:
        raise GainMapConfigError("bloom_weight must be in the range [0.0, 1.0].")


@lru_cache(maxsize=16)
def _cached_validate_config(cfg: GainMapConfig) -> None:
    """Cached validation for frozen GainMapConfig dataclass instances."""
    _validate_config(cfg)


def generate_gain_map(
    luminance: np.ndarray,
    config: GainMapConfig | None = None,
) -> UInt8Array:
    """Generate a highlight-targeted single-channel gain map from CIE Y luminance.

    Uses inverse tone mapping to expand compressed highlights while leaving
    midtones and shadows untouched.

    Args:
        luminance: 2-D array of linear luminance values (CIE Y from XYZ),
            shape (H, W), dtype float32.
        config: Optional generation configuration.

    Returns:
        Single-channel uint8 gain map of shape (H, W), dtype uint8.

    Raises:
        GainMapDimensionError: If luminance is not 2-D.
        GainMapConfigError: If any configuration parameter is out of range.
    """
    cfg = config or GainMapConfig()
    _cached_validate_config(cfg)

    luma = np.asarray(luminance, dtype=np.float32)
    if luma.ndim != GRAYSCALE_NDIM:
        raise GainMapDimensionError(f"Luminance array must be 2-D (H, W), got {luma.ndim}D.")

    highlight_mask = _smoothstep(cfg.highlight_threshold, 1.0, luma)

    # Core HDR expansion with in-place operations to avoid memory copies
    stretched = np.power(luma, cfg.expansion_gamma, dtype=np.float32)
    stretched *= cfg.max_boost_factor
    stretched *= highlight_mask

    y_hdr = stretched  # Reuse 'stretched' buffer
    y_hdr += luma

    # Gain map calculation: log2((y_hdr + 1e-6) / (luma + 1e-6))
    y_hdr += 1e-6
    temp_luma = luma + 1e-6
    y_hdr /= temp_luma

    gain_raw = np.log2(y_hdr, out=y_hdr)  # Reuse buffer again

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

    if cfg.bloom_weight > 0.0:
        bloom = _gaussian_blur(gain_guided, sigma=cfg.guided_radius * 0.5)
        gain_final = gain_guided * (1.0 - cfg.bloom_weight) + bloom * cfg.bloom_weight
    else:
        gain_final = gain_guided

    gain_min = float(gain_final.min())
    gain_max = float(gain_final.max())
    if gain_max - gain_min < NORMALIZATION_EPSILON:
        return np.zeros(luma.shape, dtype=np.uint8)

    gain_final -= gain_min
    gain_final *= 255.0 / (gain_max - gain_min)
    return np.rint(gain_final).astype(np.uint8)
