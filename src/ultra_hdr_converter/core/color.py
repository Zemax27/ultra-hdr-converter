"""Pure color math and array-shape helpers (no I/O or CMS dependencies)."""

from __future__ import annotations

import numpy as np
from numpy.typing import DTypeLike

from ultra_hdr_converter.errors import ColorTransformError

GRAYSCALE_NDIM = 2
COLOR_NDIM = 3
SINGLE_CHANNEL = 1


def extract_y_channel(xyz_array: np.ndarray, outdtype: DTypeLike = np.float32) -> np.ndarray:
    """Extract the Y (luminance) channel from a CIE XYZ array.

    Args:
        xyz_array: CIE XYZ array of shape (H, W, 3) with dtype float32.
        outdtype: Output floating dtype for the luminance array.

    Returns:
        2-D array of CIE Y (luminance) values with shape (H, W).

    Raises:
        ColorTransformError: If the array does not have at least 3 channels.
    """
    if xyz_array.ndim != COLOR_NDIM or xyz_array.shape[2] < COLOR_NDIM:
        raise ColorTransformError("XYZ array must be shape (H, W, C>=3).")
    return np.asarray(xyz_array[..., 1], dtype=outdtype)


def luminance_from_grayscale(sdr_array: np.ndarray, outdtype: DTypeLike = np.float32) -> np.ndarray:
    """Return grayscale pixel values normalized to [0.0, 1.0] and cast to the requested floating dtype.

    For integer input types (uint8, uint16, etc.), values are divided by the maximum
    representable value of the dtype to produce normalized floats. For floating-point
    inputs, values are passed through (assumed already normalized) and cast.

    Args:
        sdr_array: 2-D grayscale image of shape (H, W), any dtype.
        outdtype: Output floating dtype.

    Returns:
        2-D float array of shape (H, W) with values in [0.0, 1.0] for integer inputs.
    """
    arr = np.asarray(sdr_array)
    if np.issubdtype(arr.dtype, np.integer):
        max_val = np.iinfo(arr.dtype).max
        # Normalize to [0, 1] using float64 for accuracy, then cast to outdtype
        arr = (arr.astype(np.float64) / max_val).astype(outdtype)
    else:
        arr = arr.astype(outdtype, copy=False)
    return arr
