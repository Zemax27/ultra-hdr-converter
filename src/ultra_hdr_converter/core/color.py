"""Pure color math and array-shape helpers (no I/O or CMS dependencies)."""

from __future__ import annotations

import numpy as np
from numpy.typing import DTypeLike

from ultra_hdr_converter.errors import ColorTransformError

GRAYSCALE_NDIM = 2
COLOR_NDIM = 3
SINGLE_CHANNEL = 1

# Bit depth of a conventional SDR raster, and of the widest sample this
# pipeline handles.
SDR_BIT_DEPTH = 8
MAX_BIT_DEPTH = 16


def max_sample_value(bit_depth: int) -> int:
    """Return the largest sample value representable at a given bit depth.

    Args:
        bit_depth: Bits per sample (for example 8, 10 or 12).

    Returns:
        The maximum sample value, e.g. 1023 for 10-bit.

    Raises:
        ColorTransformError: If the bit depth is outside the supported range.
    """
    if not 1 <= bit_depth <= MAX_BIT_DEPTH:
        raise ColorTransformError(f"Unsupported sample bit depth {bit_depth}; expected 1-{MAX_BIT_DEPTH}.")
    return (1 << bit_depth) - 1


def rescale_sample_depth(image: np.ndarray, source_bit_depth: int, target_bit_depth: int) -> np.ndarray:
    """Rescale integer samples from one bit depth to another.

    Full-scale white maps to full-scale white, so a 10-bit value of 1023
    becomes 255 at 8 bits rather than being truncated.

    Args:
        image: Integer pixel array of any shape.
        source_bit_depth: Bits per sample actually used by the input.
        target_bit_depth: Bits per sample wanted in the output.

    Returns:
        Rescaled array, dtype ``uint8`` for targets up to 8 bits and ``uint16``
        above. Returned unchanged when the depths already match.

    Raises:
        ColorTransformError: If the array is not integer typed or a bit depth
            is out of range.
    """
    array = np.asarray(image)
    if not np.issubdtype(array.dtype, np.integer):
        raise ColorTransformError(f"Sample rescaling requires an integer array, got dtype {array.dtype}.")

    source_max = max_sample_value(source_bit_depth)
    target_max = max_sample_value(target_bit_depth)
    target_dtype = np.uint8 if target_bit_depth <= SDR_BIT_DEPTH else np.uint16

    if source_bit_depth == target_bit_depth:
        return np.asarray(array.astype(target_dtype, copy=False))

    scaled = array.astype(np.float32) * (target_max / source_max)
    return np.asarray(np.clip(np.rint(scaled, out=scaled), 0, target_max).astype(target_dtype))


def to_full_range(image: np.ndarray, bit_depth: int) -> np.ndarray:
    """Expand samples so they span the full range of their own dtype.

    Colour management treats an integer array as covering its dtype's whole
    range, so 10-bit samples stored in ``uint16`` must be stretched to 0-65535
    before any CMS transform — otherwise the image is read as almost black.

    Args:
        image: Integer pixel array of any shape.
        bit_depth: Bits per sample actually used by the input.

    Returns:
        The array rescaled to its dtype's full range.
    """
    dtype_bit_depth = np.iinfo(np.asarray(image).dtype).bits
    return rescale_sample_depth(image, bit_depth, dtype_bit_depth)


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
