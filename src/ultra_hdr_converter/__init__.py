"""Ultra HDR conversion package."""

from .core.color import extract_y_channel, luminance_from_grayscale
from .core.color_cms import extract_xyz_luminance, linearize_from_icc
from .core.converter import ConversionResult, convert_jpeg_to_ultrahdr
from .core.gain_map import GainMapConfig, generate_gain_map, validate_gain_map
from .core.jpeg_io import has_mpf_secondary_image, has_ultrahdr_metadata
from .errors import (
    AlreadyUltraHDRError,
    ColorTransformError,
    GainMapConfigError,
    GainMapDimensionError,
    GainMapError,
    GainMapShapeMismatchError,
    JpegStructureError,
    UltraHdrError,
)

__all__ = [
    "AlreadyUltraHDRError",
    "ColorTransformError",
    "ConversionResult",
    "GainMapConfig",
    "GainMapConfigError",
    "GainMapDimensionError",
    "GainMapError",
    "GainMapShapeMismatchError",
    "JpegStructureError",
    "UltraHdrError",
    "convert_jpeg_to_ultrahdr",
    "extract_xyz_luminance",
    "extract_y_channel",
    "generate_gain_map",
    "has_mpf_secondary_image",
    "has_ultrahdr_metadata",
    "linearize_from_icc",
    "luminance_from_grayscale",
    "validate_gain_map",
]
