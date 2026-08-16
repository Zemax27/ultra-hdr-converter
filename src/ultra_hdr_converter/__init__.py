"""Ultra HDR conversion package."""

from .core.color import extract_y_channel, luminance_from_grayscale
from .core.color_cms import extract_xyz_luminance, linearize_from_icc
from .core.converter import ConversionResult, convert_to_ultrahdr
from .core.formats import SUPPORTED_SUFFIXES, ImageFormat, detect_format, is_supported_path
from .core.gain_map import GainMapConfig, generate_gain_map, validate_gain_map
from .core.image_io import has_embedded_gain_map, has_ultrahdr_metadata
from .core.iso21496 import GainMapMetadata
from .errors import (
    AlreadyUltraHDRError,
    AvifStructureError,
    ColorTransformError,
    GainMapConfigError,
    GainMapDimensionError,
    GainMapError,
    GainMapShapeMismatchError,
    ImageStructureError,
    JpegStructureError,
    UltraHdrError,
    UnsupportedFormatError,
)

__all__ = [
    "SUPPORTED_SUFFIXES",
    "AlreadyUltraHDRError",
    "AvifStructureError",
    "ColorTransformError",
    "ConversionResult",
    "GainMapConfig",
    "GainMapConfigError",
    "GainMapDimensionError",
    "GainMapError",
    "GainMapMetadata",
    "GainMapShapeMismatchError",
    "ImageFormat",
    "ImageStructureError",
    "JpegStructureError",
    "UltraHdrError",
    "UnsupportedFormatError",
    "convert_to_ultrahdr",
    "detect_format",
    "extract_xyz_luminance",
    "extract_y_channel",
    "generate_gain_map",
    "has_embedded_gain_map",
    "has_ultrahdr_metadata",
    "is_supported_path",
    "linearize_from_icc",
    "luminance_from_grayscale",
    "validate_gain_map",
]
