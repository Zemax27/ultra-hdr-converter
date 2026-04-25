"""Custom exceptions for the Ultra HDR conversion pipeline."""


class UltraHdrError(Exception):
    """Base exception for all Ultra HDR conversion errors."""


class AlreadyUltraHDRError(UltraHdrError):
    """Image already contains Ultra HDR metadata."""


class GainMapError(UltraHdrError):
    """Base exception for gain map validation and generation errors."""


class GainMapShapeMismatchError(GainMapError):
    """Gain map spatial dimensions (H, W) do not match the SDR base image."""

    def __init__(self, gain_map_shape: tuple[int, ...], sdr_shape: tuple[int, ...]) -> None:
        self.gain_map_shape = gain_map_shape
        self.sdr_shape = sdr_shape
        super().__init__(f"Gain map shape {gain_map_shape} does not match SDR base shape {sdr_shape[:2]}")


class GainMapDimensionError(GainMapError):
    """Gain map has invalid number of dimensions or channel count."""


class GainMapConfigError(GainMapError):
    """Gain map configuration parameter is out of valid range."""


class ColorTransformError(UltraHdrError):
    """ICC color management transform or profile operation failed."""


class JpegStructureError(UltraHdrError):
    """JPEG byte stream is missing required markers or is malformed."""
