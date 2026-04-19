from pathlib import Path

import imagecodecs
import numpy as np

from ultra_hdr_converter.color import linearize_from_icc
from ultra_hdr_converter.gainmap import RadianceMapConfig, generate_radiance_gain_map
from ultra_hdr_converter.io import decode_jpeg, extract_icc_profile, read_bytes


class RadianceMapEstimator:
    """
    Estimates a reflectance-aware radiance (illumination) map from a single RGB image
    using ICC-aware linearization and guided Retinex decomposition.

    Produces a stable single-channel 8-bit illumination map without hallucination.
    """

    def __init__(
        self,
        resizeFactor: float = 0.5,
        guidedRadius: int = 100,
        guidedEps: float = 0.5,
        clipPercentileHigh: float = 99.5,
        clipPercentileLow: float = 5,
    ):
        """Store radiance generation parameters for compatibility with legacy callers."""
        self._config = RadianceMapConfig(
            resize_factor=resizeFactor,
            guided_radius=guidedRadius,
            guided_eps=guidedEps,
            clip_percentile_high=clipPercentileHigh,
            clip_percentile_low=float(clipPercentileLow),
        )

    def computeRadianceMap(self, inputPath: str, outputPath: str):
        """
        Full pipeline.

        Args:
            inputPath: Input image path
            outputPath: Output grayscale JPG path

        Returns:
            Radiance map (8-bit)
        """
        jpeg_bytes = read_bytes(inputPath)
        sdr_array = decode_jpeg(jpeg_bytes)
        icc_profile = extract_icc_profile(jpeg_bytes)
        linear_sdr = linearize_from_icc(sdr_array, icc_profile, outdtype=np.float32)
        radiance_map = generate_radiance_gain_map(linear_sdr, config=self._config)

        output = Path(outputPath)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() == ".npy":
            np.save(output, radiance_map)
        else:
            imagecodecs.imwrite(str(output), radiance_map)

        return radiance_map
