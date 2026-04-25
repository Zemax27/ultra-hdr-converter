"""Pipeline orchestration for Ultra HDR conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ultra_hdr_converter.core.color_cms import extract_xyz_luminance
from ultra_hdr_converter.core.gain_map import (
    GainMapConfig,
    generate_gain_map,
    validate_gain_map,
)
from ultra_hdr_converter.core.jpeg_io import (
    decode_jpeg,
    extract_icc_profile,
    load_gain_map,
    read_bytes,
    write_bytes,
    has_ultrahdr_metadata,
    extract_mpf_gain_map,
)
from ultra_hdr_converter.core.ultrahdr_encoder import encode_ultrahdr
from ultra_hdr_converter.errors import GainMapShapeMismatchError, AlreadyUltraHDRError

ProgressCallback = Callable[[str, int, int], None]
PROGRESS_STEP_COUNT = 5


@dataclass(frozen=True)
class ConversionResult:
    """Summary of one conversion run."""

    output_path: Path
    has_icc: bool
    gain_map_source: str


def _notify_progress(progress_callback: ProgressCallback | None, message: str, step: int) -> None:
    """Emit a coarse-grained progress update when a callback is available."""
    if progress_callback is not None:
        progress_callback(message, step, PROGRESS_STEP_COUNT)


def _validate_gain_map_shape(gain_map: object, sdr_shape: tuple[int, ...]) -> None:
    """Validate gain map spatial dimensions against the SDR base image.

    Args:
        gain_map: Validated gain map array (2D or 3D with channel axis).
        sdr_shape: Shape of the SDR base image (H, W[, C]).

    Raises:
        GainMapShapeMismatchError: If the spatial dimensions differ.
    """
    gm = np.asarray(gain_map)
    gm_spatial = (gm.shape[0], gm.shape[1])
    sdr_spatial = (sdr_shape[0], sdr_shape[1])
    if gm_spatial != sdr_spatial:
        raise GainMapShapeMismatchError(gain_map_shape=gm.shape, sdr_shape=sdr_shape)


def convert_jpeg_to_ultrahdr(
    input_jpeg: Path | str,
    output_jpeg: Path | str,
    gain_map_path: Path | str | None = None,
    gain_map_config: GainMapConfig | None = None,
    progress_callback: ProgressCallback | None = None,
    jpeg_quality: int = 95,
    max_content_boost: float | None = None,
) -> ConversionResult:
    """Convert an SDR JPEG to Ultra HDR JPEG using either external or generated gain map.

    Args:
        input_jpeg: Path to the SDR base JPEG.
        output_jpeg: Path for the Ultra HDR output JPEG.
        gain_map_path: Optional external gain map file.
        gain_map_config: Optional configuration for generated gain maps.
        progress_callback: Optional callback invoked at coarse pipeline phase boundaries.

    Returns:
        Summary of the completed conversion.

    Raises:
        GainMapShapeMismatchError: If an external gain map's spatial dimensions
            do not match the SDR base image.
    """
    _notify_progress(progress_callback, "Reading and decoding input JPEG", 1)
    input_bytes = read_bytes(input_jpeg)
    
    if has_ultrahdr_metadata(input_bytes):
        raise AlreadyUltraHDRError(f"File {input_jpeg} is already an Ultra HDR image.")
        
    sdr_base = decode_jpeg(input_bytes)
    icc_profile = extract_icc_profile(input_bytes)

    if gain_map_path is not None:
        _notify_progress(progress_callback, "Loading external gain map", 2)
        gain_map = validate_gain_map(load_gain_map(gain_map_path))
        _validate_gain_map_shape(gain_map, sdr_base.shape)
        gain_map_source = "external"
        _notify_progress(progress_callback, "Skipping gain map generation", 3)
        actual_boost = max_content_boost if max_content_boost is not None else 3.0
    else:
        mpf_bytes = extract_mpf_gain_map(input_bytes)
        if mpf_bytes is not None:
            _notify_progress(progress_callback, "Extracting embedded MPF gain map", 2)
            gain_map = validate_gain_map(decode_jpeg(mpf_bytes))
            _validate_gain_map_shape(gain_map, sdr_base.shape)
            gain_map_source = "embedded"
            _notify_progress(progress_callback, "Skipping gain map generation", 3)
            actual_boost = max_content_boost if max_content_boost is not None else 3.0
        else:
            _notify_progress(progress_callback, "Extracting luminance from SDR", 2)
            sdr_half = sdr_base[::2, ::2]
            luminance = extract_xyz_luminance(sdr_half, icc_profile)
            _notify_progress(progress_callback, "Generating highlight-targeted gain map", 3)
            gain_map = generate_gain_map(luminance, config=gain_map_config)
            gain_map = validate_gain_map(gain_map)
            gain_map_source = "generated"
            actual_boost = (
                gain_map_config.max_boost_factor if gain_map_config is not None else GainMapConfig().max_boost_factor
            )

    _notify_progress(progress_callback, "Encoding Ultra HDR metadata and container", 4)
    ultrahdr_bytes = encode_ultrahdr(
        sdr_jpeg=input_bytes,
        gain_map=gain_map,
        jpeg_quality=jpeg_quality,
        max_content_boost=actual_boost,
    )
    _notify_progress(progress_callback, "Writing final output file", 5)
    write_bytes(output_jpeg, ultrahdr_bytes)

    return ConversionResult(
        output_path=Path(output_jpeg),
        has_icc=icc_profile is not None,
        gain_map_source=gain_map_source,
    )
