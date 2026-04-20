"""Pipeline orchestration for Ultra HDR conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ultra_hdr_converter.color import extract_xyz_luminance
from ultra_hdr_converter.encoder import encode_ultrahdr
from ultra_hdr_converter.gainmap import (
    GainMapConfig,
    generate_gain_map,
    validate_gain_map,
)
from ultra_hdr_converter.io import decode_jpeg, extract_icc_profile, load_gain_map, read_bytes, write_bytes

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


def convert_jpeg_to_ultrahdr(
    input_jpeg: Path | str,
    output_jpeg: Path | str,
    gain_map_path: Path | str | None = None,
    gain_map_config: GainMapConfig | None = None,
    progress_callback: ProgressCallback | None = None,
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
    """
    _notify_progress(progress_callback, "Reading and decoding input JPEG", 1)
    input_bytes = read_bytes(input_jpeg)
    sdr_base = decode_jpeg(input_bytes)
    icc_profile = extract_icc_profile(input_bytes)

    if gain_map_path is not None:
        _notify_progress(progress_callback, "Loading external gain map", 2)
        gain_map = validate_gain_map(load_gain_map(gain_map_path))
        gain_map_source = "external"
        _notify_progress(progress_callback, "Skipping gain map generation", 3)
    else:
        _notify_progress(progress_callback, "Extracting luminance from SDR", 2)
        # Downsample to half resolution for faster luminance and gain map computation.
        # The encoder accepts gain maps of any size relative to the SDR JPEG.
        sdr_half = sdr_base[::2, ::2]
        luminance = extract_xyz_luminance(sdr_half, icc_profile)
        _notify_progress(progress_callback, "Generating highlight-targeted gain map", 3)
        gain_map = generate_gain_map(luminance, config=gain_map_config)
        gain_map = validate_gain_map(gain_map)
        gain_map_source = "generated"

    _notify_progress(progress_callback, "Encoding Ultra HDR metadata and container", 4)
    # API-4 composition: the original SDR JPEG is preserved byte-for-byte,
    # keeping its encoding quality and ICC profile intact.
    ultrahdr_bytes = encode_ultrahdr(
        sdr_jpeg=input_bytes,
        gain_map=gain_map,
    )
    _notify_progress(progress_callback, "Writing final output file", 5)
    write_bytes(output_jpeg, ultrahdr_bytes)

    return ConversionResult(
        output_path=Path(output_jpeg),
        has_icc=icc_profile is not None,
        gain_map_source=gain_map_source,
    )
