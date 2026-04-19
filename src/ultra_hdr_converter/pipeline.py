"""Pipeline orchestration for Ultra HDR conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .color import extract_xyz_luminance
from .encoder import encode_ultrahdr
from .gainmap import (
    GainMapConfig,
    generate_gain_map,
    validate_gain_map,
)
from .io import decode_jpeg, extract_icc_profile, load_gain_map, read_bytes, write_bytes


@dataclass(frozen=True)
class ConversionResult:
    """Summary of one conversion run."""

    output_path: Path
    has_icc: bool
    gain_map_source: str


def convert_jpeg_to_ultrahdr(
    input_jpeg: Path | str,
    output_jpeg: Path | str,
    gain_map_path: Path | str | None = None,
    gain_map_config: GainMapConfig | None = None,
) -> ConversionResult:
    """Convert an SDR JPEG to Ultra HDR JPEG using either external or generated gain map."""
    input_bytes = read_bytes(input_jpeg)
    sdr_base = decode_jpeg(input_bytes)
    icc_profile = extract_icc_profile(input_bytes)

    if gain_map_path is not None:
        gain_map = validate_gain_map(load_gain_map(gain_map_path))
        gain_map_source = "external"
    else:
        # Downsample to half resolution for faster luminance and gain map computation.
        # The encoder accepts gain maps of any size relative to the SDR JPEG.
        sdr_half = sdr_base[::2, ::2]
        luminance = extract_xyz_luminance(sdr_half, icc_profile)
        gain_map = generate_gain_map(luminance, config=gain_map_config)
        gain_map = validate_gain_map(gain_map)
        gain_map_source = "generated"

    # API-4 composition: the original SDR JPEG is preserved byte-for-byte,
    # keeping its encoding quality and ICC profile intact.
    ultrahdr_bytes = encode_ultrahdr(
        sdr_jpeg=input_bytes,
        gain_map=gain_map,
    )
    write_bytes(output_jpeg, ultrahdr_bytes)

    return ConversionResult(
        output_path=Path(output_jpeg),
        has_icc=icc_profile is not None,
        gain_map_source=gain_map_source,
    )
