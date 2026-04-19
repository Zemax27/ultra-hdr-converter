"""Pipeline orchestration for Ultra HDR conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import DTypeLike

from .color import extract_xyz_luminance, linearize_from_icc
from .encoder import encode_ultrahdr
from .gainmap import (
    RadianceMapConfig,
    generate_log2_gain_map,
    generate_radiance_gain_map,
    validate_gain_map,
)
from .io import decode_jpeg, extract_icc_profile, load_gain_map, read_bytes, write_bytes


@dataclass(frozen=True)
class ConversionResult:
    """Summary of one conversion run."""

    output_path: Path
    embedded_icc: bool
    gain_map_source: str
    linear_dtype: str


def convert_jpeg_to_ultrahdr(
    input_jpeg: Path | str,
    output_jpeg: Path | str,
    gain_map_path: Path | str | None = None,
    generated_gain_map_method: Literal["log2", "radiance"] = "radiance",
    radiance_config: RadianceMapConfig | None = None,
    linear_outdtype: DTypeLike = np.float32,
    embed_icc_profile: bool = True,
    save_linear_npy: Path | str | None = None,
) -> ConversionResult:
    """Convert an SDR JPEG to Ultra HDR JPEG using either external or generated gain map."""
    input_bytes = read_bytes(input_jpeg)
    sdr_base = decode_jpeg(input_bytes)
    icc_profile = extract_icc_profile(input_bytes)

    if save_linear_npy is not None:
        linear_sdr = linearize_from_icc(sdr_base, icc_profile, outdtype=linear_outdtype)
        linear_path = Path(save_linear_npy)
        linear_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(linear_path, linear_sdr)

    if generated_gain_map_method not in {"log2", "radiance"}:
        raise ValueError("generated_gain_map_method must be 'log2' or 'radiance'.")

    if gain_map_path is not None:
        gain_map = validate_gain_map(load_gain_map(gain_map_path))
        gain_map_source = "external"
    else:
        # Downsample to half resolution for faster luminance and gain map computation.
        # The encoder accepts gain maps of any size relative to the SDR JPEG.
        sdr_half = sdr_base[::2, ::2]
        luminance = extract_xyz_luminance(sdr_half, icc_profile)

        if generated_gain_map_method == "radiance":
            gain_map = generate_radiance_gain_map(luminance, config=radiance_config)
            gain_map_source = "generated-radiance"
        else:
            gain_map = generate_log2_gain_map(luminance)
            gain_map_source = "generated-log2"

        gain_map = validate_gain_map(gain_map)

    # API-4 composition: the original SDR JPEG is preserved byte-for-byte,
    # keeping its encoding quality and ICC profile intact.
    ultrahdr_bytes = encode_ultrahdr(
        sdr_jpeg=input_bytes,
        gain_map=gain_map,
    )
    write_bytes(output_jpeg, ultrahdr_bytes)

    return ConversionResult(
        output_path=Path(output_jpeg),
        embedded_icc=icc_profile is not None,
        gain_map_source=gain_map_source,
        linear_dtype=str(np.dtype(linear_outdtype)),
    )
