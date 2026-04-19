"""Pipeline orchestration for Ultra HDR conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import DTypeLike

from .color import linearize_from_icc
from .encoder import encode_ultrahdr
from .gainmap import generate_log2_gain_map, validate_gain_map
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
    linear_outdtype: DTypeLike = np.float32,
    embed_icc_profile: bool = True,
    save_linear_npy: Path | str | None = None,
) -> ConversionResult:
    """
    Convert an SDR JPEG to Ultra HDR JPEG using either external or generated gain map.
    """
    input_bytes = read_bytes(input_jpeg)
    sdr_base = decode_jpeg(input_bytes)
    icc_profile = extract_icc_profile(input_bytes)

    linear_sdr = linearize_from_icc(sdr_base, icc_profile, outdtype=linear_outdtype)

    if save_linear_npy is not None:
        linear_path = Path(save_linear_npy)
        linear_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(linear_path, linear_sdr)

    if gain_map_path is not None:
        gain_map = validate_gain_map(load_gain_map(gain_map_path), sdr_base.shape)
        gain_map_source = "external"
    else:
        gain_map = generate_log2_gain_map(linear_sdr)
        gain_map = validate_gain_map(gain_map, sdr_base.shape)
        gain_map_source = "generated"

    icc_for_output = icc_profile if embed_icc_profile else None
    ultrahdr_bytes = encode_ultrahdr(sdr_base=sdr_base, gain_map=gain_map, icc_profile=icc_for_output)
    write_bytes(output_jpeg, ultrahdr_bytes)

    return ConversionResult(
        output_path=Path(output_jpeg),
        embedded_icc=icc_for_output is not None,
        gain_map_source=gain_map_source,
        linear_dtype=str(np.dtype(linear_sdr.dtype)),
    )
