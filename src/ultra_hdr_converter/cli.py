"""Command line interface for Ultra HDR conversion."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .pipeline import convert_jpeg_to_ultrahdr


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="uhdr-convert",
        description="Convert SDR JPEG + gain map to Ultra HDR JPEG using imagecodecs.",
    )
    parser.add_argument("input_jpeg", type=Path, help="Input SDR JPEG file.")
    parser.add_argument("output_jpeg", type=Path, help="Output Ultra HDR JPEG file.")
    parser.add_argument(
        "--gain-map",
        type=Path,
        default=None,
        help="Optional gain map path (.npy or image). If omitted, a baseline map is generated.",
    )
    parser.add_argument(
        "--linear-dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Output dtype for ICC linearization stage.",
    )
    parser.add_argument(
        "--save-linear-npy",
        type=Path,
        default=None,
        help="Optional path to save linearized SDR array as .npy.",
    )
    parser.add_argument(
        "--no-embed-icc",
        action="store_true",
        help="Do not embed original ICC profile in Ultra HDR metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    linear_dtype = np.float32 if args.linear_dtype == "float32" else np.float64

    result = convert_jpeg_to_ultrahdr(
        input_jpeg=args.input_jpeg,
        output_jpeg=args.output_jpeg,
        gain_map_path=args.gain_map,
        linear_outdtype=linear_dtype,
        embed_icc_profile=not args.no_embed_icc,
        save_linear_npy=args.save_linear_npy,
    )

    print(f"Wrote Ultra HDR JPEG: {result.output_path}")
    print(f"Gain map source: {result.gain_map_source}")
    print(f"Linear dtype: {result.linear_dtype}")
    print(f"Embedded ICC: {result.embedded_icc}")


if __name__ == "__main__":
    main()
