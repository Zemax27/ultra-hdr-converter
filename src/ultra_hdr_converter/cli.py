"""Command line interface for Ultra HDR conversion."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .gainmap import RadianceMapConfig
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
        "--generated-gain-map",
        choices=("log2", "radiance"),
        default="radiance",
        help="Algorithm used when --gain-map is omitted.",
    )
    parser.add_argument(
        "--radiance-resize-factor",
        type=float,
        default=0.5,
        help="Downscale factor for radiance generation in the range (0, 1].",
    )
    parser.add_argument(
        "--radiance-guided-radius",
        type=int,
        default=100,
        help="Guided filter radius for radiance generation.",
    )
    parser.add_argument(
        "--radiance-guided-eps",
        type=float,
        default=1e-3,
        help="Guided filter epsilon for radiance generation.",
    )
    parser.add_argument(
        "--radiance-clip-low",
        type=float,
        default=50.0,
        help="Low percentile for radiance normalization.",
    )
    parser.add_argument(
        "--radiance-clip-high",
        type=float,
        default=99.5,
        help="High percentile for radiance normalization.",
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
    radiance_config = RadianceMapConfig(
        resize_factor=args.radiance_resize_factor,
        guided_radius=args.radiance_guided_radius,
        guided_eps=args.radiance_guided_eps,
        clip_percentile_low=args.radiance_clip_low,
        clip_percentile_high=args.radiance_clip_high,
    )

    result = convert_jpeg_to_ultrahdr(
        input_jpeg=args.input_jpeg,
        output_jpeg=args.output_jpeg,
        gain_map_path=args.gain_map,
        generated_gain_map_method=args.generated_gain_map,
        radiance_config=radiance_config,
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
