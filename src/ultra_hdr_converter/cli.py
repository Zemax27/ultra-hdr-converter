"""Command line interface for Ultra HDR conversion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .gainmap import GainMapConfig
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
        help="Optional gain map path (.npy or image). If omitted, a gain map is generated automatically.",
    )
    parser.add_argument(
        "--highlight-threshold",
        type=float,
        default=0.5,
        help="Linear luminance value where HDR boost begins (0.0-1.0).",
    )
    parser.add_argument(
        "--expansion-gamma",
        type=float,
        default=2.2,
        help="Exponent for non-linear highlight stretch.",
    )
    parser.add_argument(
        "--max-boost-factor",
        type=float,
        default=4.0,
        help="Maximum HDR multiplier for the brightest pixels.",
    )
    parser.add_argument(
        "--guided-radius",
        type=int,
        default=20,
        help="Guided filter radius for edge-aware smoothing.",
    )
    parser.add_argument(
        "--guided-eps",
        type=float,
        default=1e-3,
        help="Guided filter epsilon.",
    )
    parser.add_argument(
        "--bloom-weight",
        type=float,
        default=0.15,
        help="Weight of aesthetic bloom effect (0 to disable).",
    )
    return parser.parse_args()


def _validate_inputs(args: argparse.Namespace) -> None:
    """Validate that input files exist before starting conversion."""
    if not args.input_jpeg.is_file():
        print(f"Error: input file not found: {args.input_jpeg}", file=sys.stderr)
        sys.exit(1)
    if args.gain_map is not None and not args.gain_map.is_file():
        print(f"Error: gain map file not found: {args.gain_map}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    args = _parse_args()
    _validate_inputs(args)

    gain_map_config = GainMapConfig(
        highlight_threshold=args.highlight_threshold,
        expansion_gamma=args.expansion_gamma,
        max_boost_factor=args.max_boost_factor,
        guided_radius=args.guided_radius,
        guided_eps=args.guided_eps,
        bloom_weight=args.bloom_weight,
    )

    result = convert_jpeg_to_ultrahdr(
        input_jpeg=args.input_jpeg,
        output_jpeg=args.output_jpeg,
        gain_map_path=args.gain_map,
        gain_map_config=gain_map_config,
    )

    print(f"Wrote Ultra HDR JPEG: {result.output_path}")
    print(f"Gain map source: {result.gain_map_source}")
    print(f"ICC profile found: {result.has_icc}")


if __name__ == "__main__":
    main()
