"""Examples of using ultra_hdr_converter programmatically."""

from ultra_hdr_converter import (
    GainMapConfig,
    convert_jpeg_to_ultrahdr,
)


def convert_with_external_gain_map() -> None:
    """Convert using a pre-made gain map image."""
    result = convert_jpeg_to_ultrahdr(
        input_jpeg="input.jpg",
        output_jpeg="output_ultrahdr.jpg",
        gain_map_path="gain_map.png",
    )
    print(result)


def convert_with_auto_generation() -> None:
    """Convert with automatic highlight-targeted gain map generation."""
    config = GainMapConfig(
        highlight_threshold=0.5,
        expansion_gamma=2.2,
        max_boost_factor=4.0,
    )
    result = convert_jpeg_to_ultrahdr(
        input_jpeg="input.jpg",
        output_jpeg="output_ultrahdr.jpg",
        gain_map_config=config,
    )
    print(result)


if __name__ == "__main__":
    convert_with_auto_generation()
