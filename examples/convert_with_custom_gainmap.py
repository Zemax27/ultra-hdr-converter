"""Examples of using ultra_hdr_converter programmatically."""

from ultra_hdr_converter import (
    GainMapConfig,
    ImageFormat,
    convert_to_ultrahdr,
)


def convert_with_external_gain_map() -> None:
    """Convert using a pre-made gain map image."""
    result = convert_to_ultrahdr(
        input_path="input.jpg",
        output_path="output_ultrahdr.jpg",
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
    result = convert_to_ultrahdr(
        input_path="input.jpg",
        output_path="output_ultrahdr.jpg",
        gain_map_config=config,
    )
    print(result)


def convert_avif_keeping_the_coded_base() -> None:
    """Convert an AVIF photo, reusing its coded image instead of re-encoding it."""
    result = convert_to_ultrahdr(
        input_path="input.avif",
        output_path="output_ultrahdr.avif",
    )
    print(f"{result.input_format.value} -> {result.output_format.value}")


def convert_jpeg_to_avif() -> None:
    """Cross-container conversion: the output format follows the file suffix."""
    result = convert_to_ultrahdr(
        input_path="input.jpg",
        output_path="output_ultrahdr.avif",
    )
    print(result)


def convert_forcing_the_output_container() -> None:
    """Force a container explicitly, regardless of the output file suffix."""
    result = convert_to_ultrahdr(
        input_path="input.jpg",
        output_path="output_ultrahdr.bin",
        output_format=ImageFormat.AVIF,
    )
    print(result)


if __name__ == "__main__":
    convert_with_auto_generation()
