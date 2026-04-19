"""Minimal script to convert SDR JPEG + custom gain map into Ultra HDR JPEG."""

from ultra_hdr_converter.pipeline import convert_jpeg_to_ultrahdr


def main() -> None:
    result = convert_jpeg_to_ultrahdr(
        input_jpeg="input.jpg",
        output_jpeg="output_ultrahdr.jpg",
        gain_map_path="gain_map.png",
        linear_outdtype="float32",
    )
    print(result)


if __name__ == "__main__":
    main()
