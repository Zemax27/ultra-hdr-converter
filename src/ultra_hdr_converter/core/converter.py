"""Pipeline orchestration for Ultra HDR conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ultra_hdr_converter.core.avif_encoder import (
    encode_coded_image,
    encode_ultrahdr_avif,
    extract_base_image,
)
from ultra_hdr_converter.core.color import SDR_BIT_DEPTH, rescale_sample_depth, to_full_range
from ultra_hdr_converter.core.color_cms import extract_xyz_luminance
from ultra_hdr_converter.core.formats import ImageFormat, detect_format, resolve_output_format
from ultra_hdr_converter.core.gain_map import (
    GRAYSCALE_NDIM,
    GainMapConfig,
    generate_gain_map,
    validate_gain_map,
)
from ultra_hdr_converter.core.image_io import (
    decode_image,
    extract_embedded_gain_map,
    extract_icc_profile,
    has_ultrahdr_metadata,
    load_gain_map,
    probe_bit_depth,
    read_bytes,
    write_bytes,
)
from ultra_hdr_converter.core.jpeg_encoder import encode_ultrahdr_jpeg
from ultra_hdr_converter.core.jpeg_io import encode_jpeg
from ultra_hdr_converter.errors import (
    AlreadyUltraHDRError,
    GainMapShapeMismatchError,
    UnsupportedFormatError,
)

ProgressCallback = Callable[[str, int, int], None]
PROGRESS_STEP_COUNT = 5
QUALITY_RANGE = (0, 100)
COLOR_NDIM = 3
RGB_CHANNELS = 3
DEFAULT_EXTERNAL_BOOST = 3.0

# Gain map sources, in the priority order the pipeline resolves them.
GAIN_MAP_SOURCE_EXTERNAL = "external"
GAIN_MAP_SOURCE_EMBEDDED = "embedded"
GAIN_MAP_SOURCE_GENERATED = "generated"


@dataclass(frozen=True)
class ConversionResult:
    """Summary of one conversion run."""

    output_path: Path
    has_icc: bool
    gain_map_source: str
    input_format: ImageFormat
    output_format: ImageFormat


def _notify_progress(progress_callback: ProgressCallback | None, message: str, step: int) -> None:
    """Emit a coarse-grained progress update when a callback is available."""
    if progress_callback is not None:
        progress_callback(message, step, PROGRESS_STEP_COUNT)


def _drop_alpha(image: np.ndarray) -> np.ndarray:
    """Return the colour channels of an image, discarding any alpha channel.

    Luminance analysis and JPEG encoding both operate on colour channels only.
    """
    if image.ndim == COLOR_NDIM and image.shape[2] > RGB_CHANNELS:
        return image[..., :RGB_CHANNELS]
    return image


def _validate_gain_map_shape(gain_map: np.ndarray, sdr_shape: tuple[int, ...]) -> None:
    """Validate gain map spatial dimensions against the SDR base image.

    Args:
        gain_map: Validated gain map array (2D or 3D with channel axis).
        sdr_shape: Shape of the SDR base image (H, W[, C]).

    Raises:
        GainMapShapeMismatchError: If the spatial dimensions differ.
    """
    gm = np.asarray(gain_map)
    if gm.ndim < GRAYSCALE_NDIM or len(sdr_shape) < GRAYSCALE_NDIM:
        raise GainMapShapeMismatchError(gain_map_shape=gm.shape, sdr_shape=sdr_shape)
    gm_spatial = (gm.shape[0], gm.shape[1])
    sdr_spatial = (sdr_shape[0], sdr_shape[1])
    if gm_spatial != sdr_spatial:
        raise GainMapShapeMismatchError(gain_map_shape=gm.shape, sdr_shape=sdr_shape)


def _validate_parameters(quality: int, max_content_boost: float | None, input_path: Path, output_path: Path) -> None:
    """Validate caller-supplied parameters before any heavy work begins.

    Raises:
        ValueError: If a parameter is out of range or input and output collide.
    """
    if not QUALITY_RANGE[0] <= quality <= QUALITY_RANGE[1]:
        raise ValueError(f"quality must be between {QUALITY_RANGE[0]} and {QUALITY_RANGE[1]}, got {quality}")
    if max_content_boost is not None and max_content_boost <= 0:
        raise ValueError(f"max_content_boost must be positive, got {max_content_boost}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output cannot be the same file.")


def _resolve_gain_map(
    input_bytes: bytes,
    input_format: ImageFormat,
    sdr_base: np.ndarray,
    bit_depth: int,
    icc_profile: bytes | None,
    gain_map_path: Path | str | None,
    gain_map_config: GainMapConfig | None,
    progress_callback: ProgressCallback | None,
) -> tuple[np.ndarray, str]:
    """Resolve the gain map to package, following the documented source priority.

    Returns:
        The validated gain map array and the name of the source it came from.
    """
    if gain_map_path is not None:
        _notify_progress(progress_callback, "Loading external gain map", 2)
        gain_map = validate_gain_map(load_gain_map(gain_map_path))
        _validate_gain_map_shape(gain_map, sdr_base.shape)
        _notify_progress(progress_callback, "Skipping gain map generation", 3)
        return gain_map, GAIN_MAP_SOURCE_EXTERNAL

    embedded = extract_embedded_gain_map(input_bytes, input_format)
    if embedded is not None:
        _notify_progress(progress_callback, "Extracting embedded gain map", 2)
        gain_map = validate_gain_map(embedded)
        _validate_gain_map_shape(gain_map, sdr_base.shape)
        _notify_progress(progress_callback, "Skipping gain map generation", 3)
        return gain_map, GAIN_MAP_SOURCE_EMBEDDED

    _notify_progress(progress_callback, "Extracting luminance from SDR", 2)
    # Colour management reads an integer array as spanning its dtype's whole
    # range, so deeper-than-8-bit samples must be expanded first; otherwise a
    # 10-bit image is measured as almost black and the gain map comes out empty.
    sdr_half = to_full_range(_drop_alpha(sdr_base)[::2, ::2], bit_depth)
    luminance = extract_xyz_luminance(sdr_half, icc_profile)
    _notify_progress(progress_callback, "Generating highlight-targeted gain map", 3)
    gain_map = validate_gain_map(generate_gain_map(luminance, config=gain_map_config))
    return gain_map, GAIN_MAP_SOURCE_GENERATED


def _encode_output(
    output_format: ImageFormat,
    input_bytes: bytes,
    input_format: ImageFormat,
    sdr_base: np.ndarray,
    bit_depth: int,
    icc_profile: bytes | None,
    gain_map: np.ndarray,
    quality: int,
    max_content_boost: float,
) -> bytes:
    """Package the SDR base and gain map into the requested container.

    When the output container matches the input, the original compressed base
    image is reused verbatim so no generation loss is introduced — and a 10- or
    12-bit AVIF keeps its full precision. Otherwise the decoded raster is
    re-encoded into the target container.

    Raises:
        UnsupportedFormatError: If the requested output container is not supported.
    """
    match output_format:
        case ImageFormat.JPEG:
            if input_format is ImageFormat.JPEG:
                sdr_jpeg = input_bytes
            else:
                # JPEG output is always 8-bit: a deeper array would produce a
                # 12-bit JPEG that most viewers cannot open.
                sdr_8bit = rescale_sample_depth(_drop_alpha(sdr_base), bit_depth, SDR_BIT_DEPTH)
                sdr_jpeg = encode_jpeg(sdr_8bit, quality=quality, icc_profile=icc_profile)
            return encode_ultrahdr_jpeg(
                sdr_jpeg=sdr_jpeg,
                gain_map=gain_map,
                quality=quality,
                max_content_boost=max_content_boost,
            )

        case ImageFormat.AVIF:
            if input_format is ImageFormat.AVIF:
                base_image, alpha_image = extract_base_image(input_bytes)
            else:
                base_image = encode_coded_image(
                    sdr_base,
                    quality=quality,
                    icc_profile=icc_profile,
                    bit_depth=bit_depth,
                )
                alpha_image = None
            return encode_ultrahdr_avif(
                base_image=base_image,
                gain_map=gain_map,
                quality=quality,
                max_content_boost=max_content_boost,
                alpha_image=alpha_image,
            )

        case unsupported:
            raise UnsupportedFormatError(f"Cannot write {unsupported.value} output.")


def convert_to_ultrahdr(
    input_path: Path | str,
    output_path: Path | str,
    gain_map_path: Path | str | None = None,
    gain_map_config: GainMapConfig | None = None,
    progress_callback: ProgressCallback | None = None,
    quality: int = 95,
    max_content_boost: float | None = None,
    output_format: ImageFormat | None = None,
) -> ConversionResult:
    """Convert an SDR image to a gain map encoded Ultra HDR image.

    JPEG and AVIF are supported for both input and output, in any combination.
    The gain map source is determined by the following priority:

    1. If the input already carries ISO 21496-1 gain map metadata, raise
       ``AlreadyUltraHDRError`` so callers can skip it.
    2. If *gain_map_path* is provided, the external file is used.
    3. If the input embeds an auxiliary gain map image but lacks the required
       metadata (a JPEG MPF secondary image), that gain map is reused and
       proper XMP + ISO 21496-1 metadata is written.
    4. Otherwise a gain map is generated from the SDR luminance data.

    The compressed base image is preserved byte-for-byte whenever the output
    container matches the input container, so re-packaging never re-compresses
    the photo.

    Args:
        input_path: Path to the SDR base image (JPEG or AVIF).
        output_path: Path for the Ultra HDR output image.
        gain_map_path: Optional external gain map file.
        gain_map_config: Optional configuration for generated gain maps.
        progress_callback: Optional callback invoked at coarse pipeline phase boundaries.
        quality: Encoder quality (0-100) for the gain map, and for the base
            image when a container change forces it to be re-encoded.
        max_content_boost: Maximum HDR content boost in stops.  When ``None``,
            defaults to the config's ``max_boost_factor`` for generated maps
            or 3.0 for external/embedded maps.
        output_format: Container to write.  When ``None``, it is taken from the
            output file suffix, falling back to the input container.

    Returns:
        Summary of the completed conversion.

    Raises:
        AlreadyUltraHDRError: If the input already contains gain map metadata.
        GainMapShapeMismatchError: If an external gain map's spatial dimensions
            do not match the SDR base image.
        UnsupportedFormatError: If the input or output container is unsupported.
        ValueError: If quality is not in 0-100, if max_content_boost is
            non-positive, or if input_path and output_path refer to the same file.
    """
    resolved_input = Path(input_path)
    resolved_output = Path(output_path)
    _validate_parameters(quality, max_content_boost, resolved_input, resolved_output)

    _notify_progress(progress_callback, "Reading and decoding input image", 1)
    input_bytes = read_bytes(resolved_input)
    input_format = detect_format(input_bytes)

    if has_ultrahdr_metadata(input_bytes, input_format):
        raise AlreadyUltraHDRError(f"File {input_path} is already an Ultra HDR image.")

    sdr_base = decode_image(input_bytes, input_format)
    bit_depth = probe_bit_depth(input_bytes, input_format)
    icc_profile = extract_icc_profile(input_bytes, input_format)

    gain_map, gain_map_source = _resolve_gain_map(
        input_bytes=input_bytes,
        input_format=input_format,
        sdr_base=sdr_base,
        bit_depth=bit_depth,
        icc_profile=icc_profile,
        gain_map_path=gain_map_path,
        gain_map_config=gain_map_config,
        progress_callback=progress_callback,
    )

    # Generated maps are built to the configured boost; supplied maps have no
    # intrinsic boost, so an explicit value (or the documented default) is used.
    if gain_map_source == GAIN_MAP_SOURCE_GENERATED:
        actual_boost = (gain_map_config or GainMapConfig()).max_boost_factor
    else:
        actual_boost = max_content_boost if max_content_boost is not None else DEFAULT_EXTERNAL_BOOST

    target_format = resolve_output_format(resolved_output, output_format, input_format)

    _notify_progress(progress_callback, "Encoding Ultra HDR metadata and container", 4)
    ultrahdr_bytes = _encode_output(
        output_format=target_format,
        input_bytes=input_bytes,
        input_format=input_format,
        sdr_base=sdr_base,
        bit_depth=bit_depth,
        icc_profile=icc_profile,
        gain_map=gain_map,
        quality=quality,
        max_content_boost=actual_boost,
    )

    _notify_progress(progress_callback, "Writing final output file", 5)
    write_bytes(resolved_output, ultrahdr_bytes)

    return ConversionResult(
        output_path=resolved_output,
        has_icc=icc_profile is not None,
        gain_map_source=gain_map_source,
        input_format=input_format,
        output_format=target_format,
    )
