import struct

import imagecodecs
import numpy as np
import pytest

from ultra_hdr_converter.core import avif_io
from ultra_hdr_converter.core.avif_encoder import (
    build_single_item_avif,
    encode_coded_image,
    encode_ultrahdr_avif,
)
from ultra_hdr_converter.errors import AvifStructureError

IMAGE_HEIGHT = 48
IMAGE_WIDTH = 64
GAIN_MAP_HEIGHT = 24
GAIN_MAP_WIDTH = 32
EXPECTED_TMAP_DIMG_COUNT = 2
EXPECTED_ITEM_COUNT = 3
# Mean absolute error tolerated after a lossy AVIF round trip of the gain map.
GAIN_MAP_ROUND_TRIP_TOLERANCE = 12


@pytest.fixture(scope="module")
def sdr_image() -> np.ndarray:
    rng = np.random.default_rng(11)
    return (rng.random((IMAGE_HEIGHT, IMAGE_WIDTH, 3)) * 255).astype(np.uint8)


@pytest.fixture(scope="module")
def gain_map() -> np.ndarray:
    rng = np.random.default_rng(12)
    return (rng.random((GAIN_MAP_HEIGHT, GAIN_MAP_WIDTH)) * 255).astype(np.uint8)


@pytest.fixture(scope="module")
def plain_avif(sdr_image: np.ndarray) -> bytes:
    return bytes(imagecodecs.avif_encode(sdr_image, level=80))


@pytest.fixture(scope="module")
def ultrahdr_avif(sdr_image: np.ndarray, gain_map: np.ndarray) -> bytes:
    base = encode_coded_image(sdr_image, quality=80)
    return encode_ultrahdr_avif(base_image=base, gain_map=gain_map, quality=90, max_content_boost=3.0)


# ---- Box parsing -------------------------------------------------------------


def test_parse_meta_reads_primary_item(plain_avif: bytes) -> None:
    meta = avif_io.parse_meta(plain_avif)

    assert meta.primary_item_id in meta.items
    primary = avif_io.get_primary_item(meta)
    assert primary.item_type == avif_io.ITEM_TYPE_AV01


def test_parse_meta_raises_without_meta_box() -> None:
    with pytest.raises(AvifStructureError, match="missing 'meta' box"):
        avif_io.parse_meta(b"\x00\x00\x00\x20ftypavif" + b"\x00" * 16)


def test_build_coded_image_reads_dimensions_and_properties(plain_avif: bytes) -> None:
    meta = avif_io.parse_meta(plain_avif)
    coded = avif_io.build_coded_image(plain_avif, meta, avif_io.get_primary_item(meta))

    assert coded.width == IMAGE_WIDTH
    assert coded.height == IMAGE_HEIGHT
    assert coded.av1c is not None
    assert coded.payload


def test_build_coded_image_rejects_grid_items(plain_avif: bytes) -> None:
    meta = avif_io.parse_meta(plain_avif)
    primary = avif_io.get_primary_item(meta)
    grid_item = avif_io.AvifItem(
        item_id=primary.item_id,
        item_type=avif_io.ITEM_TYPE_GRID,
        name="grid",
        is_hidden=False,
        extents=primary.extents,
        construction_method=primary.construction_method,
        properties=primary.properties,
    )

    with pytest.raises(AvifStructureError, match="grid"):
        avif_io.build_coded_image(plain_avif, meta, grid_item)


def test_read_item_payload_rejects_out_of_range_extent(plain_avif: bytes) -> None:
    meta = avif_io.parse_meta(plain_avif)
    primary = avif_io.get_primary_item(meta)
    broken = avif_io.AvifItem(
        item_id=primary.item_id,
        item_type=primary.item_type,
        name=primary.name,
        is_hidden=False,
        extents=((len(plain_avif) - 4, 999999),),
        construction_method=primary.construction_method,
        properties=primary.properties,
    )

    with pytest.raises(AvifStructureError, match="exceeds"):
        avif_io.read_item_payload(plain_avif, meta, broken)


# ---- av1C decoding and bit depth ---------------------------------------------


def _av1c(flags_byte: int) -> bytes:
    """Build a minimal av1C property box with the given sequence-header flags."""
    return struct.pack(">I", 12) + b"av1C" + bytes([0x81, 0x00, flags_byte, 0x00])


@pytest.mark.parametrize(
    ("flags", "expected_depth"),
    [(0x00, avif_io.DEPTH_8), (0x40, avif_io.DEPTH_10), (0x60, avif_io.DEPTH_12)],
)
def test_parse_av1c_reads_bit_depth(flags: int, expected_depth: int) -> None:
    assert avif_io.parse_av1c(_av1c(flags)).depth == expected_depth


def test_parse_av1c_detects_monochrome() -> None:
    config = avif_io.parse_av1c(_av1c(0x1C))

    assert config.is_monochrome is True
    assert config.channel_count == 1


def test_parse_av1c_defaults_when_box_is_missing() -> None:
    config = avif_io.parse_av1c(None)

    assert config.depth == avif_io.DEPTH_8
    assert config.is_monochrome is False


def test_parse_av1c_defaults_when_box_is_truncated() -> None:
    assert avif_io.parse_av1c(b"\x00\x00\x00\x09av1C\x81").depth == avif_io.DEPTH_8


def test_get_bit_depth_reports_eight_for_standard_avif(plain_avif: bytes) -> None:
    assert avif_io.get_bit_depth(plain_avif) == avif_io.DEPTH_8


def test_get_bit_depth_reports_ten_for_high_bit_depth_avif(sdr_image: np.ndarray) -> None:
    """A 10-bit AVIF must be reported as 10-bit, not as its uint16 storage width."""
    image10 = (sdr_image.astype(np.uint16) * 1023) // 255
    encoded = bytes(imagecodecs.avif_encode(image10, level=80, bitspersample=10))

    assert avif_io.get_bit_depth(encoded) == avif_io.DEPTH_10


def test_get_bit_depth_defaults_for_malformed_input() -> None:
    assert avif_io.get_bit_depth(b"garbage") == avif_io.DEPTH_8


# ---- Gain map detection ------------------------------------------------------


def test_has_gain_map_metadata_false_for_plain_avif(plain_avif: bytes) -> None:
    assert avif_io.has_gain_map_metadata(plain_avif) is False


def test_has_gain_map_metadata_true_for_ultrahdr_avif(ultrahdr_avif: bytes) -> None:
    assert avif_io.has_gain_map_metadata(ultrahdr_avif) is True


def test_has_gain_map_metadata_false_for_malformed_input() -> None:
    assert avif_io.has_gain_map_metadata(b"not an avif file") is False


def test_find_gain_map_item_returns_second_dimg_reference(ultrahdr_avif: bytes) -> None:
    meta = avif_io.parse_meta(ultrahdr_avif)
    gain_map_item = avif_io.find_gain_map_item(meta)

    assert gain_map_item is not None
    assert gain_map_item.item_type == avif_io.ITEM_TYPE_AV01
    assert gain_map_item.is_hidden is True


def test_find_gain_map_item_returns_none_without_tmap(plain_avif: bytes) -> None:
    assert avif_io.find_gain_map_item(avif_io.parse_meta(plain_avif)) is None


# ---- Decoding and ICC --------------------------------------------------------


def test_decode_avif_round_trips_dimensions(plain_avif: bytes) -> None:
    decoded = avif_io.decode_avif(plain_avif)
    assert decoded.shape[:2] == (IMAGE_HEIGHT, IMAGE_WIDTH)


def test_decode_avif_raises_on_invalid_data() -> None:
    with pytest.raises(AvifStructureError, match="decoding failed"):
        avif_io.decode_avif(b"definitely not an avif")


def test_extract_icc_profile_returns_none_when_absent(plain_avif: bytes) -> None:
    assert avif_io.extract_icc_profile(plain_avif) is None


def test_extract_icc_profile_round_trips(sdr_image: np.ndarray) -> None:
    """An ICC profile attached at encode time must be readable back out."""
    icc = bytes(imagecodecs.cms_profile("srgb"))
    coded = encode_coded_image(sdr_image, quality=80, icc_profile=icc)
    avif_bytes = build_single_item_avif(coded)

    assert avif_io.extract_icc_profile(avif_bytes) == icc


def test_extract_icc_profile_returns_none_for_malformed_input() -> None:
    assert avif_io.extract_icc_profile(b"garbage") is None


# ---- Round trip through the muxer --------------------------------------------


def test_gain_map_item_round_trips_through_container(ultrahdr_avif: bytes, gain_map: np.ndarray) -> None:
    """The gain map extracted from the muxed file must match what went in."""
    meta = avif_io.parse_meta(ultrahdr_avif)
    gain_map_item = avif_io.find_gain_map_item(meta)
    assert gain_map_item is not None

    coded = avif_io.build_coded_image(ultrahdr_avif, meta, gain_map_item)
    decoded = avif_io.decode_avif(build_single_item_avif(coded))

    assert decoded.shape[:2] == gain_map.shape
    # AVIF encoding is lossy at this quality, so compare with a tolerance.
    mean_error = np.abs(decoded.astype(np.int16) - gain_map.astype(np.int16)).mean()
    assert mean_error < GAIN_MAP_ROUND_TRIP_TOLERANCE


def test_base_image_survives_muxing(ultrahdr_avif: bytes) -> None:
    """Gain-map-unaware decoders must still see the SDR base as the primary image."""
    decoded = avif_io.decode_avif(ultrahdr_avif)
    assert decoded.shape[:2] == (IMAGE_HEIGHT, IMAGE_WIDTH)


def test_muxed_file_declares_expected_items(ultrahdr_avif: bytes) -> None:
    meta = avif_io.parse_meta(ultrahdr_avif)

    assert len(meta.items) == EXPECTED_ITEM_COUNT
    tmap_items = [item for item in meta.items.values() if item.item_type == avif_io.ITEM_TYPE_TMAP]
    assert len(tmap_items) == 1

    dimg_targets = meta.references[avif_io.REF_TYPE_DIMG][tmap_items[0].item_id]
    assert len(dimg_targets) == EXPECTED_TMAP_DIMG_COUNT
    # The base image must come first so libavif pairs it with the gain map.
    assert dimg_targets[0] == meta.primary_item_id
