import struct

import numpy as np
import pytest

from ultra_hdr_converter.core import avif_io
from ultra_hdr_converter.core.avif_encoder import (
    SRGB_CICP,
    CodedImage,
    build_auxc,
    build_ispe,
    build_nclx,
    build_pixi,
    encode_coded_image,
    encode_ultrahdr_avif,
    miaf_brand,
)
from ultra_hdr_converter.core.avif_io import Av1Config
from ultra_hdr_converter.core.iso21496 import GainMapMetadata

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 40
EXPECTED_TMAP_PAYLOAD_SIZE = 1 + 2 + 2 + 1 + 16 + 40  # version + versions + flags + headrooms + channel
EXPECTED_BOOST = 4.0
DEPTH_8 = 8
DEPTH_10 = 10
DEPTH_12 = 12
RGB_CHANNELS = 3
ALPHA_ITEM_COUNT = 4
FULL_RANGE_FLAG = 0x80


@pytest.fixture(scope="module")
def sdr_image() -> np.ndarray:
    rng = np.random.default_rng(21)
    return (rng.random((IMAGE_HEIGHT, IMAGE_WIDTH, 3)) * 255).astype(np.uint8)


@pytest.fixture(scope="module")
def base_image(sdr_image: np.ndarray) -> CodedImage:
    return encode_coded_image(sdr_image, quality=80)


# ---- Property builders -------------------------------------------------------


def test_build_ispe_encodes_dimensions() -> None:
    box = build_ispe(IMAGE_WIDTH, IMAGE_HEIGHT)

    assert box[4:8] == b"ispe"
    assert struct.unpack(">II", box[12:20]) == (IMAGE_WIDTH, IMAGE_HEIGHT)


def test_build_pixi_lists_every_channel_depth() -> None:
    box = build_pixi(RGB_CHANNELS, DEPTH_8)

    assert box[4:8] == b"pixi"
    assert box[12] == RGB_CHANNELS
    assert box[13:16] == bytes([DEPTH_8] * RGB_CHANNELS)


def test_build_nclx_encodes_cicp_and_full_range() -> None:
    box = build_nclx(*SRGB_CICP)

    assert box[4:8] == b"colr"
    assert box[8:12] == b"nclx"
    assert struct.unpack(">HHH", box[12:18]) == SRGB_CICP
    assert box[18] == FULL_RANGE_FLAG


def test_build_auxc_carries_the_alpha_urn() -> None:
    box = build_auxc()

    assert box[4:8] == b"auxC"
    assert b"auxiliary:alpha" in box


# ---- MIAF brand selection ----------------------------------------------------


@pytest.mark.parametrize(
    ("config", "expected_brand"),
    [
        (Av1Config(DEPTH_8, False, True, True), b"MA1B"),
        (Av1Config(DEPTH_8, False, False, False), b"MA1A"),
        (Av1Config(DEPTH_10, False, True, True), b"MA1B"),
        (Av1Config(DEPTH_10, False, False, False), b"MA1A"),
        (Av1Config(DEPTH_8, True, True, True), None),
        (Av1Config(DEPTH_12, False, False, False), None),
    ],
)
def test_miaf_brand_selection(config: Av1Config, expected_brand: bytes | None) -> None:
    """MIAF profiles cover 8/10-bit 4:2:0 and 4:4:4 only."""
    assert miaf_brand(config) == expected_brand


# ---- Container composition ---------------------------------------------------


def test_encode_ultrahdr_avif_declares_the_tmap_brand(base_image: CodedImage) -> None:
    """ISO/IEC 23008-12:2024 requires the 'tmap' brand when a tone map item exists."""
    result = encode_ultrahdr_avif(base_image=base_image, gain_map=np.zeros((16, 20), dtype=np.uint8))

    ftyp_end = struct.unpack(">I", result[:4])[0]
    assert b"tmap" in result[8:ftyp_end]
    assert result[8:12] == b"avif"


def test_encode_ultrahdr_avif_marks_gain_map_hidden(base_image: CodedImage) -> None:
    result = encode_ultrahdr_avif(base_image=base_image, gain_map=np.zeros((16, 20), dtype=np.uint8))
    meta = avif_io.parse_meta(result)

    gain_map_item = avif_io.find_gain_map_item(meta)
    assert gain_map_item is not None
    assert gain_map_item.is_hidden is True
    # The base image stays the primary item so SDR viewers are unaffected.
    assert meta.primary_item_id != gain_map_item.item_id


def test_encode_ultrahdr_avif_writes_expected_tmap_payload(base_image: CodedImage) -> None:
    result = encode_ultrahdr_avif(
        base_image=base_image,
        gain_map=np.zeros((16, 20), dtype=np.uint8),
        max_content_boost=EXPECTED_BOOST,
    )
    meta = avif_io.parse_meta(result)
    tmap_item = next(item for item in meta.items.values() if item.item_type == avif_io.ITEM_TYPE_TMAP)
    payload = avif_io.read_item_payload(result, meta, tmap_item)

    assert len(payload) == EXPECTED_TMAP_PAYLOAD_SIZE
    assert payload == GainMapMetadata(max_content_boost=EXPECTED_BOOST).to_tone_map_payload()


def test_encode_ultrahdr_avif_accepts_multichannel_gain_map(base_image: CodedImage) -> None:
    """A 3-channel gain map is reduced to its first channel, as the JPEG path does."""
    gain_map = np.zeros((16, 20, 3), dtype=np.uint8)
    result = encode_ultrahdr_avif(base_image=base_image, gain_map=gain_map)

    assert avif_io.has_gain_map_metadata(result) is True


def test_encode_ultrahdr_avif_carries_alpha_plane(base_image: CodedImage) -> None:
    """An alpha auxiliary image must survive muxing instead of being dropped."""
    alpha = encode_coded_image(np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 255, dtype=np.uint8), quality=80)
    result = encode_ultrahdr_avif(
        base_image=base_image,
        gain_map=np.zeros((16, 20), dtype=np.uint8),
        alpha_image=alpha,
    )
    meta = avif_io.parse_meta(result)

    assert len(meta.items) == ALPHA_ITEM_COUNT
    assert avif_io.find_alpha_item(meta) is not None


def test_encode_ultrahdr_avif_embeds_icc_profile(sdr_image: np.ndarray) -> None:
    icc = b"\x00" * 128 + b"fake-icc-profile"
    base = encode_coded_image(sdr_image, quality=80, icc_profile=icc)
    result = encode_ultrahdr_avif(base_image=base, gain_map=np.zeros((16, 20), dtype=np.uint8))

    assert avif_io.extract_icc_profile(result) == icc


def test_encode_ultrahdr_avif_iloc_offsets_point_at_payloads(base_image: CodedImage) -> None:
    """Every declared extent must resolve to the bytes actually stored in mdat."""
    result = encode_ultrahdr_avif(base_image=base_image, gain_map=np.zeros((16, 20), dtype=np.uint8))
    meta = avif_io.parse_meta(result)

    base_item = meta.items[meta.primary_item_id]
    assert avif_io.read_item_payload(result, meta, base_item) == base_image.payload

    for item in meta.items.values():
        for offset, length in item.extents:
            assert offset + length <= len(result)
