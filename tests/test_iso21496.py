import struct

import pytest

from ultra_hdr_converter.core.iso21496 import (
    DEFAULT_OFFSET,
    GainMapMetadata,
    float_to_fraction,
)

# GainMapVersion (4) + flags (1) + two headroom rationals (16) + one channel (40).
EXPECTED_METADATA_SIZE = 61
EXPECTED_TONE_MAP_SIZE = EXPECTED_METADATA_SIZE + 1
EXPECTED_BOOST = 3.0
OFFSET_DENOMINATOR = 64
USE_BASE_COLOUR_SPACE_BIT = 0x40


def test_float_to_fraction_handles_zero() -> None:
    assert float_to_fraction(0.0) == (0, 1)


def test_float_to_fraction_approximates_offsets() -> None:
    assert float_to_fraction(DEFAULT_OFFSET) == (1, OFFSET_DENOMINATOR)


def test_float_to_fraction_handles_negative_values() -> None:
    numerator, denominator = float_to_fraction(-2.5)
    assert numerator / denominator == pytest.approx(-2.5)


def test_metadata_has_the_size_libavif_expects() -> None:
    """libavif reads a fixed-size single-channel structure; a size drift breaks it."""
    assert len(GainMapMetadata(max_content_boost=EXPECTED_BOOST).to_bytes()) == EXPECTED_METADATA_SIZE


def test_metadata_starts_with_zero_versions() -> None:
    payload = GainMapMetadata(max_content_boost=EXPECTED_BOOST).to_bytes()
    assert struct.unpack(">HH", payload[:4]) == (0, 0)


def test_metadata_encodes_boost_as_headroom_and_gain_max() -> None:
    payload = GainMapMetadata(max_content_boost=EXPECTED_BOOST).to_bytes()

    alternate_headroom = struct.unpack(">II", payload[13:21])
    gain_map_max = struct.unpack(">iI", payload[29:37])

    assert alternate_headroom == (3, 1)
    assert gain_map_max == (3, 1)


def test_metadata_flags_default_to_single_channel_base_space_off() -> None:
    payload = GainMapMetadata(max_content_boost=EXPECTED_BOOST).to_bytes()
    assert payload[4] == 0


def test_metadata_sets_use_base_colour_space_bit() -> None:
    payload = GainMapMetadata(max_content_boost=EXPECTED_BOOST, use_base_colour_space=True).to_bytes()
    assert payload[4] == USE_BASE_COLOUR_SPACE_BIT


def test_tone_map_payload_prefixes_a_version_byte() -> None:
    """The AVIF 'tmap' carriage wraps the metadata in a ToneMapImage version byte."""
    metadata = GainMapMetadata(max_content_boost=EXPECTED_BOOST)
    payload = metadata.to_tone_map_payload()

    assert len(payload) == EXPECTED_TONE_MAP_SIZE
    assert payload[0] == 0
    assert payload[1:] == metadata.to_bytes()
