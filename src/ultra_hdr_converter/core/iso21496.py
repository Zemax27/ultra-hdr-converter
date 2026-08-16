"""ISO 21496-1 gain map metadata, shared by every container format.

The binary ``GainMapMetadata`` structure defined in clause C.2.2 of ISO 21496-1
is identical regardless of how it is carried:

* JPEG stores it in an APP2 segment prefixed with the ISO namespace string.
* AVIF stores it inside a ``tmap`` derived image item, prefixed with a single
  ``version`` byte (``ToneMapImage`` syntax, ISO/IEC 23008-12:2024 § 6.6.2.4.2).

Keeping the encoder here means both containers describe the same gain map with
the same numbers.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from fractions import Fraction

# Maximum denominator when converting floats to rationals.
FRACTION_DENOM_LIMIT = 100000

# ISO 21496-1 namespace identifier used by the JPEG APP2 carriage.
ISO_NAMESPACE = b"urn:iso:std:iso:ts:21496:-1\x00"

# Default SDR/HDR offsets (1/64) used to avoid division by zero in the
# reconstruction formula.
DEFAULT_OFFSET = 1.0 / 64.0

# Number of channels written for a single-channel (luminance) gain map.
SINGLE_CHANNEL = 1

_IS_MULTICHANNEL_BIT = 0x80
_USE_BASE_COLOUR_SPACE_BIT = 0x40


def float_to_fraction(value: float) -> tuple[int, int]:
    """Convert a float to a rational (numerator, denominator).

    Args:
        value: Value to approximate.

    Returns:
        Numerator and denominator, with a denominator of 1 for zero.
    """
    if value == 0.0:
        return (0, 1)
    frac = Fraction(value).limit_denominator(FRACTION_DENOM_LIMIT)
    return (frac.numerator, frac.denominator)


@dataclass(frozen=True)
class GainMapMetadata:
    """ISO 21496-1 gain map parameters for a single-channel forward gain map.

    Attributes:
        max_content_boost: Maximum HDR content boost, in stops. Written as both
            ``gain_map_max`` and ``alternate_hdr_headroom``.
        min_content_boost: Minimum HDR content boost, in stops.
        gamma: Gain map transfer function exponent.
        base_offset: SDR offset applied during reconstruction.
        alternate_offset: HDR offset applied during reconstruction.
        base_hdr_headroom: Headroom of the base rendition, in stops (0 for SDR).
        use_base_colour_space: Whether the gain map is applied in the base
            image colour space rather than the alternate one.
    """

    max_content_boost: float
    min_content_boost: float = 0.0
    gamma: float = 1.0
    base_offset: float = DEFAULT_OFFSET
    alternate_offset: float = DEFAULT_OFFSET
    base_hdr_headroom: float = 0.0
    use_base_colour_space: bool = False

    def to_bytes(self) -> bytes:
        """Encode the ISO 21496-1 ``GainMapMetadata`` structure.

        Field semantics (SDR base rendition, forward direction):

        * ``gain_map_min`` / ``gain_map_max`` — log2 of the linear content
          boost range.
        * ``gamma`` — gain map transfer function exponent.
        * ``base_offset`` / ``alternate_offset`` — SDR and HDR offsets.
        * ``base_hdr_headroom`` — log2 of the base rendition headroom.
        * ``alternate_hdr_headroom`` — log2 of the alternate (HDR) headroom.

        All values are stored as big-endian rationals (numerator, denominator).

        Returns:
            The packed metadata blob.
        """
        buf = bytearray()

        # GainMapVersion: minimum_version = 0, writer_version = 0.
        buf += struct.pack(">HH", 0, 0)

        # is_multichannel (1 bit), use_base_colour_space (1 bit), reserved (6 bits).
        flags = _USE_BASE_COLOUR_SPACE_BIT if self.use_base_colour_space else 0
        buf += struct.pack(">B", flags)

        buf += struct.pack(">II", *float_to_fraction(self.base_hdr_headroom))
        buf += struct.pack(">II", *float_to_fraction(self.max_content_boost))

        # ---- Per-channel fields (single channel) ----
        buf += struct.pack(">iI", *float_to_fraction(self.min_content_boost))
        buf += struct.pack(">iI", *float_to_fraction(self.max_content_boost))
        buf += struct.pack(">II", *float_to_fraction(self.gamma))
        buf += struct.pack(">iI", *float_to_fraction(self.base_offset))
        buf += struct.pack(">iI", *float_to_fraction(self.alternate_offset))

        return bytes(buf)

    def to_tone_map_payload(self) -> bytes:
        """Encode the ``ToneMapImage`` payload carried by an AVIF ``tmap`` item.

        Defined in ISO/IEC 23008-12:2024 § 6.6.2.4.2: a single ``version`` byte
        followed by the ISO 21496-1 ``GainMapMetadata`` structure.

        Returns:
            The packed ``tmap`` item payload.
        """
        return struct.pack(">B", 0) + self.to_bytes()
