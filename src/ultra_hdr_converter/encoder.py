"""Ultra HDR encoder — API-4 composition.

Composes an Ultra HDR JPEG by combining the original compressed SDR JPEG
with a JPEG-encoded gain map and ISO 21496-1 metadata.  This preserves
the original SDR encoding quality and embedded ICC profile.
"""

from __future__ import annotations

import struct
from fractions import Fraction

import imagecodecs
import numpy as np

COLOR_NDIM = 3

# ---- JPEG markers ------------------------------------------------------------

_SOI = b"\xff\xd8"
_MARKER_PREFIX = 0xFF
_APP2 = 0xE2

# ---- XMP gain map metadata (Adobe hdrgm namespace) ---------------------------

_XMP_NAMESPACE = b"http://ns.adobe.com/xap/1.0/\x00"

_XMP_TEMPLATE = (
    '<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>'
    '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
    '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    '<rdf:Description rdf:about=""'
    " xmlns:hdrgm=\"http://ns.adobe.com/hdr-gain-map/1.0/\""
    ' hdrgm:Version="1.0"'
    ' hdrgm:BaseRenditionIsHDR="False"'
    " hdrgm:GainMapMin=\"0.0\""
    ' hdrgm:GainMapMax="{max_content_boost}"'
    ' hdrgm:Gamma="1.0"'
    ' hdrgm:OffsetSDR="0.015625"'
    ' hdrgm:OffsetHDR="0.015625"'
    ' hdrgm:HDRCapacityMin="0.0"'
    ' hdrgm:HDRCapacityMax="{hdr_capacity_max}"/>'
    "</rdf:RDF>"
    "</x:xmpmeta>"
    '<?xpacket end="w"?>'
)

# ---- ISO 21496-1 binary gain map metadata ------------------------------------

_ISO_NAMESPACE = b"urn:iso:std:iso:ts:21496:-1\x00"

# Maximum denominator when converting floats to rationals.
_FRACTION_DENOM_LIMIT = 100000

# ---- MPF (Multi-Picture Format, CIPA DC-007) ---------------------------------

_MPF_ID = b"MPF\x00"
_MPF_TIFF_HEADER_SIZE = 8  # byte-order + magic + first-IFD offset
_MPF_IFD_ENTRY_COUNT = 3
_MPF_IFD_SIZE = 2 + _MPF_IFD_ENTRY_COUNT * 12 + 4  # count + entries + next-IFD
_MPF_ENTRY_SIZE = 16  # per image
_MPF_IMAGE_COUNT = 2  # primary + gain map
_MPF_ENTRIES_SIZE = _MPF_IMAGE_COUNT * _MPF_ENTRY_SIZE

# Individual Image Attribute flags (CIPA DC-007, big-endian uint32).
# Bit 25: representative image.
_ATTR_PRIMARY = 0x02000000
_ATTR_SECONDARY = 0x00000000


# ---- Helpers -----------------------------------------------------------------


def _float_to_unsigned_fraction(value: float) -> tuple[int, int]:
    """Convert a non-negative float to an unsigned rational (numerator, denominator)."""
    if value == 0.0:
        return (0, 1)
    frac = Fraction(value).limit_denominator(_FRACTION_DENOM_LIMIT)
    return (frac.numerator, frac.denominator)


def _float_to_signed_fraction(value: float) -> tuple[int, int]:
    """Convert a float to a signed rational (numerator, denominator)."""
    if value == 0.0:
        return (0, 1)
    frac = Fraction(value).limit_denominator(_FRACTION_DENOM_LIMIT)
    return (frac.numerator, frac.denominator)


def _encode_gain_map_jpeg(gain_map: np.ndarray, quality: int) -> bytes:
    """JPEG-encode the gain map as single-channel grayscale."""
    gm = np.asarray(gain_map, dtype=np.uint8)
    if gm.ndim == COLOR_NDIM:
        gm = gm[..., 0]
    return bytes(imagecodecs.jpeg_encode(gm, level=quality))


# ---- XMP segment builder -----------------------------------------------------


def _build_xmp_segment(max_content_boost: float, hdr_capacity_max: float) -> bytes:
    """Build an APP1 XMP segment carrying gain map metadata."""
    xmp_xml = _XMP_TEMPLATE.format(
        max_content_boost=max_content_boost,
        hdr_capacity_max=hdr_capacity_max,
    ).encode("utf-8")

    payload = _XMP_NAMESPACE + xmp_xml
    length = 2 + len(payload)  # length field includes its own 2 bytes
    return b"\xff\xe1" + struct.pack(">H", length) + payload


# ---- ISO 21496-1 segment builders -------------------------------------------


def _build_iso_version_segment() -> bytes:
    """Build an APP2 ISO 21496-1 version-only segment for the primary JPEG.

    Contains only the namespace identifier and version fields (4 zero bytes)
    as required by libultrahdr for the primary image.
    """
    # min_version = 0, writer_version = 0
    payload = _ISO_NAMESPACE + struct.pack(">HH", 0, 0)
    length = 2 + len(payload)
    return b"\xff\xe2" + struct.pack(">H", length) + payload


def _build_iso_metadata_bytes(max_content_boost: float) -> bytes:
    """Encode ISO 21496-1 gain map metadata as a binary blob.

    Field semantics (when base rendition is SDR, forward direction):
      - gainMapMin/Max:       log2 of the linear content boost range.
      - gamma:                gain map transfer function exponent.
      - baseOffset:           SDR offset (avoids division by zero).
      - alternateOffset:      HDR offset.
      - baseHdrHeadroom:      log2(hdr_capacity_min), typically 0.
      - alternateHdrHeadroom: log2(hdr_capacity_max), equals max_content_boost.

    All values stored as big-endian rational numbers (numerator / denominator).
    """
    buf = bytearray()

    # Version header: min_version=0, writer_version=0.
    buf += struct.pack(">HH", 0, 0)

    # Flags: single channel, no base color space, forward direction,
    #         no common denominator.
    buf += struct.pack(">B", 0)

    # baseHdrHeadroom = log2(hdrCapacityMin) = 0.0
    n, d = _float_to_unsigned_fraction(0.0)
    buf += struct.pack(">II", n, d)

    # alternateHdrHeadroom = log2(hdrCapacityMax) = max_content_boost
    n, d = _float_to_unsigned_fraction(max_content_boost)
    buf += struct.pack(">II", n, d)

    # ---- Per-channel fields (single channel) ----

    # gainMapMin = log2(minContentBoost) = 0.0
    n, d = _float_to_signed_fraction(0.0)
    buf += struct.pack(">iI", n, d)

    # gainMapMax = log2(maxContentBoost) = max_content_boost
    n, d = _float_to_signed_fraction(max_content_boost)
    buf += struct.pack(">iI", n, d)

    # gamma = 1.0
    n, d = _float_to_unsigned_fraction(1.0)
    buf += struct.pack(">II", n, d)

    # baseOffset = offsetSdr = 1/64
    n, d = _float_to_signed_fraction(1.0 / 64.0)
    buf += struct.pack(">iI", n, d)

    # alternateOffset = offsetHdr = 1/64
    n, d = _float_to_signed_fraction(1.0 / 64.0)
    buf += struct.pack(">iI", n, d)

    return bytes(buf)


def _build_iso_metadata_segment(max_content_boost: float) -> bytes:
    """Build an APP2 ISO 21496-1 segment with full gain map metadata.

    This segment is injected into the gain map JPEG.
    """
    metadata = _build_iso_metadata_bytes(max_content_boost)
    payload = _ISO_NAMESPACE + metadata
    length = 2 + len(payload)
    return b"\xff\xe2" + struct.pack(">H", length) + payload


def _inject_after_soi(jpeg: bytes, segment: bytes) -> bytes:
    """Inject a marker segment right after the JPEG SOI marker."""
    if jpeg[:2] != _SOI:
        raise ValueError("Not a valid JPEG (missing SOI marker).")
    return jpeg[:2] + segment + jpeg[2:]


# ---- MPF segment builder -----------------------------------------------------


def _build_mpf_segment(
    primary_size: int,
    gainmap_size: int,
    mp_header_file_offset: int,
) -> bytes:
    """Build an APP2 MPF segment describing primary + gain map images.

    Args:
        primary_size: Total byte length of the composed primary JPEG.
        gainmap_size: Byte length of the gain map JPEG.
        mp_header_file_offset: File offset of the MP Header byte-order mark
            ("II") inside the primary JPEG. All MPF data offsets are
            relative to this position.
    """
    # Gain map starts right after the primary JPEG ends.
    gainmap_data_offset = primary_size - mp_header_file_offset

    # MP Entry data: 16 bytes per image.
    mp_entries = struct.pack(
        "<IIIHH IIIHH",
        _ATTR_PRIMARY, primary_size, 0, 0, 0,  # primary (offset 0 = self)
        _ATTR_SECONDARY, gainmap_size, gainmap_data_offset, 0, 0,  # gain map
    )

    # Offset from tiff header to the MP Entry data block.
    mp_entry_data_offset = _MPF_TIFF_HEADER_SIZE + _MPF_IFD_SIZE

    # IFD: three tags (MPFVersion, NumberOfImages, MPEntry).
    ifd = struct.pack("<H", _MPF_IFD_ENTRY_COUNT)
    ifd += struct.pack("<HHI4s", 0xB000, 7, 4, b"0100")  # MPFVersion
    ifd += struct.pack("<HHII", 0xB001, 4, 1, _MPF_IMAGE_COUNT)  # NumberOfImages
    ifd += struct.pack("<HHII", 0xB002, 7, _MPF_ENTRIES_SIZE, mp_entry_data_offset)  # MPEntry
    ifd += struct.pack("<I", 0)  # next IFD = none

    # TIFF header: little-endian, magic 0x002A, first IFD at offset 8.
    tiff_header = b"II" + struct.pack("<HI", 0x002A, _MPF_TIFF_HEADER_SIZE)

    mpf_payload = _MPF_ID + tiff_header + ifd + mp_entries
    length = 2 + len(mpf_payload)
    return b"\xff\xe2" + struct.pack(">H", length) + mpf_payload


# ---- JPEG surgery helpers -----------------------------------------------------


def _find_injection_point(jpeg: bytes) -> int:
    """Return the byte offset just after SOI and any leading APP0/APP1 markers.

    The XMP and MPF segments are inserted at this position so they sit after
    JFIF (APP0) and EXIF (APP1) but before quantisation / frame markers.
    """
    if jpeg[:2] != _SOI:
        raise ValueError("Not a valid JPEG (missing SOI marker).")

    pos = 2  # skip SOI
    while pos + 3 < len(jpeg):
        if jpeg[pos] != _MARKER_PREFIX:
            break
        marker = jpeg[pos + 1]
        # Walk past APP0 and APP1 only; stop before any other marker.
        if marker not in (0xE0, 0xE1):
            break
        seg_length = struct.unpack(">H", jpeg[pos + 2 : pos + 4])[0]
        pos += 2 + seg_length
    return pos


def _strip_mpf_segments(jpeg: bytes) -> bytes:
    """Remove any existing MPF APP2 segments from a JPEG byte stream."""
    if jpeg[:2] != _SOI:
        raise ValueError("Not a valid JPEG (missing SOI marker).")

    parts: list[bytes] = [_SOI]
    pos = 2

    while pos + 3 < len(jpeg):
        if jpeg[pos] != _MARKER_PREFIX:
            # Rest is scan data; keep everything from here.
            parts.append(jpeg[pos:])
            break

        marker = jpeg[pos + 1]

        # Non-parameterised markers (RST, SOI, EOI, TEM) have no length.
        if marker in (0x00, 0x01, 0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xFF):
            parts.append(jpeg[pos : pos + 2])
            pos += 2
            continue

        seg_length = struct.unpack(">H", jpeg[pos + 2 : pos + 4])[0]
        seg_end = pos + 2 + seg_length

        # Drop APP2 segments whose payload starts with "MPF\0".
        if marker == _APP2 and jpeg[pos + 4 : pos + 8] == _MPF_ID:
            pos = seg_end
            continue

        parts.append(jpeg[pos:seg_end])
        pos = seg_end

    return b"".join(parts)


# ---- Public API ---------------------------------------------------------------


def encode_ultrahdr(
    sdr_jpeg: bytes,
    gain_map: np.ndarray,
    jpeg_quality: int = 95,
    max_content_boost: float = 6.0,
) -> bytes:
    """
    Compose an Ultra HDR JPEG from an SDR JPEG and a gain map (API-4).

    The original SDR JPEG is preserved byte-for-byte (including its ICC
    profile and encoding quality). The gain map is JPEG-encoded and
    appended as a secondary MPF image with both XMP and ISO 21496-1
    binary gain map metadata.

    Args:
        sdr_jpeg: Original compressed SDR JPEG bytes.
        gain_map: Single-channel uint8 gain map array.
        jpeg_quality: JPEG quality level for gain map compression.
        max_content_boost: Maximum HDR content boost in stops, written
            into the gain map metadata.

    Returns:
        Composed Ultra HDR JPEG bytes.
    """
    # JPEG-encode the gain map, then inject ISO 21496-1 full metadata.
    raw_gm_jpeg = _encode_gain_map_jpeg(gain_map, quality=jpeg_quality)
    iso_full_segment = _build_iso_metadata_segment(max_content_boost)
    gain_map_jpeg = _inject_after_soi(raw_gm_jpeg, iso_full_segment)

    # Remove any existing MPF from the source JPEG to avoid conflicts.
    clean_jpeg = _strip_mpf_segments(sdr_jpeg)

    # Build the XMP segment with gain map parameters.
    xmp_segment = _build_xmp_segment(
        max_content_boost=max_content_boost,
        hdr_capacity_max=max_content_boost,
    )

    # Build the ISO 21496-1 version-only segment for the primary image.
    iso_version_segment = _build_iso_version_segment()

    # Determine where to splice in the new APP segments.
    injection_point = _find_injection_point(clean_jpeg)

    # The MPF segment has a fixed structure size.  Compute it so we can
    # determine the total primary JPEG length before writing the MPF.
    mpf_payload_size = len(_MPF_ID) + _MPF_TIFF_HEADER_SIZE + _MPF_IFD_SIZE + _MPF_ENTRIES_SIZE
    mpf_segment_size = 2 + 2 + mpf_payload_size  # marker + length field + payload

    injected_size = len(xmp_segment) + len(iso_version_segment) + mpf_segment_size
    primary_size = len(clean_jpeg) + injected_size

    # File offset of the MP Header byte-order mark ("II") inside the
    # composed primary JPEG.  MPF data offsets are relative to this.
    mp_header_file_offset = (
        injection_point + len(xmp_segment) + len(iso_version_segment) + 2 + 2 + len(_MPF_ID)
    )

    mpf_segment = _build_mpf_segment(primary_size, len(gain_map_jpeg), mp_header_file_offset)

    primary = (
        clean_jpeg[:injection_point]
        + xmp_segment
        + iso_version_segment
        + mpf_segment
        + clean_jpeg[injection_point:]
    )

    return primary + gain_map_jpeg
