"""I/O and decoding helpers for SDR and gain map assets."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Literal

import imagecodecs
import numpy as np

from ultra_hdr_converter.errors import JpegStructureError

JPEG_SOI = b"\xff\xd8"
JPEG_MIN_BYTES = 4
MARKER_PREFIX = 0xFF
START_OF_SCAN_MARKER = 0xDA
END_OF_IMAGE_MARKER = 0xD9
APP1_MARKER = 0xE1
APP2_MARKER = 0xE2
MP_ENTRY_TAG = 0xB002
MP_NUMBER_OF_IMAGES_TAG = 0xB001
MP_ENTRY_SIZE = 16
MP_MIN_IMAGE_COUNT = 2
TIFF_HEADER_MIN_SIZE = 8
MARKER_STUFFED_ZERO = 0x00
MARKER_TEM = 0x01
RST_MARKER_MIN = 0xD0
RST_MARKER_MAX = 0xD7
SEGMENT_LENGTH_BYTES = 2
MIN_SEGMENT_LENGTH = 2
ICC_SIGNATURE = b"ICC_PROFILE\x00"
ICC_HEADER_BYTES = 2
ICC_MIN_SEQUENCE = 1


def read_bytes(path: Path | str) -> bytes:
    """Read file contents as bytes."""
    return Path(path).read_bytes()


def write_bytes(path: Path | str, data: bytes) -> None:
    """Write bytes to disk, creating parent folders if needed."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)


def decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes into a NumPy array."""
    try:
        return np.asarray(imagecodecs.jpeg_decode(jpeg_bytes))
    except Exception as exc:
        raise JpegStructureError(f"JPEG decoding failed: {exc}") from exc


def _skip_marker_prefixes(jpeg_bytes: bytes, offset: int) -> int:
    """Advance offset past repeated 0xFF marker prefix bytes."""
    while offset < len(jpeg_bytes) and jpeg_bytes[offset] == MARKER_PREFIX:
        offset += 1
    return offset


def _is_standalone_marker(marker: int) -> bool:
    """Return True for markers that are not followed by a payload length field."""
    if marker in {MARKER_STUFFED_ZERO, MARKER_TEM}:
        return True
    return RST_MARKER_MIN <= marker <= RST_MARKER_MAX


def _iter_jpeg_segments(jpeg_bytes: bytes) -> list[tuple[int, bytes]]:
    """Iterate marker/payload segments until SOS or EOI."""
    if len(jpeg_bytes) < JPEG_MIN_BYTES or jpeg_bytes[:SEGMENT_LENGTH_BYTES] != JPEG_SOI:
        return []

    offset = SEGMENT_LENGTH_BYTES
    segments: list[tuple[int, bytes]] = []

    while offset + 1 < len(jpeg_bytes):
        if jpeg_bytes[offset] != MARKER_PREFIX:
            offset += 1
            continue

        offset = _skip_marker_prefixes(jpeg_bytes, offset)
        if offset >= len(jpeg_bytes):
            break

        marker = jpeg_bytes[offset]
        offset += 1

        if marker in {START_OF_SCAN_MARKER, END_OF_IMAGE_MARKER}:
            break
        if _is_standalone_marker(marker):
            continue

        if offset + SEGMENT_LENGTH_BYTES > len(jpeg_bytes):
            break

        segment_length = int.from_bytes(jpeg_bytes[offset : offset + SEGMENT_LENGTH_BYTES], "big")
        offset += SEGMENT_LENGTH_BYTES
        if segment_length < MIN_SEGMENT_LENGTH:
            break

        payload_length = segment_length - MIN_SEGMENT_LENGTH
        if offset + payload_length > len(jpeg_bytes):
            break

        payload = jpeg_bytes[offset : offset + payload_length]
        offset += payload_length
        segments.append((marker, payload))

    return segments


def _parse_icc_app2_payload(payload: bytes) -> tuple[int, int, bytes] | None:
    """Parse ICC APP2 payload header and return sequence metadata plus chunk bytes."""
    signature_length = len(ICC_SIGNATURE)
    minimum_payload = signature_length + ICC_HEADER_BYTES
    if len(payload) < minimum_payload or not payload.startswith(ICC_SIGNATURE):
        return None

    sequence_index = int(payload[signature_length])
    chunk_count = int(payload[signature_length + 1])
    if sequence_index < ICC_MIN_SEQUENCE or chunk_count < ICC_MIN_SEQUENCE:
        return None

    chunk = payload[minimum_payload:]
    return sequence_index, chunk_count, chunk


def _extract_icc_from_jpeg_app2(jpeg_bytes: bytes) -> bytes | None:
    """Extract ICC profile from JPEG APP2 markers according to the ICC JPEG convention."""
    chunks: dict[int, bytes] = {}
    expected_count: int | None = None

    for marker, payload in _iter_jpeg_segments(jpeg_bytes):
        if marker != APP2_MARKER:
            continue

        parsed = _parse_icc_app2_payload(payload)
        if parsed is None:
            continue

        sequence_index, count, chunk = parsed

        if expected_count is None:
            expected_count = count
        elif expected_count != count:
            return None

        chunks[sequence_index] = chunk

    if not chunks:
        return None

    if expected_count is None:
        expected_count = len(chunks)

    if any(index not in chunks for index in range(1, expected_count + 1)):
        return None

    return b"".join(chunks[index] for index in range(1, expected_count + 1))


def extract_icc_profile(jpeg_bytes: bytes) -> bytes | None:
    """Extract embedded ICC profile bytes from JPEG payload, if present."""
    jpeg_metadata = getattr(imagecodecs, "jpeg_metadata", None)
    if callable(jpeg_metadata):
        metadata = jpeg_metadata(jpeg_bytes)
        if isinstance(metadata, dict):
            icc_profile = metadata.get("icc_profile")
            if isinstance(icc_profile, (bytes, bytearray, memoryview)):
                return bytes(icc_profile)

    return _extract_icc_from_jpeg_app2(jpeg_bytes)


def load_gain_map(path: Path | str) -> np.ndarray:
    """Load a gain map from .npy or standard image codecs."""
    source = Path(path)
    if source.suffix.lower() == ".npy":
        return np.asarray(np.load(source))
    return np.asarray(imagecodecs.imread(str(source)))


def has_ultrahdr_metadata(jpeg_bytes: bytes) -> bool:
    """Check if the JPEG contains ISO 21496-1 or Adobe UltraHDR gain map metadata.

    Detects both the ISO binary metadata (APP2 with ``urn:iso:std:iso:ts:21496:-1``)
    and the Adobe XMP metadata (APP1 with ``hdrgm:`` namespace attributes).
    """
    for marker, payload in _iter_jpeg_segments(jpeg_bytes):
        if marker == APP2_MARKER:
            if payload.startswith(b"urn:iso:std:iso:ts:21496:-1\x00"):
                return True
        elif marker == APP1_MARKER:
            if payload.startswith(b"http://ns.adobe.com/xap/1.0/\x00"):
                if b"hdrgm:" in payload or b"HDRCapacityMin" in payload or b"GainMapMin" in payload:
                    return True
    return False


def has_mpf_secondary_image(jpeg_bytes: bytes) -> bool:
    """Check whether the JPEG contains an MPF secondary image entry.

    This detects the presence of an MPF APP2 segment with at least one
    secondary image, regardless of Ultra HDR metadata.  Useful for
    distinguishing files that have a gain map embedded as an auxiliary
    image but lack the required XMP/ISO metadata.
    """
    return extract_mpf_gain_map(jpeg_bytes) is not None


def _parse_tiff_byte_order(tiff_header: bytes) -> tuple[str, Literal["little", "big"]] | None:
    """Determine TIFF endianness from the byte-order mark.

    Returns (struct_prefix, byteorder_name) or None for invalid headers.
    """
    if len(tiff_header) < TIFF_HEADER_MIN_SIZE:
        return None
    byte_order_mark = tiff_header[:2]
    if byte_order_mark == b"II":
        return ("<", "little")
    if byte_order_mark == b"MM":
        return (">", "big")
    return None


def _parse_mpf_ifd_entries(
    tiff_header: bytes,
    first_ifd_offset: int,
    endian: str,
) -> tuple[int | None, int | None]:
    """Walk MPF IFD entries and return (mp_entry_data_offset, mp_entry_count)."""
    entry_count = int.from_bytes(
        tiff_header[first_ifd_offset : first_ifd_offset + 2],
        "little" if endian == "<" else "big",
    )

    pos_ifd = first_ifd_offset + 2
    mp_entry_data_offset: int | None = None
    mp_entry_count: int | None = None

    for _ in range(entry_count):
        if pos_ifd + 12 > len(tiff_header):
            break
        tag = struct.unpack(endian + "H", tiff_header[pos_ifd : pos_ifd + 2])[0]
        if tag == MP_ENTRY_TAG:
            mp_entry_data_offset = struct.unpack(
                endian + "I", tiff_header[pos_ifd + 8 : pos_ifd + 12]
            )[0]
        elif tag == MP_NUMBER_OF_IMAGES_TAG:
            mp_entry_count = struct.unpack(
                endian + "I", tiff_header[pos_ifd + 8 : pos_ifd + 12]
            )[0]
        pos_ifd += 12

    return mp_entry_data_offset, mp_entry_count


def _extract_mpf_secondary(
    tiff_header: bytes,
    tiff_header_offset: int,
    jpeg_bytes: bytes,
    endian: str,
    mp_entry_data_offset: int,
) -> bytes | None:
    """Extract and validate the secondary (gain map) image bytes from MPF entries.

    Returns the raw JPEG bytes of the secondary image, or None if the
    entry is invalid or the secondary data is not a valid JPEG.
    """
    entry2_pos = mp_entry_data_offset + MP_ENTRY_SIZE
    if entry2_pos + MP_ENTRY_SIZE > len(tiff_header):
        return None

    size = struct.unpack(endian + "I", tiff_header[entry2_pos + 4 : entry2_pos + 8])[0]
    data_offset = struct.unpack(endian + "I", tiff_header[entry2_pos + 8 : entry2_pos + 12])[0]

    if size == 0:
        return None

    absolute_offset = tiff_header_offset + data_offset
    if absolute_offset + size > len(jpeg_bytes):
        return None

    secondary_bytes = jpeg_bytes[absolute_offset : absolute_offset + size]

    # Validate the extracted secondary image starts with a JPEG SOI marker.
    if len(secondary_bytes) < JPEG_MIN_BYTES or secondary_bytes[:2] != JPEG_SOI:
        return None

    return secondary_bytes


def _parse_mpf_tiff(tiff_header: bytes, tiff_header_offset: int, jpeg_bytes: bytes) -> bytes | None:
    """Parse the TIFF structure within an MPF segment to find the gain map.

    Returns the raw bytes of the secondary (gain map) image, or None if
    the structure is invalid or no secondary image is present.
    """
    parsed = _parse_tiff_byte_order(tiff_header)
    if parsed is None:
        return None
    endian, endian_str = parsed

    first_ifd_offset = int.from_bytes(tiff_header[4:8], endian_str)
    if first_ifd_offset + 2 > len(tiff_header):
        return None

    mp_entry_data_offset, mp_entry_count = _parse_mpf_ifd_entries(
        tiff_header, first_ifd_offset, endian
    )

    # Need at least 2 images (primary + secondary) and a valid entry offset.
    if mp_entry_data_offset is None:
        return None
    if mp_entry_count is not None and mp_entry_count < MP_MIN_IMAGE_COUNT:
        return None

    return _extract_mpf_secondary(
        tiff_header, tiff_header_offset, jpeg_bytes, endian, mp_entry_data_offset
    )


def extract_mpf_gain_map(jpeg_bytes: bytes) -> bytes | None:
    """Extract the secondary image (gain map) from an MPF-encoded JPEG.

    Scans the JPEG header segments for an APP2 MPF marker and parses its
    TIFF/IFD structure to locate the secondary image.  Returns the raw
    JPEG bytes of the secondary image (typically a gain map), or ``None``
    if no MPF segment is found or the structure is invalid.
    """
    mpf_id = b"MPF\x00"

    # Fast path: skip full parsing when no MPF marker is present.
    if mpf_id not in jpeg_bytes:
        return None

    if len(jpeg_bytes) < JPEG_MIN_BYTES or jpeg_bytes[:2] != JPEG_SOI:
        return None

    pos = 2
    while pos + 3 < len(jpeg_bytes):
        if jpeg_bytes[pos] != MARKER_PREFIX:
            break

        marker_pos = _skip_marker_prefixes(jpeg_bytes, pos + 1)

        if marker_pos >= len(jpeg_bytes):
            break

        marker = jpeg_bytes[marker_pos]
        if _is_standalone_marker(marker) or marker in {START_OF_SCAN_MARKER, END_OF_IMAGE_MARKER}:
            pos = marker_pos + 1
            continue

        if marker_pos + 2 >= len(jpeg_bytes):
            break

        seg_length = int.from_bytes(jpeg_bytes[marker_pos + 1 : marker_pos + 3], "big")
        if seg_length < MIN_SEGMENT_LENGTH:
            break
        seg_end = marker_pos + 1 + seg_length

        if seg_end > len(jpeg_bytes):
            break

        if marker == APP2_MARKER and jpeg_bytes[marker_pos + 3 : marker_pos + 3 + len(mpf_id)] == mpf_id:
            tiff_header_offset = marker_pos + 3 + len(mpf_id)
            result = _parse_mpf_tiff(jpeg_bytes[tiff_header_offset:seg_end], tiff_header_offset, jpeg_bytes)
            if result is not None:
                return result

        pos = seg_end

    return None
