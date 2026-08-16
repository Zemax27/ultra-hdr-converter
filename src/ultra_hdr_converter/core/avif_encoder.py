"""AVIF gain map encoder — ISOBMFF composition.

Composes an AVIF file carrying an ISO 21496-1 gain map by muxing two coded AV1
images into a single HEIF container:

* the SDR base image (the primary item, so gain-map-unaware viewers show it),
* the gain map image (a hidden item),
* a ``tmap`` tone map derived image item holding the ISO 21496-1 metadata and
  referencing the two images through ``iref``/``dimg``.

The structure follows ISO/IEC 23008-12:2024 § 6.6.2.4 and matches what
libavif's own encoder writes, so libavif, Chrome and Android read the result.

``imagecodecs`` only exposes plain single-image AVIF encoding, so the container
is assembled here.  Coded AV1 payloads are copied verbatim, which means an AVIF
input keeps its original quality — the same byte-preserving approach the JPEG
path takes with the original SDR JPEG.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import imagecodecs
import numpy as np

from ultra_hdr_converter.core.avif_io import (
    BOX_HEADER_SIZE,
    DEPTH_8,
    DEPTH_10,
    ITEM_TYPE_AV01,
    ITEM_TYPE_TMAP,
    REF_TYPE_AUXL,
    REF_TYPE_DIMG,
    Av1Config,
    CodedImage,
    build_coded_image,
    find_alpha_item,
    get_primary_item,
    parse_av1c,
    parse_meta,
)
from ultra_hdr_converter.core.iso21496 import GainMapMetadata
from ultra_hdr_converter.errors import AvifStructureError

COLOR_NDIM = 3

# ---- Brands ------------------------------------------------------------------

_BRAND_AVIF = b"avif"
_BRAND_MIF1 = b"mif1"
_BRAND_MIAF = b"miaf"
_BRAND_TMAP = b"tmap"
_BRAND_MA1A = b"MA1A"  # AVIF Advanced Profile (4:4:4)
_BRAND_MA1B = b"MA1B"  # AVIF Baseline Profile (4:2:0)

# ---- Item identifiers --------------------------------------------------------

_ITEM_ID_BASE = 1
_ITEM_ID_TMAP = 2
_ITEM_ID_GAIN_MAP = 3
_ITEM_ID_ALPHA = 4

_INFE_NAME_BASE = b"Color"
_INFE_NAME_GAIN_MAP = b"GMap"
_INFE_NAME_ALPHA = b"Alpha"

_ALPHA_AUX_URN = b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha\x00"

# ---- Field widths and layout -------------------------------------------------

_ILOC_OFFSET_SIZE = 4
_ILOC_LENGTH_SIZE = 4
_ILOC_BASE_OFFSET_SIZE = 0
_INFE_VERSION = 2
_INFE_HIDDEN_FLAG = 0x1
_IPMA_ESSENTIAL_MASK = 0x80
_MAX_DEDUPED_PROPERTIES = 127  # one-byte ipma property indices

# Gain maps are single-channel 8-bit images regardless of the base image depth,
# which is what the ISO 21496-1 ecosystem expects.
GAIN_MAP_BIT_DEPTH = DEPTH_8

# ---- Colour information (CICP, ITU-T H.273) ----------------------------------

_CICP_PRIMARIES_BT709 = 1
_CICP_TRANSFER_SRGB = 13
_CICP_MATRIX_BT601 = 6
_CICP_FULL_RANGE_FLAG = 0x80

# Default sRGB tagging applied to base images encoded from raster input.
SRGB_CICP = (_CICP_PRIMARIES_BT709, _CICP_TRANSFER_SRGB, _CICP_MATRIX_BT601)

# CICP code 2 means "unspecified" in ITU-T H.273; used for gain map images,
# which carry ratios rather than colour.
_CICP_UNSPECIFIED = 2
_UNSPECIFIED_CICP = (_CICP_UNSPECIFIED, _CICP_UNSPECIFIED, _CICP_UNSPECIFIED)

# ``avif_encode`` maps ``level`` to a 0-100 quality scale, matching the JPEG path.
DEFAULT_QUALITY = 95


# ---- Box writers -------------------------------------------------------------


def _box(box_type: bytes, payload: bytes) -> bytes:
    """Serialise a plain ISOBMFF box."""
    return struct.pack(">I", BOX_HEADER_SIZE + len(payload)) + box_type + payload


def _full_box(box_type: bytes, version: int, flags: int, payload: bytes) -> bytes:
    """Serialise an ISOBMFF FullBox with its version/flags header."""
    header = struct.pack(">B", version) + flags.to_bytes(3, "big")
    return _box(box_type, header + payload)


# ---- Property builders -------------------------------------------------------


def build_ispe(width: int, height: int) -> bytes:
    """Build an ``ispe`` (image spatial extents) property box."""
    return _full_box(b"ispe", 0, 0, struct.pack(">II", width, height))


def build_pixi(channel_count: int, depth: int) -> bytes:
    """Build a ``pixi`` (pixel information) property box."""
    return _full_box(b"pixi", 0, 0, struct.pack(">B", channel_count) + bytes([depth] * channel_count))


def build_nclx(primaries: int, transfer: int, matrix: int, is_full_range: bool = True) -> bytes:
    """Build a ``colr`` property box carrying nclx colour information."""
    payload = b"nclx" + struct.pack(">HHH", primaries, transfer, matrix)
    payload += struct.pack(">B", _CICP_FULL_RANGE_FLAG if is_full_range else 0)
    return _box(b"colr", payload)


def build_icc_colr(icc_profile: bytes) -> bytes:
    """Build a ``colr`` property box carrying a full ICC profile."""
    return _box(b"colr", b"prof" + icc_profile)


def build_auxc(aux_type: bytes = _ALPHA_AUX_URN) -> bytes:
    """Build an ``auxC`` (auxiliary type) property box."""
    return _full_box(b"auxC", 0, 0, aux_type)


def miaf_brand(config: Av1Config) -> bytes | None:
    """Return the MIAF profile brand implied by a codec configuration.

    Args:
        config: Sequence-header fields of the base image.

    Returns:
        ``MA1B`` for 8/10-bit 4:2:0, ``MA1A`` for 8/10-bit 4:4:4, otherwise
        ``None`` — monochrome, 12-bit and 4:2:2 images match no MIAF profile.
    """
    if config.is_monochrome or config.depth not in (DEPTH_8, DEPTH_10):
        return None
    if config.subsampling_x and config.subsampling_y:
        return _BRAND_MA1B
    if not config.subsampling_x and not config.subsampling_y:
        return _BRAND_MA1A
    return None


# ---- Item model --------------------------------------------------------------


@dataclass(frozen=True)
class _Item:
    """One item to write into the output container.

    Attributes:
        item_id: Item identifier, unique within the file.
        item_type: Four-character item type, e.g. ``b"av01"``.
        name: ``infe`` item name (a null terminator is appended on write).
        payload: Bytes stored for this item in ``mdat``.
        properties: Ordered (property box, essential flag) pairs.
        is_hidden: Whether the item is marked as a hidden image item.
        dimg_from: Item id of the derived item that consumes this item as an
            input, or 0 when this item is not a derivation input.
        iref_type: Item reference type emitted from this item, if any.
        iref_to: Target item id for ``iref_type``.
    """

    item_id: int
    item_type: bytes
    name: bytes
    payload: bytes
    properties: tuple[tuple[bytes, bool], ...] = ()
    is_hidden: bool = False
    dimg_from: int = 0
    iref_type: bytes | None = None
    iref_to: int = 0


@dataclass
class _PropertyTable:
    """Deduplicated ``ipco`` property store with per-item associations."""

    boxes: list[bytes] = field(default_factory=list)
    _index_by_box: dict[bytes, int] = field(default_factory=dict)

    def associate(self, item: _Item) -> list[tuple[int, bool]]:
        """Register an item's properties and return its ``ipma`` associations."""
        associations: list[tuple[int, bool]] = []
        for box, is_essential in item.properties:
            index = self._index_by_box.get(box)
            if index is None:
                self.boxes.append(box)
                index = len(self.boxes)
                self._index_by_box[box] = index
            associations.append((index, is_essential))
        return associations


# ---- Container writer --------------------------------------------------------


def _build_ftyp(brands: list[bytes]) -> bytes:
    """Build the file type box with the given compatible brands."""
    payload = _BRAND_AVIF + struct.pack(">I", 0) + b"".join(brands)
    return _box(b"ftyp", payload)


def _build_hdlr() -> bytes:
    """Build the ``pict`` handler box required by HEIF."""
    payload = struct.pack(">I", 0) + b"pict" + bytes(12) + b"\x00"
    return _full_box(b"hdlr", 0, 0, payload)


def _build_iinf(items: list[_Item]) -> bytes:
    """Build the item information box."""
    entries = b""
    for item in items:
        flags = _INFE_HIDDEN_FLAG if item.is_hidden else 0
        payload = struct.pack(">HH", item.item_id, 0) + item.item_type + item.name + b"\x00"
        entries += _full_box(b"infe", _INFE_VERSION, flags, payload)
    return _full_box(b"iinf", 0, 0, struct.pack(">H", len(items)) + entries)


def _build_iloc(items: list[_Item], payload_offsets: dict[int, int]) -> bytes:
    """Build the item location box pointing into ``mdat``.

    Args:
        items: Items in write order.
        payload_offsets: Absolute file offset of each item's payload. All
            entries use fixed-width fields, so the box size does not depend on
            the offset values and a two-pass layout is stable.
    """
    packed = struct.pack(
        ">BB",
        (_ILOC_OFFSET_SIZE << 4) | _ILOC_LENGTH_SIZE,
        (_ILOC_BASE_OFFSET_SIZE << 4),
    )
    payload = packed + struct.pack(">H", len(items))
    for item in items:
        payload += struct.pack(">HHH", item.item_id, 0, 1)
        payload += struct.pack(">II", payload_offsets[item.item_id], len(item.payload))
    return _full_box(b"iloc", 0, 0, payload)


def _build_iref(items: list[_Item]) -> bytes:
    """Build the item reference box for ``dimg`` and auxiliary references."""
    references = b""

    for item in items:
        inputs = [other.item_id for other in items if other.dimg_from == item.item_id]
        if inputs:
            payload = struct.pack(">HH", item.item_id, len(inputs))
            payload += b"".join(struct.pack(">H", input_id) for input_id in inputs)
            references += _box(REF_TYPE_DIMG, payload)

    for item in items:
        if item.iref_type is not None and item.iref_to:
            payload = struct.pack(">HHH", item.item_id, 1, item.iref_to)
            references += _box(item.iref_type, payload)

    if not references:
        return b""
    return _full_box(b"iref", 0, 0, references)


def _build_iprp(items: list[_Item]) -> bytes:
    """Build the item properties box (``ipco`` plus ``ipma``)."""
    table = _PropertyTable()
    associations = {item.item_id: table.associate(item) for item in items}

    if len(table.boxes) > _MAX_DEDUPED_PROPERTIES:
        raise AvifStructureError(
            f"Too many distinct item properties ({len(table.boxes)}) for one-byte ipma indices."
        )

    ipco = _box(b"ipco", b"".join(table.boxes))

    entries = b""
    entry_count = 0
    for item in items:
        item_associations = associations[item.item_id]
        if not item_associations:
            continue
        entry_count += 1
        entries += struct.pack(">HB", item.item_id, len(item_associations))
        for index, is_essential in item_associations:
            entries += struct.pack(">B", index | (_IPMA_ESSENTIAL_MASK if is_essential else 0))

    ipma = _full_box(b"ipma", 0, 0, struct.pack(">I", entry_count) + entries)
    return _box(b"iprp", ipco + ipma)


def _build_grpl_altr(group_id: int, entity_ids: list[int]) -> bytes:
    """Build a ``grpl``/``altr`` entity group listing alternative renditions."""
    payload = struct.pack(">II", group_id, len(entity_ids))
    payload += b"".join(struct.pack(">I", entity_id) for entity_id in entity_ids)
    return _box(b"grpl", _full_box(b"altr", 0, 0, payload))


def _build_meta(items: list[_Item], primary_item_id: int, altr_ids: list[int], payload_offsets: dict[int, int]) -> bytes:
    """Assemble the ``meta`` box from its child boxes."""
    children = _build_hdlr()
    children += _full_box(b"pitm", 0, 0, struct.pack(">H", primary_item_id))
    children += _build_iloc(items, payload_offsets)
    children += _build_iinf(items)
    children += _build_iref(items)
    children += _build_iprp(items)
    if altr_ids:
        children += _build_grpl_altr(max(item.item_id for item in items) + 1, altr_ids)
    return _full_box(b"meta", 0, 0, children)


def _write_container(items: list[_Item], primary_item_id: int, brands: list[bytes], altr_ids: list[int]) -> bytes:
    """Serialise a complete AVIF file.

    ``iloc`` stores absolute file offsets, which are only known once the
    ``meta`` box size is fixed. Because every ``iloc`` field has a fixed width,
    the box is first built with placeholder offsets purely to measure it, then
    rebuilt with the real offsets — the size is identical both times.
    """
    ftyp = _build_ftyp(brands)
    placeholder_offsets = dict.fromkeys((item.item_id for item in items), 0)
    meta_size = len(_build_meta(items, primary_item_id, altr_ids, placeholder_offsets))

    mdat_payload_start = len(ftyp) + meta_size + BOX_HEADER_SIZE
    payload_offsets: dict[int, int] = {}
    cursor = mdat_payload_start
    for item in items:
        payload_offsets[item.item_id] = cursor
        cursor += len(item.payload)

    meta = _build_meta(items, primary_item_id, altr_ids, payload_offsets)
    if len(meta) != meta_size:
        raise AvifStructureError("Internal error: 'meta' box size changed between layout passes.")

    mdat = _box(b"mdat", b"".join(item.payload for item in items))
    return ftyp + meta + mdat


# ---- Coded image helpers -----------------------------------------------------


def _image_properties(image: CodedImage, *, include_icc: bool) -> tuple[tuple[bytes, bool], ...]:
    """Build the descriptive property list for a coded image item.

    ISO/IEC 23008-12:2024 § 6.5.1 asks writers to place descriptive properties
    before any others, so ``ispe`` and ``pixi`` come first and the codec
    configuration — the only essential property — follows.
    """
    config = parse_av1c(image.av1c)
    properties: list[tuple[bytes, bool]] = [
        (build_ispe(image.width, image.height), False),
        (image.pixi or build_pixi(config.channel_count, config.depth), False),
    ]
    if image.av1c is not None:
        properties.append((image.av1c, True))
    if include_icc and image.icc_profile:
        properties.append((build_icc_colr(image.icc_profile), False))
    if image.nclx is not None:
        properties.append((image.nclx, False))
    return tuple(properties)


def encode_coded_image(
    image: np.ndarray,
    quality: int = DEFAULT_QUALITY,
    icc_profile: bytes | None = None,
    cicp: tuple[int, int, int] = SRGB_CICP,
    bit_depth: int = DEPTH_8,
) -> CodedImage:
    """Encode a raster image to AVIF and lift the coded item back out.

    ``imagecodecs.avif_encode`` can only produce a standalone single-image
    file, so the result is parsed straight back into a :class:`CodedImage` that
    the muxer can place into a multi-item container.

    Args:
        image: Pixel array of shape (H, W) or (H, W, C). ``uint8`` for 8-bit
            samples, ``uint16`` for deeper ones.
        quality: Encoder quality level (0-100).
        icc_profile: Optional ICC profile to attach to the coded image.
        cicp: (primaries, transfer, matrix) CICP codes used to tag the image.
        bit_depth: Bits per sample to encode, e.g. 10 for a 10-bit image.

    Returns:
        The coded image with its descriptive properties.

    Raises:
        AvifStructureError: If encoding or re-parsing fails.
    """
    primaries, transfer, matrix = cicp
    try:
        encoded = bytes(
            imagecodecs.avif_encode(
                np.ascontiguousarray(image),
                level=quality,
                bitspersample=bit_depth,
                primaries=primaries,
                transfer=transfer,
                matrix=matrix,
            )
        )
    except Exception as exc:
        raise AvifStructureError(f"AVIF encoding failed: {exc}") from exc

    meta = parse_meta(encoded)
    coded = build_coded_image(encoded, meta, get_primary_item(meta))
    if icc_profile:
        return CodedImage(
            payload=coded.payload,
            width=coded.width,
            height=coded.height,
            av1c=coded.av1c,
            pixi=coded.pixi,
            nclx=coded.nclx,
            icc_profile=icc_profile,
            auxc=coded.auxc,
        )
    return coded


def build_single_item_avif(image: CodedImage) -> bytes:
    """Wrap one coded image into a standalone single-item AVIF file.

    Used to hand an extracted item back to ``imagecodecs`` for decoding.

    Args:
        image: The coded image to wrap.

    Returns:
        Complete AVIF file bytes.
    """
    item = _Item(
        item_id=_ITEM_ID_BASE,
        item_type=ITEM_TYPE_AV01,
        name=_INFE_NAME_BASE,
        payload=image.payload,
        properties=_image_properties(image, include_icc=True),
    )
    brands = [_BRAND_AVIF, _BRAND_MIF1, _BRAND_MIAF]
    brand = miaf_brand(parse_av1c(image.av1c))
    if brand is not None:
        brands.append(brand)
    return _write_container([item], primary_item_id=_ITEM_ID_BASE, brands=brands, altr_ids=[])


def extract_base_image(avif_bytes: bytes) -> tuple[CodedImage, CodedImage | None]:
    """Lift the base colour image (and its alpha plane) out of an AVIF file.

    Reusing the coded payload keeps the original encoding quality instead of
    decoding and re-encoding the image.

    Args:
        avif_bytes: Complete AVIF file bytes.

    Returns:
        The base coded image and its alpha auxiliary image, if present.

    Raises:
        AvifStructureError: If the file is not a plain single-tile AVIF.
    """
    meta = parse_meta(avif_bytes)
    primary = get_primary_item(meta)
    base = build_coded_image(avif_bytes, meta, primary)

    alpha_item = find_alpha_item(meta)
    alpha = build_coded_image(avif_bytes, meta, alpha_item) if alpha_item is not None else None
    return base, alpha


# ---- Public API --------------------------------------------------------------


def encode_ultrahdr_avif(
    base_image: CodedImage,
    gain_map: np.ndarray,
    quality: int = DEFAULT_QUALITY,
    max_content_boost: float = 3.0,
    alpha_image: CodedImage | None = None,
) -> bytes:
    """Compose an AVIF file carrying an ISO 21496-1 gain map.

    The base image is written as the primary item so viewers without gain map
    support display the SDR rendition unchanged. A ``tmap`` derived item holds
    the gain map metadata and references the base and gain map images, and both
    are placed in an ``altr`` entity group as recommended by the AVIF
    specification.

    Args:
        base_image: Coded SDR base image, reused without re-encoding.
        gain_map: Single-channel uint8 gain map array of shape (H, W) or
            (H, W, C); only the first channel is used.
        quality: AVIF quality level for gain map compression (0-100).
        max_content_boost: Maximum HDR content boost in stops, written into the
            gain map metadata.
        alpha_image: Optional coded alpha plane carried through from the input.

    Returns:
        Composed AVIF file bytes.

    Raises:
        AvifStructureError: If encoding the gain map or muxing fails.
    """
    gain = np.asarray(gain_map, dtype=np.uint8)
    if gain.ndim == COLOR_NDIM:
        gain = gain[..., 0]

    # The gain map is a single-channel image with no meaningful colour space of
    # its own, so it is tagged as unspecified rather than sRGB.  It stays 8-bit
    # even when the base image is deeper, as ISO 21496-1 readers expect.
    coded_gain_map = encode_coded_image(
        gain,
        quality=quality,
        cicp=_UNSPECIFIED_CICP,
        bit_depth=GAIN_MAP_BIT_DEPTH,
    )

    metadata = GainMapMetadata(max_content_boost=max_content_boost)

    items = [
        _Item(
            item_id=_ITEM_ID_BASE,
            item_type=ITEM_TYPE_AV01,
            name=_INFE_NAME_BASE,
            payload=base_image.payload,
            properties=_image_properties(base_image, include_icc=True),
            dimg_from=_ITEM_ID_TMAP,
        ),
        _Item(
            item_id=_ITEM_ID_TMAP,
            item_type=ITEM_TYPE_TMAP,
            name=_INFE_NAME_GAIN_MAP,
            payload=metadata.to_tone_map_payload(),
            properties=_tmap_properties(base_image),
        ),
        _Item(
            item_id=_ITEM_ID_GAIN_MAP,
            item_type=ITEM_TYPE_AV01,
            name=_INFE_NAME_GAIN_MAP,
            payload=coded_gain_map.payload,
            properties=_image_properties(coded_gain_map, include_icc=False),
            is_hidden=True,
            dimg_from=_ITEM_ID_TMAP,
        ),
    ]

    if alpha_image is not None:
        items.append(
            _Item(
                item_id=_ITEM_ID_ALPHA,
                item_type=ITEM_TYPE_AV01,
                name=_INFE_NAME_ALPHA,
                payload=alpha_image.payload,
                properties=(*_image_properties(alpha_image, include_icc=False), (build_auxc(), False)),
                is_hidden=True,
                iref_type=REF_TYPE_AUXL,
                iref_to=_ITEM_ID_BASE,
            )
        )

    brands = [_BRAND_AVIF, _BRAND_MIF1, _BRAND_MIAF]
    brand = miaf_brand(parse_av1c(base_image.av1c))
    if brand is not None:
        brands.append(brand)
    brands.append(_BRAND_TMAP)

    return _write_container(
        items,
        primary_item_id=_ITEM_ID_BASE,
        brands=brands,
        altr_ids=[_ITEM_ID_TMAP, _ITEM_ID_BASE],
    )


def _tmap_properties(base_image: CodedImage) -> tuple[tuple[bytes, bool], ...]:
    """Build the property list for the ``tmap`` derived image item.

    A derived image item needs an ``ispe`` giving the dimensions of the
    reconstructed image, plus colour information describing the alternate (HDR)
    rendition. The base image's colour information is mirrored: the gain map is
    computed in the base colour space, so the alternate rendition shares it and
    differs only in headroom. ``pixi`` is optional for ``tmap`` items and is
    omitted.
    """
    properties: list[tuple[bytes, bool]] = [(build_ispe(base_image.width, base_image.height), False)]
    properties.append((base_image.nclx or build_nclx(*SRGB_CICP), False))
    return tuple(properties)
