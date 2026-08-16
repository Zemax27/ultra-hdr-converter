"""AVIF (ISOBMFF/HEIF) parsing helpers.

``imagecodecs`` decodes AVIF pixels but exposes none of the container
structure: no ICC profile, no auxiliary items and no gain map support.  This
module implements the small subset of ISO/IEC 14496-12 and ISO/IEC 23008-12
box parsing that the Ultra HDR pipeline needs:

* locating the primary coded image item and its AV1 payload,
* reading the ICC profile and colour information from item properties,
* detecting a ``tmap`` (tone map derived image) item, which marks a file as
  already carrying an ISO 21496-1 gain map,
* extracting the gain map image referenced by that ``tmap`` item.

Only still images with plain (non-tiled) coded items are handled; grids and
image sequences raise :class:`AvifStructureError` rather than being silently
mis-parsed.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import imagecodecs
import numpy as np

from ultra_hdr_converter.errors import AvifStructureError

# ---- Box parsing constants ---------------------------------------------------

BOX_HEADER_SIZE = 8
LARGE_BOX_HEADER_SIZE = 16
FULL_BOX_EXTRA_SIZE = 4
SIZE_FIELD_LARGE = 1
SIZE_FIELD_TO_END = 0
UUID_SIZE = 16

# ---- Box types ---------------------------------------------------------------

BOX_FTYP = b"ftyp"
BOX_META = b"meta"
BOX_MDAT = b"mdat"
BOX_IDAT = b"idat"
BOX_PITM = b"pitm"
BOX_ILOC = b"iloc"
BOX_IINF = b"iinf"
BOX_INFE = b"infe"
BOX_IREF = b"iref"
BOX_IPRP = b"iprp"
BOX_IPCO = b"ipco"
BOX_IPMA = b"ipma"
BOX_COLR = b"colr"
BOX_ISPE = b"ispe"
BOX_PIXI = b"pixi"
BOX_AV1C = b"av1C"
BOX_AUXC = b"auxC"

ITEM_TYPE_AV01 = b"av01"
ITEM_TYPE_TMAP = b"tmap"
ITEM_TYPE_GRID = b"grid"

REF_TYPE_DIMG = b"dimg"
REF_TYPE_AUXL = b"auxl"

COLOUR_TYPE_ICC = frozenset({b"prof", b"rICC"})
COLOUR_TYPE_NCLX = b"nclx"

# ---- Field sizes and layout --------------------------------------------------

_INFE_VERSION_WITH_TYPE = 2
_INFE_LARGE_ID_VERSION = 3
_ILOC_VERSION_WITH_CONSTRUCTION = 1
_ILOC_VERSION_LARGE_ID = 2
_IPMA_LARGE_ID_VERSION = 1
_IPMA_LARGE_INDEX_FLAG = 0x1
_IPMA_ESSENTIAL_MASK = 0x80
_IPMA_SMALL_INDEX_MASK = 0x7F
_IPMA_LARGE_INDEX_MASK = 0x7FFF
_INFE_HIDDEN_FLAG = 0x1
_CONSTRUCTION_METHOD_FILE = 0
_CONSTRUCTION_METHOD_IDAT = 1
_ISPE_PAYLOAD_SIZE = 8
_TMAP_DIMG_ENTRY_COUNT = 2
_COLOUR_TYPE_SIZE = 4

# Boxes whose children are parsed recursively, mapped to the number of extra
# header bytes to skip before the first child (FullBox version+flags).
_CONTAINER_BOXES: dict[bytes, int] = {
    BOX_META: FULL_BOX_EXTRA_SIZE,
    BOX_IPRP: 0,
}


@dataclass(frozen=True)
class BoxHeader:
    """Location and identity of one ISOBMFF box."""

    box_type: bytes
    offset: int
    header_size: int
    size: int

    @property
    def payload_start(self) -> int:
        """Absolute offset of the first payload byte."""
        return self.offset + self.header_size

    @property
    def end(self) -> int:
        """Absolute offset one past the last byte of the box."""
        return self.offset + self.size


@dataclass(frozen=True)
class AvifItem:
    """One item declared in the ``meta`` box."""

    item_id: int
    item_type: bytes
    name: str
    is_hidden: bool
    extents: tuple[tuple[int, int], ...]
    construction_method: int
    # (one-based ``ipco`` index, essential flag) pairs, in association order.
    properties: tuple[tuple[int, bool], ...] = ()


@dataclass(frozen=True)
class AvifMeta:
    """Parsed contents of the ``meta`` box of an AVIF file."""

    primary_item_id: int
    items: dict[int, AvifItem] = field(default_factory=dict)
    # One-based indexed; ``property_boxes[0]`` corresponds to ``ipco`` index 1.
    property_boxes: tuple[bytes, ...] = ()
    # reference type -> from_item_id -> ordered to_item_ids
    references: dict[bytes, dict[int, tuple[int, ...]]] = field(default_factory=dict)
    idat: bytes = b""

    def property_bytes(self, item: AvifItem, box_type: bytes) -> bytes | None:
        """Return the first property box of *box_type* associated with *item*.

        Args:
            item: Item whose associated properties are searched.
            box_type: Four-character property box type, e.g. ``b"av1C"``.

        Returns:
            The complete property box bytes, or ``None`` when not associated.
        """
        for index, _essential in item.properties:
            box = self._property_at(index)
            if box is not None and box[4:8] == box_type:
                return box
        return None

    def _property_at(self, one_based_index: int) -> bytes | None:
        """Return the ``ipco`` child box at a one-based index, if it exists."""
        if 1 <= one_based_index <= len(self.property_boxes):
            return self.property_boxes[one_based_index - 1]
        return None


@dataclass(frozen=True)
class CodedImage:
    """A coded AV1 image item lifted out of an AVIF container.

    Carrying the compressed payload plus its descriptive properties lets the
    encoder re-package an existing image without re-encoding it, mirroring the
    byte-preserving behaviour of the JPEG path.

    Attributes:
        payload: Raw AV1 bitstream bytes of the item.
        width: Image width in pixels, from the ``ispe`` property.
        height: Image height in pixels, from the ``ispe`` property.
        av1c: Complete ``av1C`` property box, or ``None`` when absent.
        pixi: Complete ``pixi`` property box, or ``None`` when absent.
        nclx: Complete ``colr``/``nclx`` property box, or ``None`` when absent.
        icc_profile: Raw ICC profile bytes from a ``colr``/``prof`` property.
        auxc: Complete ``auxC`` property box for auxiliary items.
    """

    payload: bytes
    width: int
    height: int
    av1c: bytes | None = None
    pixi: bytes | None = None
    nclx: bytes | None = None
    icc_profile: bytes | None = None
    auxc: bytes | None = None


# ---- Low-level box iteration -------------------------------------------------


def _read_box_header(data: bytes, offset: int, limit: int) -> BoxHeader | None:
    """Read one box header, or return None when the range is exhausted."""
    if offset + BOX_HEADER_SIZE > limit:
        return None

    size = int.from_bytes(data[offset : offset + 4], "big")
    box_type = data[offset + 4 : offset + BOX_HEADER_SIZE]
    header_size = BOX_HEADER_SIZE

    if size == SIZE_FIELD_LARGE:
        if offset + LARGE_BOX_HEADER_SIZE > limit:
            return None
        size = int.from_bytes(data[offset + BOX_HEADER_SIZE : offset + LARGE_BOX_HEADER_SIZE], "big")
        header_size = LARGE_BOX_HEADER_SIZE
    elif size == SIZE_FIELD_TO_END:
        size = limit - offset

    if size < header_size or offset + size > limit:
        return None

    return BoxHeader(box_type=box_type, offset=offset, header_size=header_size, size=size)


def iter_boxes(data: bytes, start: int, limit: int) -> list[BoxHeader]:
    """List the boxes contained in ``data[start:limit]``.

    Args:
        data: Complete file bytes.
        start: Offset of the first box header.
        limit: Offset one past the last byte to consider.

    Returns:
        Box headers in file order. Parsing stops at the first malformed header.
    """
    boxes: list[BoxHeader] = []
    offset = start
    while True:
        header = _read_box_header(data, offset, limit)
        if header is None:
            break
        boxes.append(header)
        offset = header.end
    return boxes


def find_box(data: bytes, box_type: bytes, start: int, limit: int) -> BoxHeader | None:
    """Find the first box of *box_type* directly inside ``data[start:limit]``."""
    for header in iter_boxes(data, start, limit):
        if header.box_type == box_type:
            return header
    return None


def _read_full_box_header(data: bytes, header: BoxHeader) -> tuple[int, int]:
    """Return the (version, flags) pair of a FullBox."""
    start = header.payload_start
    if start + FULL_BOX_EXTRA_SIZE > header.end:
        raise AvifStructureError(f"Truncated FullBox header in '{header.box_type.decode('latin1')}'.")
    version = data[start]
    flags = int.from_bytes(data[start + 1 : start + FULL_BOX_EXTRA_SIZE], "big")
    return version, flags


def _read_uint(data: bytes, offset: int, size: int) -> int:
    """Read a big-endian unsigned integer of *size* bytes (0 yields 0)."""
    if size == 0:
        return 0
    return int.from_bytes(data[offset : offset + size], "big")


# ---- meta box parsing --------------------------------------------------------


def _parse_pitm(data: bytes, header: BoxHeader) -> int:
    """Parse the primary item box and return the primary item id."""
    version, _flags = _read_full_box_header(data, header)
    cursor = header.payload_start + FULL_BOX_EXTRA_SIZE
    width = 4 if version >= _ILOC_VERSION_WITH_CONSTRUCTION else 2
    return _read_uint(data, cursor, width)


def _parse_infe(data: bytes, header: BoxHeader) -> tuple[int, bytes, str, bool]:
    """Parse one item info entry into (item_id, item_type, name, is_hidden)."""
    version, flags = _read_full_box_header(data, header)
    cursor = header.payload_start + FULL_BOX_EXTRA_SIZE

    if version < _INFE_VERSION_WITH_TYPE:
        raise AvifStructureError(f"Unsupported 'infe' version {version}; expected 2 or later.")

    id_width = 4 if version >= _INFE_LARGE_ID_VERSION else 2
    item_id = _read_uint(data, cursor, id_width)
    cursor += id_width
    cursor += 2  # item_protection_index
    item_type = data[cursor : cursor + 4]
    cursor += 4

    terminator = data.find(b"\x00", cursor, header.end)
    name_end = terminator if terminator != -1 else header.end
    name = data[cursor:name_end].decode("utf-8", errors="replace")

    return item_id, item_type, name, bool(flags & _INFE_HIDDEN_FLAG)


def _parse_iinf(data: bytes, header: BoxHeader) -> dict[int, tuple[bytes, str, bool]]:
    """Parse the item info box into item_id -> (type, name, is_hidden)."""
    version, _flags = _read_full_box_header(data, header)
    cursor = header.payload_start + FULL_BOX_EXTRA_SIZE
    count_width = 2 if version == 0 else 4
    entry_count = _read_uint(data, cursor, count_width)
    cursor += count_width

    entries: dict[int, tuple[bytes, str, bool]] = {}
    for child in iter_boxes(data, cursor, header.end)[:entry_count]:
        if child.box_type != BOX_INFE:
            continue
        item_id, item_type, name, is_hidden = _parse_infe(data, child)
        entries[item_id] = (item_type, name, is_hidden)
    return entries


def _parse_iloc(data: bytes, header: BoxHeader) -> dict[int, tuple[int, tuple[tuple[int, int], ...]]]:
    """Parse the item location box into item_id -> (construction_method, extents)."""
    version, _flags = _read_full_box_header(data, header)
    cursor = header.payload_start + FULL_BOX_EXTRA_SIZE

    packed = data[cursor]
    offset_size = packed >> 4
    length_size = packed & 0x0F
    packed = data[cursor + 1]
    base_offset_size = packed >> 4
    index_size = packed & 0x0F if version >= _ILOC_VERSION_WITH_CONSTRUCTION else 0
    cursor += 2

    count_width = 4 if version >= _ILOC_VERSION_LARGE_ID else 2
    item_count = _read_uint(data, cursor, count_width)
    cursor += count_width

    locations: dict[int, tuple[int, tuple[tuple[int, int], ...]]] = {}
    for _ in range(item_count):
        if cursor >= header.end:
            break
        item_id = _read_uint(data, cursor, count_width)
        cursor += count_width

        construction_method = _CONSTRUCTION_METHOD_FILE
        if version >= _ILOC_VERSION_WITH_CONSTRUCTION:
            construction_method = _read_uint(data, cursor, 2) & 0x0F
            cursor += 2

        cursor += 2  # data_reference_index
        base_offset = _read_uint(data, cursor, base_offset_size)
        cursor += base_offset_size

        extent_count = _read_uint(data, cursor, 2)
        cursor += 2

        extents: list[tuple[int, int]] = []
        for _extent in range(extent_count):
            cursor += index_size  # extent_index, unused
            extent_offset = _read_uint(data, cursor, offset_size)
            cursor += offset_size
            extent_length = _read_uint(data, cursor, length_size)
            cursor += length_size
            extents.append((base_offset + extent_offset, extent_length))

        locations[item_id] = (construction_method, tuple(extents))

    return locations


def _parse_iref(data: bytes, header: BoxHeader) -> dict[bytes, dict[int, tuple[int, ...]]]:
    """Parse the item reference box into ref_type -> from_id -> to_ids."""
    version, _flags = _read_full_box_header(data, header)
    id_width = 4 if version >= _ILOC_VERSION_WITH_CONSTRUCTION else 2

    references: dict[bytes, dict[int, tuple[int, ...]]] = {}
    for child in iter_boxes(data, header.payload_start + FULL_BOX_EXTRA_SIZE, header.end):
        cursor = child.payload_start
        from_id = _read_uint(data, cursor, id_width)
        cursor += id_width
        reference_count = _read_uint(data, cursor, 2)
        cursor += 2

        to_ids: list[int] = []
        for _ in range(reference_count):
            if cursor + id_width > child.end:
                break
            to_ids.append(_read_uint(data, cursor, id_width))
            cursor += id_width

        references.setdefault(child.box_type, {})[from_id] = tuple(to_ids)

    return references


def _parse_ipma(data: bytes, header: BoxHeader) -> dict[int, tuple[tuple[int, bool], ...]]:
    """Parse the item property association box into item_id -> (index, essential)."""
    version, flags = _read_full_box_header(data, header)
    cursor = header.payload_start + FULL_BOX_EXTRA_SIZE

    entry_count = _read_uint(data, cursor, 4)
    cursor += 4
    id_width = 4 if version >= _IPMA_LARGE_ID_VERSION else 2
    large_index = bool(flags & _IPMA_LARGE_INDEX_FLAG)

    associations: dict[int, tuple[tuple[int, bool], ...]] = {}
    for _ in range(entry_count):
        if cursor >= header.end:
            break
        item_id = _read_uint(data, cursor, id_width)
        cursor += id_width
        association_count = data[cursor]
        cursor += 1

        entries: list[tuple[int, bool]] = []
        for _association in range(association_count):
            if large_index:
                raw = _read_uint(data, cursor, 2)
                cursor += 2
                essential = bool(raw & (_IPMA_ESSENTIAL_MASK << 8))
                entries.append((raw & _IPMA_LARGE_INDEX_MASK, essential))
            else:
                raw = data[cursor]
                cursor += 1
                entries.append((raw & _IPMA_SMALL_INDEX_MASK, bool(raw & _IPMA_ESSENTIAL_MASK)))

        associations[item_id] = tuple(entries)

    return associations


def _parse_iprp(data: bytes, header: BoxHeader) -> tuple[tuple[bytes, ...], dict[int, tuple[tuple[int, bool], ...]]]:
    """Parse the item properties box into (ipco children, ipma associations)."""
    property_boxes: tuple[bytes, ...] = ()
    associations: dict[int, tuple[tuple[int, bool], ...]] = {}

    for child in iter_boxes(data, header.payload_start, header.end):
        if child.box_type == BOX_IPCO:
            property_boxes = tuple(
                data[box.offset : box.end] for box in iter_boxes(data, child.payload_start, child.end)
            )
        elif child.box_type == BOX_IPMA:
            associations.update(_parse_ipma(data, child))

    return property_boxes, associations


def parse_meta(data: bytes) -> AvifMeta:
    """Parse the ``meta`` box of an AVIF file.

    Args:
        data: Complete AVIF file bytes.

    Returns:
        The parsed metadata structure.

    Raises:
        AvifStructureError: If the file has no ``meta`` box or it is malformed.
    """
    meta_header = find_box(data, BOX_META, 0, len(data))
    if meta_header is None:
        raise AvifStructureError("Not a valid AVIF file (missing 'meta' box).")

    children = iter_boxes(data, meta_header.payload_start + FULL_BOX_EXTRA_SIZE, meta_header.end)

    primary_item_id = 0
    info: dict[int, tuple[bytes, str, bool]] = {}
    locations: dict[int, tuple[int, tuple[tuple[int, int], ...]]] = {}
    references: dict[bytes, dict[int, tuple[int, ...]]] = {}
    property_boxes: tuple[bytes, ...] = ()
    associations: dict[int, tuple[tuple[int, bool], ...]] = {}
    idat = b""

    for child in children:
        if child.box_type == BOX_PITM:
            primary_item_id = _parse_pitm(data, child)
        elif child.box_type == BOX_IINF:
            info = _parse_iinf(data, child)
        elif child.box_type == BOX_ILOC:
            locations = _parse_iloc(data, child)
        elif child.box_type == BOX_IREF:
            references = _parse_iref(data, child)
        elif child.box_type == BOX_IPRP:
            property_boxes, associations = _parse_iprp(data, child)
        elif child.box_type == BOX_IDAT:
            idat = data[child.payload_start : child.end]

    items = {
        item_id: AvifItem(
            item_id=item_id,
            item_type=item_type,
            name=name,
            is_hidden=is_hidden,
            extents=locations.get(item_id, (_CONSTRUCTION_METHOD_FILE, ()))[1],
            construction_method=locations.get(item_id, (_CONSTRUCTION_METHOD_FILE, ()))[0],
            properties=associations.get(item_id, ()),
        )
        for item_id, (item_type, name, is_hidden) in info.items()
    }

    return AvifMeta(
        primary_item_id=primary_item_id,
        items=items,
        property_boxes=property_boxes,
        references=references,
        idat=idat,
    )


# ---- Item payload and property accessors -------------------------------------


def read_item_payload(data: bytes, meta: AvifMeta, item: AvifItem) -> bytes:
    """Concatenate the extents that make up an item's payload.

    Args:
        data: Complete AVIF file bytes.
        meta: Parsed metadata, used to resolve ``idat``-relative extents.
        item: Item whose payload is read.

    Returns:
        The item payload bytes.

    Raises:
        AvifStructureError: If an extent lies outside the source buffer or an
            unsupported construction method is used.
    """
    source = data
    if item.construction_method == _CONSTRUCTION_METHOD_IDAT:
        source = meta.idat
    elif item.construction_method != _CONSTRUCTION_METHOD_FILE:
        raise AvifStructureError(
            f"Item {item.item_id} uses unsupported construction_method {item.construction_method}."
        )

    chunks: list[bytes] = []
    for offset, length in item.extents:
        if offset + length > len(source):
            raise AvifStructureError(f"Item {item.item_id} extent exceeds the source buffer.")
        chunks.append(source[offset : offset + length])
    return b"".join(chunks)


def _parse_ispe(box: bytes | None) -> tuple[int, int]:
    """Return (width, height) from an ``ispe`` property box."""
    if box is None or len(box) < BOX_HEADER_SIZE + FULL_BOX_EXTRA_SIZE + _ISPE_PAYLOAD_SIZE:
        raise AvifStructureError("Image item is missing a valid 'ispe' property.")
    start = BOX_HEADER_SIZE + FULL_BOX_EXTRA_SIZE
    width, height = struct.unpack(">II", box[start : start + _ISPE_PAYLOAD_SIZE])
    return int(width), int(height)


def _split_colour_properties(meta: AvifMeta, item: AvifItem) -> tuple[bytes | None, bytes | None]:
    """Return the (nclx box, ICC profile bytes) associated with an item."""
    nclx: bytes | None = None
    icc: bytes | None = None
    for index, _essential in item.properties:
        box = meta._property_at(index)  # noqa: SLF001 - private accessor within module
        if box is None or box[4:8] != BOX_COLR:
            continue
        payload_start = BOX_HEADER_SIZE
        colour_type = box[payload_start : payload_start + _COLOUR_TYPE_SIZE]
        if colour_type == COLOUR_TYPE_NCLX:
            nclx = box
        elif colour_type in COLOUR_TYPE_ICC:
            icc = box[payload_start + _COLOUR_TYPE_SIZE :]
    return nclx, icc


def build_coded_image(data: bytes, meta: AvifMeta, item: AvifItem) -> CodedImage:
    """Lift a coded ``av01`` item out of the container without re-encoding it.

    Args:
        data: Complete AVIF file bytes.
        meta: Parsed metadata.
        item: The ``av01`` item to extract.

    Returns:
        The coded image with its descriptive properties.

    Raises:
        AvifStructureError: If the item is not a plain coded AV1 image.
    """
    if item.item_type == ITEM_TYPE_GRID:
        raise AvifStructureError(
            "Tiled (grid) AVIF images are not supported; re-save the file as a single-tile AVIF."
        )
    if item.item_type != ITEM_TYPE_AV01:
        raise AvifStructureError(
            f"Unsupported AVIF image item type {item.item_type!r}; expected {ITEM_TYPE_AV01!r}."
        )

    width, height = _parse_ispe(meta.property_bytes(item, BOX_ISPE))
    nclx, icc = _split_colour_properties(meta, item)

    return CodedImage(
        payload=read_item_payload(data, meta, item),
        width=width,
        height=height,
        av1c=meta.property_bytes(item, BOX_AV1C),
        pixi=meta.property_bytes(item, BOX_PIXI),
        nclx=nclx,
        icc_profile=icc,
        auxc=meta.property_bytes(item, BOX_AUXC),
    )


def get_primary_item(meta: AvifMeta) -> AvifItem:
    """Return the primary image item.

    Args:
        meta: Parsed metadata.

    Returns:
        The item referenced by ``pitm``, or the sole ``av01`` item when the
        primary item is a derived (``tmap``) item.

    Raises:
        AvifStructureError: If no usable image item exists.
    """
    primary = meta.items.get(meta.primary_item_id)
    if primary is not None and primary.item_type == ITEM_TYPE_AV01:
        return primary

    coded_items = [item for item in meta.items.values() if item.item_type == ITEM_TYPE_AV01]
    if not coded_items:
        raise AvifStructureError("AVIF file contains no coded image item.")
    return min(coded_items, key=lambda item: item.item_id)


def find_alpha_item(meta: AvifMeta) -> AvifItem | None:
    """Return the alpha auxiliary item attached to the primary image, if any."""
    primary_id = get_primary_item(meta).item_id
    for item_id, to_ids in meta.references.get(REF_TYPE_AUXL, {}).items():
        if primary_id in to_ids:
            candidate = meta.items.get(item_id)
            if candidate is not None and candidate.item_type == ITEM_TYPE_AV01:
                return candidate
    return None


# ---- Public helpers ----------------------------------------------------------


def decode_avif(avif_bytes: bytes) -> np.ndarray:
    """Decode AVIF bytes into a NumPy array.

    Args:
        avif_bytes: Complete AVIF file bytes.

    Returns:
        Decoded pixel array of shape (H, W), (H, W, 3) or (H, W, 4).

    Raises:
        AvifStructureError: If decoding fails.
    """
    try:
        return np.asarray(imagecodecs.avif_decode(avif_bytes))
    except Exception as exc:
        raise AvifStructureError(f"AVIF decoding failed: {exc}") from exc


def extract_icc_profile(avif_bytes: bytes) -> bytes | None:
    """Extract the ICC profile associated with the primary image item.

    Args:
        avif_bytes: Complete AVIF file bytes.

    Returns:
        Raw ICC profile bytes, or ``None`` when the file carries none.
    """
    try:
        meta = parse_meta(avif_bytes)
        _nclx, icc = _split_colour_properties(meta, get_primary_item(meta))
    except AvifStructureError:
        return None
    return icc


def has_gain_map_metadata(avif_bytes: bytes) -> bool:
    """Check whether the AVIF file already carries an ISO 21496-1 gain map.

    A gain map is signalled by a ``tmap`` (tone map derived image) item, as
    specified in ISO/IEC 23008-12:2024 and used by the AVIF gain map profile.

    Args:
        avif_bytes: Complete AVIF file bytes.

    Returns:
        True when a ``tmap`` item is present.
    """
    try:
        meta = parse_meta(avif_bytes)
    except AvifStructureError:
        return False
    return any(item.item_type == ITEM_TYPE_TMAP for item in meta.items.values())


def find_gain_map_item(meta: AvifMeta) -> AvifItem | None:
    """Locate the gain map image item referenced by a ``tmap`` item.

    The ``tmap`` item references exactly two items through ``iref``/``dimg``:
    the base image first, then the gain map.

    Args:
        meta: Parsed metadata.

    Returns:
        The gain map image item, or ``None`` when the file has no gain map.
    """
    dimg = meta.references.get(REF_TYPE_DIMG, {})
    for item in meta.items.values():
        if item.item_type != ITEM_TYPE_TMAP:
            continue
        to_ids = dimg.get(item.item_id, ())
        if len(to_ids) != _TMAP_DIMG_ENTRY_COUNT:
            continue
        return meta.items.get(to_ids[1])
    return None
