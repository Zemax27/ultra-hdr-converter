from pathlib import Path

import numpy as np
import pytest

from ultra_hdr_converter.core.converter import convert_to_ultrahdr
from ultra_hdr_converter.core.formats import ImageFormat
from ultra_hdr_converter.errors import AlreadyUltraHDRError, GainMapShapeMismatchError

EXPECTED_EXTERNAL_BOOST = 6.0
EXPECTED_EMBEDDED_BOOST = 5.0

# Leading bytes that make ``detect_format`` report each container. The pipeline
# sniffs magic bytes, so stubbed payloads must still start with a real header.
JPEG_MAGIC = b"\xff\xd8\xff\xe0"
AVIF_MAGIC = b"\x00\x00\x00\x20ftypavif"

CONVERTER = "ultra_hdr_converter.core.converter"


def _patch_pipeline(
    monkeypatch: object,
    *,
    input_bytes: bytes = JPEG_MAGIC,
    sdr: np.ndarray | None = None,
    icc: bytes | None = None,
    embedded: np.ndarray | None = None,
    encode: object = None,
) -> dict[str, bytes]:
    """Stub the pipeline's I/O boundary and capture what gets written."""
    written: dict[str, bytes] = {}

    monkeypatch.setattr(f"{CONVERTER}.read_bytes", lambda _: input_bytes)
    monkeypatch.setattr(f"{CONVERTER}.has_ultrahdr_metadata", lambda *_a, **_k: False)
    monkeypatch.setattr(
        f"{CONVERTER}.decode_image",
        lambda *_a, **_k: np.zeros((4, 4, 3), dtype=np.uint8) if sdr is None else sdr,
    )
    monkeypatch.setattr(f"{CONVERTER}.extract_icc_profile", lambda *_a, **_k: icc)
    monkeypatch.setattr(f"{CONVERTER}.extract_embedded_gain_map", lambda *_a, **_k: embedded)
    monkeypatch.setattr(f"{CONVERTER}.encode_ultrahdr_jpeg", encode or (lambda **_kwargs: b"ultrahdr"))
    monkeypatch.setattr(f"{CONVERTER}.write_bytes", lambda path, payload: written.__setitem__(str(path), payload))
    return written


def test_pipeline_uses_external_gain_map(monkeypatch: object, tmp_path: Path) -> None:
    output_file = tmp_path / "output.jpg"
    gain_map_file = tmp_path / "gain.npy"
    np.save(gain_map_file, np.full((4, 4), 100, dtype=np.uint8))

    written = _patch_pipeline(monkeypatch, icc=b"icc")

    result = convert_to_ultrahdr(
        input_path=tmp_path / "input.jpg",
        output_path=output_file,
        gain_map_path=gain_map_file,
    )

    assert result.gain_map_source == "external"
    assert result.has_icc is True
    assert result.input_format is ImageFormat.JPEG
    assert result.output_format is ImageFormat.JPEG
    assert written[str(output_file)] == b"ultrahdr"


def test_pipeline_uses_generated_gain_map(monkeypatch: object, tmp_path: Path) -> None:
    output_file = tmp_path / "output.jpg"
    written = _patch_pipeline(monkeypatch)
    monkeypatch.setattr(f"{CONVERTER}.extract_xyz_luminance", lambda *_a, **_k: np.ones((2, 2), dtype=np.float32))
    monkeypatch.setattr(f"{CONVERTER}.generate_gain_map", lambda *_a, **_k: np.full((2, 2), 111, dtype=np.uint8))

    result = convert_to_ultrahdr(input_path=tmp_path / "input.jpg", output_path=output_file)

    assert result.gain_map_source == "generated"
    assert result.has_icc is False
    assert written[str(output_file)] == b"ultrahdr"


def test_pipeline_reports_progress_steps(monkeypatch: object, tmp_path: Path) -> None:
    _patch_pipeline(monkeypatch)
    monkeypatch.setattr(f"{CONVERTER}.extract_xyz_luminance", lambda *_a, **_k: np.ones((2, 2), dtype=np.float32))
    monkeypatch.setattr(f"{CONVERTER}.generate_gain_map", lambda *_a, **_k: np.full((2, 2), 111, dtype=np.uint8))

    progress_updates: list[tuple[str, int, int]] = []

    convert_to_ultrahdr(
        input_path=tmp_path / "input.jpg",
        output_path=tmp_path / "output.jpg",
        progress_callback=lambda message, step, total: progress_updates.append((message, step, total)),
    )

    assert progress_updates == [
        ("Reading and decoding input image", 1, 5),
        ("Extracting luminance from SDR", 2, 5),
        ("Generating highlight-targeted gain map", 3, 5),
        ("Encoding Ultra HDR metadata and container", 4, 5),
        ("Writing final output file", 5, 5),
    ]


def test_pipeline_raises_on_gain_map_shape_mismatch(monkeypatch: object, tmp_path: Path) -> None:
    gain_map_file = tmp_path / "gain.npy"
    np.save(gain_map_file, np.full((2, 2), 100, dtype=np.uint8))

    _patch_pipeline(monkeypatch)

    with pytest.raises(GainMapShapeMismatchError, match="does not match"):
        convert_to_ultrahdr(
            input_path=tmp_path / "input.jpg",
            output_path=tmp_path / "output.jpg",
            gain_map_path=gain_map_file,
        )


def test_pipeline_raises_already_ultrahdr_error(monkeypatch: object, tmp_path: Path) -> None:
    _patch_pipeline(monkeypatch)
    monkeypatch.setattr(f"{CONVERTER}.has_ultrahdr_metadata", lambda *_a, **_k: True)

    with pytest.raises(AlreadyUltraHDRError, match="already an Ultra HDR image"):
        convert_to_ultrahdr(input_path=tmp_path / "input.jpg", output_path=tmp_path / "output.jpg")


def test_pipeline_uses_embedded_gain_map(monkeypatch: object, tmp_path: Path) -> None:
    written = _patch_pipeline(monkeypatch, embedded=np.full((4, 4), 111, dtype=np.uint8))

    result = convert_to_ultrahdr(input_path=tmp_path / "input.jpg", output_path=tmp_path / "output.jpg")

    assert result.gain_map_source == "embedded"
    assert written


def test_pipeline_already_ultrahdr_error_includes_path(monkeypatch: object, tmp_path: Path) -> None:
    _patch_pipeline(monkeypatch)
    monkeypatch.setattr(f"{CONVERTER}.has_ultrahdr_metadata", lambda *_a, **_k: True)

    with pytest.raises(AlreadyUltraHDRError) as exc_info:
        convert_to_ultrahdr(input_path=tmp_path / "my_photo.jpg", output_path=tmp_path / "output.jpg")

    assert "my_photo.jpg" in str(exc_info.value)


def test_pipeline_embedded_gain_map_shape_mismatch(monkeypatch: object, tmp_path: Path) -> None:
    """An embedded gain map with wrong dimensions should raise GainMapShapeMismatchError."""
    _patch_pipeline(
        monkeypatch,
        sdr=np.zeros((8, 8, 3), dtype=np.uint8),
        embedded=np.full((4, 4), 111, dtype=np.uint8),
    )

    with pytest.raises(GainMapShapeMismatchError, match="does not match"):
        convert_to_ultrahdr(input_path=tmp_path / "input.jpg", output_path=tmp_path / "output.jpg")


def test_pipeline_forwards_max_content_boost_for_external(monkeypatch: object, tmp_path: Path) -> None:
    """max_content_boost should be forwarded to the encoder when using an external gain map."""
    gain_map_file = tmp_path / "gain.npy"
    np.save(gain_map_file, np.full((4, 4), 100, dtype=np.uint8))

    captured: list[dict[str, object]] = []

    def _capture(**kwargs: object) -> bytes:
        captured.append(kwargs)
        return b"ultrahdr"

    _patch_pipeline(monkeypatch, encode=_capture)

    convert_to_ultrahdr(
        input_path=tmp_path / "input.jpg",
        output_path=tmp_path / "output.jpg",
        gain_map_path=gain_map_file,
        max_content_boost=EXPECTED_EXTERNAL_BOOST,
    )

    assert captured[0]["max_content_boost"] == EXPECTED_EXTERNAL_BOOST


def test_pipeline_forwards_max_content_boost_for_embedded(monkeypatch: object, tmp_path: Path) -> None:
    """max_content_boost should be forwarded to the encoder when using an embedded gain map."""
    captured: list[dict[str, object]] = []

    def _capture(**kwargs: object) -> bytes:
        captured.append(kwargs)
        return b"ultrahdr"

    _patch_pipeline(monkeypatch, embedded=np.full((4, 4), 111, dtype=np.uint8), encode=_capture)

    convert_to_ultrahdr(
        input_path=tmp_path / "input.jpg",
        output_path=tmp_path / "output.jpg",
        max_content_boost=EXPECTED_EMBEDDED_BOOST,
    )

    assert captured[0]["max_content_boost"] == EXPECTED_EMBEDDED_BOOST


def test_pipeline_output_format_follows_output_suffix(monkeypatch: object, tmp_path: Path) -> None:
    """An .avif output path should select the AVIF container without an explicit flag."""
    _patch_pipeline(monkeypatch, embedded=np.full((4, 4), 111, dtype=np.uint8))
    monkeypatch.setattr(f"{CONVERTER}.encode_coded_image", lambda *_a, **_k: "coded")
    monkeypatch.setattr(f"{CONVERTER}.encode_ultrahdr_avif", lambda **_kwargs: b"avif-ultrahdr")

    result = convert_to_ultrahdr(input_path=tmp_path / "input.jpg", output_path=tmp_path / "output.avif")

    assert result.input_format is ImageFormat.JPEG
    assert result.output_format is ImageFormat.AVIF


def test_pipeline_explicit_output_format_overrides_suffix(monkeypatch: object, tmp_path: Path) -> None:
    """An explicit output_format wins over the output file suffix."""
    _patch_pipeline(monkeypatch, embedded=np.full((4, 4), 111, dtype=np.uint8))
    monkeypatch.setattr(f"{CONVERTER}.encode_coded_image", lambda *_a, **_k: "coded")
    monkeypatch.setattr(f"{CONVERTER}.encode_ultrahdr_avif", lambda **_kwargs: b"avif-ultrahdr")

    result = convert_to_ultrahdr(
        input_path=tmp_path / "input.jpg",
        output_path=tmp_path / "output.jpg",
        output_format=ImageFormat.AVIF,
    )

    assert result.output_format is ImageFormat.AVIF


def test_pipeline_avif_input_reuses_coded_base(monkeypatch: object, tmp_path: Path) -> None:
    """An AVIF input converted to AVIF must reuse the coded payload, not re-encode it."""
    _patch_pipeline(
        monkeypatch,
        input_bytes=AVIF_MAGIC,
        embedded=np.full((4, 4), 111, dtype=np.uint8),
    )

    def _fail_reencode(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("AVIF input must not be re-encoded when the output is also AVIF")

    monkeypatch.setattr(f"{CONVERTER}.encode_coded_image", _fail_reencode)
    monkeypatch.setattr(f"{CONVERTER}.extract_base_image", lambda _data: ("base", None))
    monkeypatch.setattr(f"{CONVERTER}.encode_ultrahdr_avif", lambda **_kwargs: b"avif-ultrahdr")

    result = convert_to_ultrahdr(input_path=tmp_path / "input.avif", output_path=tmp_path / "output.avif")

    assert result.input_format is ImageFormat.AVIF
    assert result.output_format is ImageFormat.AVIF


def test_validation_quality_out_of_range() -> None:
    """quality must be between 0 and 100 inclusive."""
    with pytest.raises(ValueError, match="quality must be between 0 and 100"):
        convert_to_ultrahdr("in.jpg", "out.jpg", quality=101)
    with pytest.raises(ValueError, match="quality must be between 0 and 100"):
        convert_to_ultrahdr("in.jpg", "out.jpg", quality=-1)


def test_validation_max_content_boost_non_positive() -> None:
    """max_content_boost must be positive when provided."""
    with pytest.raises(ValueError, match="max_content_boost must be positive"):
        convert_to_ultrahdr("in.jpg", "out.jpg", max_content_boost=0)
    with pytest.raises(ValueError, match="max_content_boost must be positive"):
        convert_to_ultrahdr("in.jpg", "out.jpg", max_content_boost=-2.5)


def test_validation_same_input_output_path(tmp_path: Path) -> None:
    """Input and output cannot be the same file."""
    same = tmp_path / "photo.jpg"
    # No need to create the file; validation occurs before I/O
    with pytest.raises(ValueError, match="cannot be the same file"):
        convert_to_ultrahdr(same, same)
