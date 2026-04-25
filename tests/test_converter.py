from pathlib import Path

import numpy as np
import pytest

from ultra_hdr_converter.core.converter import convert_jpeg_to_ultrahdr
from ultra_hdr_converter.errors import GainMapShapeMismatchError, AlreadyUltraHDRError


def test_pipeline_uses_external_gain_map(monkeypatch: object, tmp_path: Path) -> None:
    input_file = tmp_path / "input.jpg"
    output_file = tmp_path / "output.jpg"
    gain_map_file = tmp_path / "gain.npy"

    input_file.write_bytes(b"jpeg")
    np.save(gain_map_file, np.full((4, 4), 100, dtype=np.uint8))

    fake_sdr = np.zeros((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr("ultra_hdr_converter.core.converter.read_bytes", lambda _: b"jpeg")
    monkeypatch.setattr("ultra_hdr_converter.core.converter.decode_jpeg", lambda _: fake_sdr)
    monkeypatch.setattr("ultra_hdr_converter.core.converter.extract_icc_profile", lambda _: b"icc")
    monkeypatch.setattr(
        "ultra_hdr_converter.core.converter.encode_ultrahdr",
        lambda **_kwargs: b"ultrahdr",
    )

    written: dict[str, bytes] = {}

    def _capture_write(path: Path, payload: bytes) -> None:
        written[str(path)] = payload

    monkeypatch.setattr("ultra_hdr_converter.core.converter.write_bytes", _capture_write)

    result = convert_jpeg_to_ultrahdr(
        input_jpeg=input_file,
        output_jpeg=output_file,
        gain_map_path=gain_map_file,
    )

    assert result.gain_map_source == "external"
    assert result.has_icc is True
    assert written[str(output_file)] == b"ultrahdr"


def test_pipeline_uses_generated_gain_map(monkeypatch: object, tmp_path: Path) -> None:
    input_file = tmp_path / "input.jpg"
    output_file = tmp_path / "output.jpg"

    fake_sdr = np.zeros((4, 4, 3), dtype=np.uint8)
    fake_gain = np.full((2, 2), 111, dtype=np.uint8)

    monkeypatch.setattr("ultra_hdr_converter.core.converter.read_bytes", lambda _: b"jpeg")
    monkeypatch.setattr("ultra_hdr_converter.core.converter.decode_jpeg", lambda _: fake_sdr)
    monkeypatch.setattr("ultra_hdr_converter.core.converter.extract_icc_profile", lambda _: None)
    monkeypatch.setattr(
        "ultra_hdr_converter.core.converter.extract_xyz_luminance",
        lambda *_args, **_kwargs: np.ones((2, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        "ultra_hdr_converter.core.converter.generate_gain_map",
        lambda *_args, **_kwargs: fake_gain,
    )
    monkeypatch.setattr(
        "ultra_hdr_converter.core.converter.encode_ultrahdr",
        lambda **_kwargs: b"ultrahdr",
    )

    written: dict[str, bytes] = {}

    def _capture_write(path: Path, payload: bytes) -> None:
        written[str(path)] = payload

    monkeypatch.setattr("ultra_hdr_converter.core.converter.write_bytes", _capture_write)

    result = convert_jpeg_to_ultrahdr(
        input_jpeg=input_file,
        output_jpeg=output_file,
    )

    assert result.gain_map_source == "generated"
    assert result.has_icc is False
    assert written[str(output_file)] == b"ultrahdr"


def test_pipeline_reports_progress_steps(monkeypatch: object, tmp_path: Path) -> None:
    input_file = tmp_path / "input.jpg"
    output_file = tmp_path / "output.jpg"

    fake_sdr = np.zeros((4, 4, 3), dtype=np.uint8)
    fake_gain = np.full((2, 2), 111, dtype=np.uint8)

    monkeypatch.setattr("ultra_hdr_converter.core.converter.read_bytes", lambda _: b"jpeg")
    monkeypatch.setattr("ultra_hdr_converter.core.converter.decode_jpeg", lambda _: fake_sdr)
    monkeypatch.setattr("ultra_hdr_converter.core.converter.extract_icc_profile", lambda _: None)
    monkeypatch.setattr(
        "ultra_hdr_converter.core.converter.extract_xyz_luminance",
        lambda *_args, **_kwargs: np.ones((2, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        "ultra_hdr_converter.core.converter.generate_gain_map",
        lambda *_args, **_kwargs: fake_gain,
    )
    monkeypatch.setattr(
        "ultra_hdr_converter.core.converter.encode_ultrahdr",
        lambda **_kwargs: b"ultrahdr",
    )
    monkeypatch.setattr("ultra_hdr_converter.core.converter.write_bytes", lambda *_args, **_kwargs: None)

    progress_updates: list[tuple[str, int, int]] = []

    def _capture_progress(message: str, step: int, total_steps: int) -> None:
        progress_updates.append((message, step, total_steps))

    convert_jpeg_to_ultrahdr(
        input_jpeg=input_file,
        output_jpeg=output_file,
        progress_callback=_capture_progress,
    )

    assert progress_updates == [
        ("Reading and decoding input JPEG", 1, 5),
        ("Extracting luminance from SDR", 2, 5),
        ("Generating highlight-targeted gain map", 3, 5),
        ("Encoding Ultra HDR metadata and container", 4, 5),
        ("Writing final output file", 5, 5),
    ]


def test_pipeline_raises_on_gain_map_shape_mismatch(monkeypatch: object, tmp_path: Path) -> None:
    input_file = tmp_path / "input.jpg"
    output_file = tmp_path / "output.jpg"
    gain_map_file = tmp_path / "gain.npy"

    input_file.write_bytes(b"jpeg")
    np.save(gain_map_file, np.full((2, 2), 100, dtype=np.uint8))

    fake_sdr = np.zeros((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr("ultra_hdr_converter.core.converter.read_bytes", lambda _: b"jpeg")
    monkeypatch.setattr("ultra_hdr_converter.core.converter.decode_jpeg", lambda _: fake_sdr)
    monkeypatch.setattr("ultra_hdr_converter.core.converter.extract_icc_profile", lambda _: None)
    monkeypatch.setattr(
        "ultra_hdr_converter.core.converter.encode_ultrahdr",
        lambda **_kwargs: b"ultrahdr",
    )

    with pytest.raises(GainMapShapeMismatchError, match="does not match"):
        convert_jpeg_to_ultrahdr(
            input_jpeg=input_file,
            output_jpeg=output_file,
            gain_map_path=gain_map_file,
        )


def test_pipeline_raises_already_ultrahdr_error(monkeypatch: object, tmp_path: Path) -> None:
    input_file = tmp_path / "input.jpg"
    output_file = tmp_path / "output.jpg"
    input_file.write_bytes(b"jpeg")

    monkeypatch.setattr("ultra_hdr_converter.core.converter.read_bytes", lambda _: b"jpeg")
    monkeypatch.setattr("ultra_hdr_converter.core.converter.has_ultrahdr_metadata", lambda _: True)

    with pytest.raises(AlreadyUltraHDRError, match="already an Ultra HDR image"):
        convert_jpeg_to_ultrahdr(
            input_jpeg=input_file,
            output_jpeg=output_file,
        )


def test_pipeline_uses_embedded_mpf_gain_map(monkeypatch: object, tmp_path: Path) -> None:
    input_file = tmp_path / "input.jpg"
    output_file = tmp_path / "output.jpg"
    input_file.write_bytes(b"jpeg")

    fake_sdr = np.zeros((4, 4, 3), dtype=np.uint8)
    fake_gain = np.full((4, 4), 111, dtype=np.uint8)

    monkeypatch.setattr("ultra_hdr_converter.core.converter.read_bytes", lambda _: b"jpeg")
    monkeypatch.setattr("ultra_hdr_converter.core.converter.has_ultrahdr_metadata", lambda _: False)
    monkeypatch.setattr("ultra_hdr_converter.core.converter.extract_mpf_gain_map", lambda _: b"mpf_jpeg")
    
    # decode_jpeg is called twice: once for SDR, once for MPF gain map.
    # We will just return the correctly shaped fake_gain for the second call.
    decode_calls = []
    def _mock_decode(b: bytes) -> np.ndarray:
        decode_calls.append(b)
        if b == b"mpf_jpeg":
            return fake_gain
        return fake_sdr
        
    monkeypatch.setattr("ultra_hdr_converter.core.converter.decode_jpeg", _mock_decode)
    monkeypatch.setattr("ultra_hdr_converter.core.converter.extract_icc_profile", lambda _: None)
    monkeypatch.setattr(
        "ultra_hdr_converter.core.converter.encode_ultrahdr",
        lambda **_kwargs: b"ultrahdr",
    )
    monkeypatch.setattr("ultra_hdr_converter.core.converter.write_bytes", lambda *_args, **_kwargs: None)

    result = convert_jpeg_to_ultrahdr(
        input_jpeg=input_file,
        output_jpeg=output_file,
    )

    assert result.gain_map_source == "embedded"
    assert b"mpf_jpeg" in decode_calls
