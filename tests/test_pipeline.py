from pathlib import Path

import numpy as np

from ultra_hdr_converter.pipeline import convert_jpeg_to_ultrahdr


def test_pipeline_uses_external_gain_map(monkeypatch: object, tmp_path: Path) -> None:
    input_file = tmp_path / "input.jpg"
    output_file = tmp_path / "output.jpg"
    gain_map_file = tmp_path / "gain.npy"

    input_file.write_bytes(b"jpeg")
    np.save(gain_map_file, np.full((4, 4), 100, dtype=np.uint8))

    fake_sdr = np.zeros((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr("ultra_hdr_converter.pipeline.read_bytes", lambda _: b"jpeg")
    monkeypatch.setattr("ultra_hdr_converter.pipeline.decode_jpeg", lambda _: fake_sdr)
    monkeypatch.setattr("ultra_hdr_converter.pipeline.extract_icc_profile", lambda _: b"icc")
    monkeypatch.setattr(
        "ultra_hdr_converter.pipeline.linearize_from_icc",
        lambda *_args, **_kwargs: np.ones((4, 4, 3), dtype=np.float32),
    )
    monkeypatch.setattr(
        "ultra_hdr_converter.pipeline.encode_ultrahdr",
        lambda **_kwargs: b"ultrahdr",
    )

    written: dict[str, bytes] = {}

    def _capture_write(path: Path, payload: bytes) -> None:
        written[str(path)] = payload

    monkeypatch.setattr("ultra_hdr_converter.pipeline.write_bytes", _capture_write)

    result = convert_jpeg_to_ultrahdr(
        input_jpeg=input_file,
        output_jpeg=output_file,
        gain_map_path=gain_map_file,
    )

    assert result.gain_map_source == "external"
    assert written[str(output_file)] == b"ultrahdr"
