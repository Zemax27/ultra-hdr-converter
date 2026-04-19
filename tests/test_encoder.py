import numpy as np
import pytest

from ultra_hdr_converter.encoder import encode_ultrahdr


def test_encode_ultrahdr_uses_gainmap_api_when_available(monkeypatch: object) -> None:
    captured: dict[str, object] = {}

    def _fake_ultrahdr_encode(data: np.ndarray, **kwargs: object) -> bytes:
        captured["data"] = data
        captured["kwargs"] = kwargs
        return b"encoded"

    monkeypatch.setattr("ultra_hdr_converter.encoder.imagecodecs.ultrahdr_encode", _fake_ultrahdr_encode)

    result = encode_ultrahdr(
        sdr_base=np.zeros((2, 2, 3), dtype=np.uint8),
        gain_map=np.zeros((2, 2), dtype=np.uint8),
        icc_profile=b"icc",
        linear_sdr=np.zeros((2, 2, 3), dtype=np.float32),
    )

    assert result == b"encoded"
    assert isinstance(captured["kwargs"], dict)
    assert "gainmap" in captured["kwargs"]


def test_encode_ultrahdr_falls_back_to_rgba_half_when_gainmap_keyword_unsupported(
    monkeypatch: object,
) -> None:
    calls: list[tuple[np.ndarray, dict[str, object]]] = []

    def _fake_ultrahdr_encode(data: np.ndarray, **kwargs: object) -> bytes:
        calls.append((data, kwargs))
        if kwargs:
            raise TypeError("ultrahdr_encode() got an unexpected keyword argument 'gainmap'")
        return b"encoded-rgba"

    monkeypatch.setattr("ultra_hdr_converter.encoder.imagecodecs.ultrahdr_encode", _fake_ultrahdr_encode)

    result = encode_ultrahdr(
        sdr_base=np.zeros((3, 4, 3), dtype=np.uint8),
        gain_map=np.full((3, 4), 100, dtype=np.uint8),
        icc_profile=None,
        linear_sdr=np.ones((3, 4, 3), dtype=np.float32),
    )

    assert result == b"encoded-rgba"
    rgba_input = calls[-1][0]
    assert rgba_input.shape == (3, 4, 4)
    assert rgba_input.dtype == np.float16
    assert calls[-1][1] == {}


def test_encode_ultrahdr_requires_linear_sdr_for_rgba_fallback(monkeypatch: object) -> None:
    def _fake_ultrahdr_encode(_data: np.ndarray, **_kwargs: object) -> bytes:
        raise TypeError("ultrahdr_encode() got an unexpected keyword argument 'gainmap'")

    monkeypatch.setattr("ultra_hdr_converter.encoder.imagecodecs.ultrahdr_encode", _fake_ultrahdr_encode)

    with pytest.raises(RuntimeError):
        encode_ultrahdr(
            sdr_base=np.zeros((2, 2, 3), dtype=np.uint8),
            gain_map=np.zeros((2, 2), dtype=np.uint8),
            icc_profile=None,
            linear_sdr=None,
        )
