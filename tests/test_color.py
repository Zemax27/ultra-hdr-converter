import numpy as np

from ultra_hdr_converter.color import linearize_from_icc


def test_linearize_from_icc_falls_back_to_srgb_on_invalid_profile(monkeypatch: object) -> None:
    sdr = np.zeros((2, 2, 3), dtype=np.uint8)
    profile_calls: list[tuple[object, object]] = []
    transform_calls: list[tuple[object, object]] = []

    def _fake_profile(profile: object, **kwargs: object) -> object:
        profile_calls.append((profile, tuple(sorted(kwargs.items()))))
        if profile == b"icc":
            raise ValueError("bad icc")
        return f"profile:{profile}:{kwargs}"

    def _fake_transform(
        _array: np.ndarray,
        src: object,
        dst: object,
        **_kwargs: object,
    ) -> np.ndarray:
        transform_calls.append((src, dst))
        return np.ones((2, 2, 3), dtype=np.float32)

    monkeypatch.setattr("ultra_hdr_converter.color.imagecodecs.cms_profile", _fake_profile)
    monkeypatch.setattr("ultra_hdr_converter.color.imagecodecs.cms_transform", _fake_transform)

    output = linearize_from_icc(sdr, b"icc", outdtype=np.float32)

    assert output.dtype == np.float32
    assert output.shape == (2, 2, 3)
    assert profile_calls[0][0] == b"icc"
    assert profile_calls[1][0] == "srgb"
    assert transform_calls[0][0] != "srgb"
    assert transform_calls[0][1] != "srgb"
