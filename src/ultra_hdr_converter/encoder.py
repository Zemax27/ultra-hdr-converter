"""Ultra HDR encoder wrapper."""

from __future__ import annotations

import imagecodecs
import numpy as np


def _ensure_ultrahdr_available() -> None:
    if not hasattr(imagecodecs, "ultrahdr_encode"):
        raise RuntimeError(
            "imagecodecs ultrahdr extension is unavailable. Install full imagecodecs with ultrahdr support."
        )


def encode_ultrahdr(
    sdr_base: np.ndarray,
    gain_map: np.ndarray,
    icc_profile: bytes | None,
) -> bytes:
    """Encode SDR base and gain map into a single Ultra HDR JPEG payload."""
    _ensure_ultrahdr_available()

    if icc_profile:
        return bytes(
            imagecodecs.ultrahdr_encode(
                sdr_base,
                gainmap=gain_map,
                metadata={"icc_profile": icc_profile},
            )
        )

    return bytes(imagecodecs.ultrahdr_encode(sdr_base, gainmap=gain_map))
