# Ultra HDR Converter Architecture

## Goals

- Implement an end-to-end Ultra HDR packaging flow from SDR JPEG + gain map.
- Preserve color intent through ICC profile handling for SDR base.
- Keep the conversion steps modular for reproducibility and testing.

## Pipeline Phases

### Phase A: Decode and Linearize via ICC

1. Read JPEG bytes.
2. Decode SDR raster with `imagecodecs.jpeg_decode`.
3. Parse JPEG metadata with `imagecodecs.jpeg_metadata` and extract `icc_profile`.
4. Build CMS source and linear destination profiles with `imagecodecs.cms_profile`.
5. Convert to linear light with `imagecodecs.cms_transform` (default `float32`).

Fallback behavior:

- If no ICC exists, linearize with the sRGB assumption.

### Phase B: Encode Ultra HDR with Gain Map

1. Load a user-provided gain map (`.npy` or standard image), or generate one from linear luminance.
2. Supported internal generation methods:
	- `log2`: deterministic baseline from clipped log2 luminance.
	- `radiance`: reflectance-aware guided-Retinex illumination map with robust percentile normalization.
3. Validate gain map dimensions and channels (single-channel or RGB).
4. Encode Ultra HDR container with `imagecodecs.ultrahdr_encode`.
5. Optionally include original ICC metadata for SDR base compatibility.

## Module Boundaries

- `io.py`: byte I/O, JPEG decode, metadata extraction, gain map loading.
- `color.py`: ICC/CMS-based linearization and type control.
- `gainmap.py`: gain map validation, baseline generation, and radiance-guided generation.
- `encoder.py`: Ultra HDR packaging and metadata embedding.
- `pipeline.py`: orchestration, conversion result object, top-level workflow.
- `cli.py`: command line interface and argument validation.

## Data Contracts

- SDR base input: `uint8` ndarray with shape `(H, W, 3)`.
- Linearized SDR: float ndarray (typically `float32`) with shape `(H, W, C)`.
- Gain map: `uint8` ndarray of shape `(H, W)` or `(H, W, 1|3)`.
- Output: Ultra HDR JPEG bytes encoded in MPF container.

## Radiance Generation Details

- Input: linear SDR luminance from ICC-aware transform.
- Optional prefilter downscale for speed (`resize_factor`).
- Guided filtering runs in pure NumPy (integral-image box filter), avoiding OpenCV dependencies.
- Illumination map is normalized in log space with low/high percentile clipping for stability.

## Error Strategy

- Raise descriptive `RuntimeError` when required imagecodecs extensions are unavailable.
- Validate shape/channel mismatches early with `ValueError`.
- Keep conversion methods deterministic and side-effect free except file writes.
