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
2. When generating, the SDR image is downsampled to half resolution (stride-2 subsampling) before computing luminance and the gain map. This reduces the pixel count by 4x through the CMS transform and gain map algorithms. The `radiance` method applies an additional `resize_factor` (default 0.5) internally for the guided filter, yielding 1/16th pixel count at that stage.
3. Supported internal generation methods:
	- `log2`: deterministic baseline from clipped log2 luminance.
	- `radiance`: reflectance-aware guided-Retinex illumination map with robust percentile normalization.
4. Validate gain map channels and dtype (single-channel or RGB, uint8).
5. Compose Ultra HDR JPEG via API-4 (MPF container with XMP and ISO 21496-1 binary metadata). The encoder accepts gain maps of any resolution relative to the SDR JPEG.

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
- Gain map: `uint8` ndarray of shape `(H, W)` or `(H, W, 1|3)`. Resolution is independent of the SDR base (typically half resolution when auto-generated).
- Output: Ultra HDR JPEG bytes encoded in MPF container.

## Radiance Generation Details

- Input: linear SDR luminance from ICC-aware transform (at half resolution).
- Optional prefilter downscale for speed (`resize_factor`, default 0.5, applied on top of the half-resolution input).
- Guided filtering runs in pure NumPy (integral-image box filter returning means directly), avoiding OpenCV dependencies. Uses OpenCV `ximgproc.guidedFilter` when available for additional speed.
- All intermediate computations use in-place NumPy operations to minimize memory allocations.
- Illumination map is normalized in log space with low/high percentile clipping for stability.

## Error Strategy

- Raise descriptive `RuntimeError` when required imagecodecs extensions are unavailable.
- Validate channel/dtype mismatches early with `ValueError`.
- Keep conversion methods deterministic and side-effect free except file writes.
