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
2. When generating, the SDR image is downsampled to half resolution (stride-2 subsampling) before computing CIE Y luminance and the gain map, reducing pixel count by 4x.
3. Gain map generation uses highlight-targeted inverse tone mapping:
	- Smoothstep mask isolates highlights above a configurable threshold.
	- Exponential stretch synthesizes HDR luminance from compressed highlights.
	- Log2 ratio between HDR and SDR luminance produces the raw gain map.
	- Guided filter (SDR luminance as guide) smooths the gain map with edge preservation.
	- Optional Gaussian bloom adds natural light halation.
4. Validate gain map channels and dtype (single-channel or RGB, uint8).
5. Compose Ultra HDR JPEG via API-4 (MPF container with XMP and ISO 21496-1 binary metadata). The encoder accepts gain maps of any resolution relative to the SDR JPEG.

## Public API

The package exports a minimal surface for programmatic use:

- `convert_jpeg_to_ultrahdr()` — end-to-end conversion (main entry point).
- `ConversionResult` — dataclass with output path, ICC presence, gain map source.
- `GainMapConfig` — configuration for the highlight-targeted generator.
- `generate_gain_map()` — standalone gain map generation from luminance arrays.
- `validate_gain_map()` — type/shape validation for external gain maps.
- `linearize_from_icc()` — ICC-aware linearization for advanced users.

## Module Boundaries

- `io.py`: byte I/O, JPEG decode, metadata extraction, gain map loading.
- `color.py`: ICC/CMS-based linearization and type control.
- `gainmap.py`: gain map validation and highlight-targeted generation.
- `encoder.py`: Ultra HDR packaging and metadata embedding.
- `pipeline.py`: orchestration, conversion result object, top-level workflow.
- `cli.py`: command line interface and argument validation.

## Data Contracts

- SDR base input: `uint8` ndarray with shape `(H, W, 3)`.
- Linearized SDR: float ndarray (typically `float32`) with shape `(H, W, C)`.
- Gain map: `uint8` ndarray of shape `(H, W)` or `(H, W, 1|3)`. Resolution is independent of the SDR base (typically half resolution when auto-generated).
- Output: Ultra HDR JPEG bytes encoded in MPF container.

## Gain Map Generation Details

- Input: linear CIE Y luminance from ICC-aware transform (at half resolution).
- Highlight isolation via Hermite smoothstep function — smooth, halo-free transition.
- Non-linear expansion targets only compressed highlights, leaving midtones/shadows untouched.
- Edge-aware guided filter uses SDR luminance as guide (not self-guided) so the gain map respects structural edges.
- Three-pass box filter approximates Gaussian blur when OpenCV is not installed.
- All intermediate computations use in-place NumPy operations to minimize memory allocations.

## Error Strategy

- Raise descriptive `RuntimeError` when required imagecodecs extensions are unavailable.
- Validate channel/dtype mismatches early with `ValueError`.
- Keep conversion methods deterministic and side-effect free except file writes.
