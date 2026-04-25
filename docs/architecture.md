# Ultra HDR Converter Architecture

## Goals

- Implement an end-to-end Ultra HDR packaging flow from SDR JPEG + gain map.
- Preserve color intent through ICC profile handling for SDR base.
- Keep the conversion steps modular for reproducibility and testing.
- Support both single-file and batch-oriented entry points without changing the core conversion path.

## Pipeline Phases

### Phase A: Decode and Linearize via ICC

1. Read JPEG bytes.
2. Check for existing Ultra HDR or ISO 21496-1 metadata. If present, skip processing by raising `AlreadyUltraHDRError`.
3. Decode SDR raster with `imagecodecs.jpeg_decode`.
4. Parse JPEG metadata with `imagecodecs.jpeg_metadata` and extract `icc_profile`.
5. Build CMS source and linear destination profiles with `imagecodecs.cms_profile`.
6. Convert to linear light with `imagecodeacs.cms_transform` (default `float32`).

**Fallback behavior:** If no ICC exists, linearize with the sRGB assumption.

### Phase B: Encode Ultra HDR with Gain Map

1. Load a user-provided gain map (`.npy` or standard image), extract an embedded MPF gain map from the input file, or generate one from linear luminance.
2. When generating, the SDR image is downsampled to half resolution (stride-2 subsampling) before computing CIE Y luminance and the gain map, reducing pixel count by 4x.
3. Gain map generation uses highlight-targeted inverse tone mapping:
   - **Smoothstep mask**: isolates highlights above a configurable threshold.
   - **Exponential stretch**: synthesizes HDR luminance from compressed highlights.
   - **Log₂ ratio**: gain map = log₂(HDR / SDR luminance).
   - **Guided filter**: smooths the gain map using SDR luminance as guide to preserve edges.
   - **Gaussian bloom**: optional halation effect on gain map peaks.
4. Validate gain map channels and dtype (single-channel or RGB, uint8).
5. Compose Ultra HDR JPEG via API-4 (MPF container with XMP and ISO 21496-1 binary metadata). The encoder accepts gain maps of any resolution relative to the SDR JPEG.

## Gain Map Algorithm

The built-in generator uses a **custom highlight-targeted inverse tone-mapping algorithm** to synthesize HDR luminance from an SDR image. While the *encoding* of the final output adheres to ISO 21496-1 and Adobe Ultra HDR container specifications, the gain map *generation* itself is a proprietary heuristic not defined by those standards.

### Step 1 — Soft Highlight Isolation

A Hermite smoothstep function computes a smooth mask over the compressed highlight range:

```python
mask = smoothstep(threshold, 1.0, luminance)
```

This isolates pixels above `threshold` (e.g. 0.5) while leaving midtones and shadows untouched, preventing halos and preserving shadow detail.

### Step 2 — Non-Linear Highlight Expansion

An exponential curve targets only the compressed highlights:

```python
stretched = (luminance ^ expansion_gamma) * max_boost_factor * mask
```

`expansion_gamma` (default 2.2) stretches the highlights toward HDR luminance range, while `max_boost_factor` (default 3.0) caps the maximum brightness multiplier.

### Step 3 — Logarithmic Gain Calculation

The gain map encodes the ratio between synthetic HDR and original SDR luminance in log₂ space:

```python
gain_map_log2 = log2(stretched + epsilon) - log2(luminance + epsilon)
gain_map_uint8 = clip(gain_map_log2 * scale + offset, 0, 255)
```

Logarithmic encoding provides perceptual uniformity and matches how gain maps are represented in the Ultra HDR standard (8-bit unsigned, with gains centered around 1.0 at value 128).

### Step 4 — Edge-Aware Refinement

A guided filter (either OpenCV's `ximgproc.guidedFilter` or a pure-NumPy box-filter approximation) smooths the gain map using the original SDR luminance as the guidance image. This preserves structural edges (e.g. tree branches, building outlines) while smoothing flat regions, preventing halo artifacts.

### Step 5 — Aesthetic Bloom (Optional)

A mild Gaussian blur (sigma ~3–5 px) is blended into the gain map peaks to simulate natural light halation around bright light sources, adding a cinematic quality to HDR rendering.

## Performance Optimizations

- **Half-resolution gain map** — The image is downsampled to half resolution before gain map computation, reducing pixel work by 4×. This is the single largest performance win with minimal quality impact on typical consumer photos.
- **In-place NumPy operations** — Array operations avoid unnecessary allocations throughout the pipeline (e.g. `np.power(luma, gamma, out=luma)` pattern).
- **Integral-image box filter** — The guided filter fallback uses a padded integral image for O(1)-per-pixel window means rather than O(r²) naive convolution.
- **OpenCV acceleration** — When `cv2` and `cv2.ximgproc` are available they are used for guided filtering and Gaussian blur. Only `numpy` and `imagecodecs` are required as core dependencies.

Typical conversion time on a modern CPU: ~2–5 seconds for a 12MP image, ~0.5–1s for a 4MP image (half-resolution gain map, OpenCV acceleration enabled).

## Public API

The package exports a minimal surface for programmatic use:

- `convert_jpeg_to_ultrahdr()` — end-to-end conversion (main entry point).
- `ConversionResult` — dataclass with output path, ICC presence, gain map source.
- `GainMapConfig` — configuration for the highlight-targeted generator.
- `has_ultrahdr_metadata()` — detects if a file is already encoded with Ultra HDR.
- `has_mpf_secondary_image()` — detects embedded MPF auxiliary images (gain maps).
- `generate_gain_map()` — standalone gain map generation from luminance arrays.
- `validate_gain_map()` — type/shape validation for external gain maps.
- `linearize_from_icc()` — ICC-aware linearization for advanced users.
- `AlreadyUltraHDRError` — raised when input is already fully Ultra HDR encoded.

The pipeline also accepts an optional coarse-grained progress callback used by the CLI and GUI. Progress notifications are emitted only at major phase boundaries to avoid affecting the numeric hot path.

## Module Boundaries

- `errors.py`: custom exception hierarchy (`UltraHdrError` base and subclasses).
- `core/`
  - `jpeg_io.py`: byte I/O, JPEG decode, metadata extraction, gain map loading.
  - `color.py`: simple luminance channel extraction without ICC (grayscale helpers).
  - `color_cms.py`: ICC/CMS-based linearisation and CIE Y luminance extraction.
  - `gain_map.py`: gain map validation and highlight-targeted generation.
  - `ultrahdr_encoder.py`: Ultra HDR packaging and metadata embedding.
  - `converter.py`: orchestration, conversion result object, top-level workflow.
- `ui/`
  - `cli.py`: command line interface, backward-compatible single-file mode, and batch job resolution.
  - `gui.py`: optional desktop UI built on PySide6, with a background `QThread` worker.
  - `_gui_style.py`: dark theme palette and QSS stylesheet.
  - `assets/`: bundled static resources (application icon) shipped with the wheel.

## Data Contracts

| Data object | Format | Shape / Type |
|-------------|--------|-------------|
| SDR base input | uint8 ndarray | `(H, W, 3)` |
| Linearized SDR | float ndarray | `(H, W, C)` (typically `float32`) |
| Luminance (CIE Y) | float ndarray | `(H, W)` |
| Gain map | uint8 ndarray | `(H, W)` or `(H, W, 1|3)`; resolution independent of SDR base (typically half resolution when auto-generated) |
| Output | bytes | Ultra HDR JPEG encoded in MPF container |

## Error Strategy

- **All library errors derive from `UltraHdrError`** — giving callers a single base to catch.
- `GainMapError` and its subclasses (`GainMapDimensionError`, `GainMapConfigError`, `GainMapShapeMismatchError`) cover gain map validation failures with actionable messages.
- `AlreadyUltraHDRError` is raised early if the input image is already an Ultra HDR file, enabling batch tools to skip it gracefully.
- `ColorTransformError` covers ICC profile or CMS transform failures.
- `JpegStructureError` covers malformed or incomplete JPEG byte streams.
- **Inputs are validated early**; errors are raised with descriptive messages before any heavy computation begins.
- **Conversion functions are deterministic and side-effect free** except for file writes.
