# Ultra HDR Converter Architecture

## Goals

- Implement an end-to-end Ultra HDR packaging flow from an SDR image + gain map.
- Support multiple containers (JPEG, AVIF) behind one format-agnostic pipeline, so adding a format does not touch the conversion core.
- Preserve color intent through ICC profile handling for the SDR base.
- Preserve the compressed base image byte-for-byte whenever the output container matches the input.
- Keep the conversion steps modular for reproducibility and testing.
- Support both single-file and batch-oriented entry points without changing the core conversion path.

## Format Support

| Container | Detection | Decode | Gain map carriage |
|-----------|-----------|--------|-------------------|
| JPEG | `FF D8 FF` magic | `imagecodecs.jpeg_decode` | MPF secondary image + Adobe XMP (`hdrgm`) + ISO 21496-1 APP2 segment |
| AVIF | `ftyp` box with an AVIF-family brand | `imagecodecs.avif_decode` | `tmap` tone map derived image item (ISO/IEC 23008-12:2024 §6.6.2.4) + ISO 21496-1 metadata |

Input and output containers are independent: all four combinations are supported. The output container is chosen by explicit request, then the output file suffix, then the input container.

### Why the AVIF container is assembled in this repository

`imagecodecs` wraps libavif but exposes only plain single-image encode/decode — its `avif_encode` signature carries no gain map, ICC, or auxiliary-item parameters — and the `libultrahdr` it bundles is built JPEG-only (`uhdr_enc_set_output_format` rejects `UHDR_CODEC_AVIF`). The ISOBMFF muxing is therefore implemented in `core/avif_encoder.py`, mirroring the MPF surgery the JPEG path already performs. The box layout follows libavif's own encoder, and output is verified against libavif's reference tooling (`avifdec --info`, `avifgainmaputil printmetadata`).

## Pipeline Phases

### Phase A: Decode and Linearize via ICC

1. Read image bytes and detect the container from its magic bytes.
2. Check for existing gain map metadata (Ultra HDR/ISO 21496-1 segments for JPEG, a `tmap` item for AVIF). If present, skip processing by raising `AlreadyUltraHDRError`.
3. Decode the SDR raster through the container's decoder.
4. Extract the embedded `icc_profile` (JPEG APP2 chain, or the AVIF `colr`/`prof` item property).
5. Build CMS source and linear destination profiles with `imagecodecs.cms_profile`.
6. Convert to linear light with `imagecodecs.cms_transform` (default `float32`).

**Fallback behavior:** If no ICC exists, linearize with the sRGB assumption.

### Phase B: Encode Ultra HDR with Gain Map

1. Load a user-provided gain map (`.npy` or standard image), extract an embedded auxiliary gain map from the input file (JPEG MPF secondary image), or generate one from linear luminance.
2. When generating, the SDR image is downsampled to half resolution (stride-2 subsampling) before computing CIE Y luminance and the gain map, reducing pixel count by 4x.
3. Gain map generation uses highlight-targeted inverse tone mapping:
   - **Smoothstep mask**: isolates highlights above a configurable threshold.
   - **Exponential stretch**: synthesizes HDR luminance from compressed highlights.
   - **Log₂ ratio**: gain map = log₂(HDR / SDR luminance).
   - **Guided filter**: smooths the gain map using SDR luminance as guide to preserve edges.
   - **Gaussian bloom**: optional halation effect on gain map peaks.
4. Validate gain map channels and dtype (single-channel or RGB, uint8).
5. Build the ISO 21496-1 `GainMapMetadata` blob once, in `core/iso21496.py`, so both containers describe the same gain map with identical numbers.
6. Package into the target container. Both encoders accept gain maps of any resolution relative to the SDR base.

#### JPEG packaging (API-4)

The original SDR JPEG is reused verbatim. The gain map is JPEG-encoded, an ISO 21496-1 APP2 segment is injected into it, and it is appended as an MPF secondary image alongside an APP1 XMP segment and an APP2 ISO version segment on the primary image.

#### AVIF packaging (ISOBMFF muxing)

A single HEIF container is written with three items:

| Item | Type | Role |
|------|------|------|
| 1 | `av01` | SDR base image, the **primary item** (`pitm`) so gain-map-unaware viewers show it unchanged |
| 2 | `tmap` | Tone map derived item; its payload is the `ToneMapImage` version byte + ISO 21496-1 metadata |
| 3 | `av01` | Gain map image, marked as a **hidden** item |
| 4 | `av01` | Alpha auxiliary image (`auxl` → item 1), only when the input carried one |

`iref`/`dimg` links item 2 to `[1, 3]` in that order — libavif requires the base image first and the gain map second. Items 2 and 1 are placed in a `grpl`/`altr` entity group, and the `tmap` brand is added to `ftyp` as required by ISO/IEC 23008-12:2024/AMD 1.

Coded AV1 payloads and their descriptive properties (`ispe`, `pixi`, `av1C`, `colr`) are copied out of the source file, so an AVIF input is re-packaged without ever being decoded and re-encoded. Because `iloc` uses fixed-width offset fields, the `meta` box is laid out in two passes: once with placeholder offsets to measure it, then again with the real `mdat` offsets.

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

- `convert_to_ultrahdr()` — end-to-end conversion (main entry point).
- `ConversionResult` — dataclass with output path, ICC presence, gain map source, and the input/output containers.
- `ImageFormat` — supported container enum, with `default_suffix` and `suffixes`.
- `detect_format()` / `is_supported_path()` / `SUPPORTED_SUFFIXES` — container identification.
- `GainMapConfig` — configuration for the highlight-targeted generator.
- `GainMapMetadata` — ISO 21496-1 metadata builder shared by both encoders.
- `has_ultrahdr_metadata()` — detects if a file is already gain map encoded, in any supported container.
- `has_embedded_gain_map()` — detects auxiliary gain map images that lack metadata.
- `generate_gain_map()` — standalone gain map generation from luminance arrays.
- `validate_gain_map()` — type/shape validation for external gain maps.
- `linearize_from_icc()` — ICC-aware linearization for advanced users.
- `AlreadyUltraHDRError` — raised when input is already fully gain map encoded.

> **Breaking change in 0.2.0:** `convert_jpeg_to_ultrahdr()` is now `convert_to_ultrahdr()`, its `input_jpeg`/`output_jpeg`/`jpeg_quality` parameters are `input_path`/`output_path`/`quality`, and `has_mpf_secondary_image()` is now `has_embedded_gain_map()`.

The pipeline also accepts an optional coarse-grained progress callback used by the CLI and GUI. Progress notifications are emitted only at major phase boundaries to avoid affecting the numeric hot path.

## Module Boundaries

- `errors.py`: custom exception hierarchy (`UltraHdrError` base and subclasses). Container parsing failures share the `ImageStructureError` base, with `JpegStructureError` and `AvifStructureError` beneath it.
- `core/`
  - `formats.py`: `ImageFormat` enum, magic-byte detection, suffix maps, output format resolution. **The only module that enumerates supported containers.**
  - `image_io.py`: byte I/O, gain map loading, and the format dispatch layer. Callers use this instead of importing a container module directly.
  - `iso21496.py`: ISO 21496-1 `GainMapMetadata`, shared by every container.
  - `jpeg_io.py`: JPEG segment parsing, decode, encode, ICC and MPF extraction.
  - `jpeg_encoder.py`: Ultra HDR JPEG packaging (MPF + XMP + ISO metadata).
  - `avif_io.py`: ISOBMFF/HEIF box parsing — items, properties, `iloc` extents, ICC, `tmap` detection.
  - `avif_encoder.py`: ISOBMFF muxing of base + gain map + `tmap` into an AVIF.
  - `color.py`: simple luminance channel extraction without ICC (grayscale helpers).
  - `color_cms.py`: ICC/CMS-based linearisation and CIE Y luminance extraction.
  - `gain_map.py`: gain map validation and highlight-targeted generation.
  - `converter.py`: orchestration, conversion result object, top-level workflow.

### Adding a container format

1. Add an `ImageFormat` member with its suffixes and magic bytes in `formats.py`.
2. Add `<format>_io.py` (detect/decode/ICC/gain map extraction) and `<format>_encoder.py` (packaging).
3. Wire both into the dispatch tables in `image_io.py` and `_encode_output()` in `converter.py`.

Nothing else changes: the CLI, GUI, gain map generator and colour pipeline are all format-agnostic.
- `ui/`
  - `cli.py`: command line interface, backward-compatible single-file mode, and batch job resolution.
  - `gui.py`: optional desktop UI built on PySide6, with a background `QThread` worker.
  - `_gui_style.py`: dark theme palette and QSS stylesheet.
  - `assets/`: bundled static resources (application icon) shipped with the wheel.

## Data Contracts

| Data object | Format | Shape / Type |
|-------------|--------|-------------|
| SDR base input | uint8 ndarray | `(H, W, 3)`, or `(H, W, 4)` for AVIF with alpha |
| Linearized SDR | float ndarray | `(H, W, C)` (typically `float32`) |
| Luminance (CIE Y) | float ndarray | `(H, W)`; any alpha channel is dropped first |
| Gain map | uint8 ndarray | `(H, W)` or `(H, W, 1|3)`; resolution independent of SDR base (typically half resolution when auto-generated) |
| Coded image (AVIF) | `CodedImage` | AV1 payload plus `ispe`/`pixi`/`av1C`/`colr` properties, reused without re-encoding |
| Gain map metadata | `GainMapMetadata` | 61-byte ISO 21496-1 blob; 62 bytes as an AVIF `tmap` payload |
| Output | bytes | Ultra HDR JPEG (MPF container) or AVIF (`tmap` derived item) |

## Error Strategy

- **All library errors derive from `UltraHdrError`** — giving callers a single base to catch.
- `GainMapError` and its subclasses (`GainMapDimensionError`, `GainMapConfigError`, `GainMapShapeMismatchError`) cover gain map validation failures with actionable messages.
- `AlreadyUltraHDRError` is raised early if the input image is already gain map encoded, enabling batch tools to skip it gracefully.
- `ColorTransformError` covers ICC profile or CMS transform failures.
- `ImageStructureError` covers malformed containers, with `JpegStructureError` and `AvifStructureError` for per-format detail. Catching the base handles any container.
- `UnsupportedFormatError` covers unrecognised or unsupported containers, including HEIF-family files that are not AVIF.
- **Inputs are validated early**; errors are raised with descriptive messages before any heavy computation begins.
- **Conversion functions are deterministic and side-effect free** except for file writes.
