# Ultra HDR Converter

Python toolkit and CLI to package Ultra HDR JPEG files from:

- an SDR base JPEG
- an external gain map (or an automatically generated one)
- optional ICC metadata passthrough

Dependency and build management use `uv`.

## Architecture

The codebase follows a two-phase workflow:

- **Phase A** -- Decode and linearize the SDR JPEG through its embedded ICC profile using `imagecodecs.cms_*`.
- **Phase B** -- Generate a gain map (at half resolution for speed) and encode a standards-aligned Ultra HDR JPEG via API-4 composition (MPF container with XMP and ISO 21496-1 metadata).

Detailed design is in [docs/architecture.md](docs/architecture.md).

## Repository Layout

```text
.
|- .github/workflows/ci.yml
|- docs/architecture.md
|- examples/convert_with_custom_gainmap.py
|- src/gainmap_generator.py
|- src/ultra_hdr_converter/
|  |- __init__.py
|  |- cli.py
|  |- color.py
|  |- encoder.py
|  |- gainmap.py
|  |- io.py
|  `- pipeline.py
|- tests/
|  |- test_color.py
|  |- test_encoder.py
|  |- test_gainmap.py
|  |- test_io.py
|  `- test_pipeline.py
|- pyproject.toml
```

## Quick Start

```powershell
uv sync --group dev
uv run uhdr-convert input.jpg output_ultrahdr.jpg --gain-map gain_map.png
uv run uhdr-convert input.jpg output_ultrahdr.jpg --generated-gain-map radiance
```

## Commands

```powershell
# Lint
uv run ruff check .

# Type check
uv run mypy src

# Tests
uv run pytest

# Build package
uv build
```

## Performance

- **Half-resolution gain map**: The pipeline downsamples the SDR image to half resolution before computing luminance and generating the gain map. This reduces pixel count by 4x through the CMS transform and gain map algorithms, while the Ultra HDR encoder accepts gain maps of any size.
- **In-place NumPy operations**: Gain map generation (`log2`, `radiance`) and normalization use in-place array operations (`np.maximum(..., out=...)`, `np.log(..., out=...)`, `*=`, `/=`) to minimize memory allocations.
- **Integral-image box filter**: The guided filter uses a padded integral image for O(1)-per-pixel window means, with the mean computed directly to avoid separate division passes.
- **OpenCV acceleration**: When `cv2` and `cv2.ximgproc` are available, the pipeline uses them for bilinear resize and guided filtering. Only `numpy` + `imagecodecs` are required as core dependencies.

## Notes

- Use the full `imagecodecs` build with both `cms` and `ultrahdr` extensions enabled.
- For math on linearized data, keep `float32` output (`--linear-dtype float32` in CLI).
- When `--gain-map` is not provided, the default generator is `radiance`.
