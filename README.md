# Ultra HDR Converter

Python toolkit and CLI to package Ultra HDR JPEG files from:

- an SDR base JPEG
- an external gain map (or an automatically generated one)

Dependency and build management use `uv`.

## Architecture

The codebase follows a two-phase workflow:

- **Phase A** -- Decode and linearize the SDR JPEG through its embedded ICC profile using `imagecodecs.cms_*`.
- **Phase B** -- Generate a highlight-targeted gain map (at half resolution for speed) and compose a standards-aligned Ultra HDR JPEG via API-4 (MPF container with XMP and ISO 21496-1 metadata).

Detailed design is in [docs/architecture.md](docs/architecture.md).

## Repository Layout

```text
.
|- .github/workflows/ci.yml
|- docs/architecture.md
|- examples/convert_with_custom_gainmap.py
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
uv run uhdr-convert input.jpg output_ultrahdr.jpg
```

## Programmatic Usage

```python
from ultra_hdr_converter import convert_jpeg_to_ultrahdr, GainMapConfig

# Auto-generate gain map with custom settings.
result = convert_jpeg_to_ultrahdr(
    input_jpeg="input.jpg",
    output_jpeg="output_ultrahdr.jpg",
    gain_map_config=GainMapConfig(highlight_threshold=0.4, max_boost_factor=6.0),
)

# Or use an external gain map.
result = convert_jpeg_to_ultrahdr(
    input_jpeg="input.jpg",
    output_jpeg="output_ultrahdr.jpg",
    gain_map_path="gain_map.png",
)
```

## Gain Map Algorithm

The built-in generator uses a highlight-targeted inverse tone mapping approach:

1. **Soft Highlight Isolation** -- A smoothstep function creates a soft mask that isolates compressed highlights (skies, light sources, reflections) while leaving midtones and shadows untouched.
2. **Non-Linear Highlight Expansion** -- An exponential curve stretches the masked highlights to synthesize an HDR luminance signal.
3. **Logarithmic Gain Calculation** -- The gain map is computed as the log2 ratio between the synthetic HDR and original SDR luminance.
4. **Edge-Aware Refinement** -- A guided filter smooths the gain map using the SDR luminance as guide, ensuring HDR boost respects structural edges and avoids halos.
5. **Aesthetic Bloom** -- A subtle Gaussian blur is blended into the gain map peaks to simulate natural light halation.

### CLI Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--highlight-threshold` | `0.5` | Linear luminance value where HDR boost begins |
| `--expansion-gamma` | `2.2` | Exponent for non-linear highlight stretch |
| `--max-boost-factor` | `4.0` | Maximum HDR multiplier for brightest pixels |
| `--guided-radius` | `20` | Guided filter radius for edge-aware smoothing |
| `--guided-eps` | `0.001` | Guided filter epsilon |
| `--bloom-weight` | `0.15` | Weight of bloom effect (0 to disable) |

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

- **Half-resolution gain map**: The pipeline downsamples the SDR image to half resolution before computing luminance and generating the gain map, reducing pixel count by 4x.
- **In-place NumPy operations**: Gain map generation uses in-place array operations to minimize memory allocations.
- **Integral-image box filter**: The guided filter uses a padded integral image for O(1)-per-pixel window means.
- **OpenCV acceleration**: When `cv2` and `cv2.ximgproc` are available, the pipeline uses them for guided filtering and Gaussian blur. Only `numpy` + `imagecodecs` are required as core dependencies.

## Notes

- Use the full `imagecodecs` build with both `cms` and `ultrahdr` extensions enabled.
