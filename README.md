# Ultra HDR Converter

Convert standard JPEG photos into **Ultra HDR JPEG** files that display with expanded brightness and vivid highlights on HDR-capable screens (iPhone, iPad, modern Android, HDR monitors), while remaining fully compatible with all existing SDR devices.

The tool works from a single SDR JPEG — no HDR camera required. It automatically synthesises a gain map that encodes the highlight information needed for HDR playback.

## Requirements

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** — fast Python package manager

## Installation

```powershell
git clone https://github.com/your-org/ultra-hdr-converter.git
cd ultra-hdr-converter
uv sync                     # core library only (programmatic API)
uv sync --extra cli         # + CLI (uhdr-convert)
uv sync --extra cli --extra gui  # + CLI and desktop GUI (uhdr-gui)
```

> **Note:** `imagecodecs` must be built with the `cms` and `ultrahdr` extensions enabled. The default PyPI wheel includes both.

## Usage

### Convert a single photo

```powershell
uv run uhdr-convert input.jpg output_ultrahdr.jpg
```

The output file is a valid JPEG readable everywhere. On HDR displays it lights up; on SDR screens it looks identical to the original.

If you omit the output path, the file is saved next to the input as `<name>_ultrahdr.jpg`:

```powershell
uv run uhdr-convert input.jpg
```

### Convert a whole folder

```powershell
uv run uhdr-convert --batch-inputs photos\ --out-dir converted\
```

You can also pass individual files in batch mode:

```powershell
uv run uhdr-convert --batch-inputs photo1.jpg photo2.jpg photo3.jpg --out-dir converted\
```

Each output is saved as `<original_name>_ultrahdr.jpg` inside `--out-dir`. If `--out-dir` is omitted the converted files are written beside the originals.

### Use an external gain map

If you have a pre-computed gain map (`.npy` or any image file) you can supply it directly and skip auto-generation:

```powershell
uv run uhdr-convert input.jpg output_ultrahdr.jpg --gain-map gain_map.png
```

### Desktop GUI

A graphical interface is available for users who prefer not to use the command line.

```powershell
uv sync --extra cli --extra gui
uv run uhdr-gui
```

The GUI runs conversions on a background thread so the interface stays responsive, and shows live progress for each pipeline phase.

## CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `input_jpeg` | — | Input SDR JPEG (single-file mode) |
| `output_jpeg` | `<input>_ultrahdr.jpg` | Output Ultra HDR JPEG (single-file mode) |
| `--batch-inputs FILE/DIR …` | — | One or more JPEG files or directories (batch mode) |
| `--out-dir DIR` | beside input | Output folder for batch mode (or single-file when omitted) |
| `--gain-map FILE` | auto-generated | External gain map file (`.npy` or image). Skips gain map synthesis. |
| `--highlight-threshold` | `0.5` | Luminance level (0–1) at which HDR boost begins. Lower = more pixels boosted. |
| `--expansion-gamma` | `2.2` | Curve exponent for highlight stretching. Higher = more aggressive expansion. |
| `--max-boost-factor` | `3.0` | Maximum HDR brightness multiplier for the brightest pixels. |
| `--jpeg-quality` | `95` | JPEG quality level for the gain map (0-100). |
| `--guided-radius` | `20` | Edge-aware smoothing radius. Larger = smoother gain map, slower. |
| `--guided-eps` | `0.001` | Edge sensitivity for guided filter. Smaller = sharper edges preserved. |
| `--bloom-weight` | `0.15` | Bloom halo intensity around bright areas. Set to `0` to disable. |

### Tuning tips

- **Skies and windows look flat** → lower `--highlight-threshold` (e.g. `0.35`) and raise `--max-boost-factor` (e.g. `6.0`).
- **Halos around edges** → raise `--guided-eps` (e.g. `0.01`) or reduce `--bloom-weight`.
- **Effect is too subtle** → raise `--max-boost-factor` and lower `--highlight-threshold`.
- **Effect is too aggressive** → raise `--highlight-threshold` closer to `0.7`.

## Programmatic Usage

```python
from ultra_hdr_converter import convert_jpeg_to_ultrahdr, GainMapConfig

# Auto-generate gain map with custom tuning.
result = convert_jpeg_to_ultrahdr(
    input_jpeg="input.jpg",
    output_jpeg="output_ultrahdr.jpg",
    gain_map_config=GainMapConfig(highlight_threshold=0.4, max_boost_factor=6.0),
)

# Use an external gain map.
result = convert_jpeg_to_ultrahdr(
    input_jpeg="input.jpg",
    output_jpeg="output_ultrahdr.jpg",
    gain_map_path="gain_map.png",
)

# result.output_path   — Path to the written file
# result.has_icc       — True if an ICC colour profile was embedded
# result.gain_map_source — "generated" or "external"
```

## How the Gain Map Works

The built-in generator uses a highlight-targeted inverse tone-mapping approach:

1. **Soft Highlight Isolation** — A smoothstep mask isolates compressed highlights (skies, light sources, reflections) while leaving midtones and shadows untouched.
2. **Non-Linear Highlight Expansion** — An exponential curve stretches masked highlights to synthesise an HDR luminance signal.
3. **Logarithmic Gain Calculation** — The gain map is computed as the log₂ ratio between the synthetic HDR and original SDR luminance.
4. **Edge-Aware Refinement** — A guided filter smooths the gain map using the SDR luminance as a guide, preventing halos along structural edges.
5. **Aesthetic Bloom** — A subtle Gaussian blur blended into gain map peaks simulates natural light halation.

## Performance

- **Half-resolution gain map** — The image is downsampled to half resolution before gain map computation, reducing pixel work by 4×.
- **In-place NumPy operations** — Array operations avoid unnecessary allocations throughout the pipeline.
- **Integral-image box filter** — The guided filter uses a padded integral image for O(1)-per-pixel window means.
- **OpenCV acceleration** — When `cv2` and `cv2.ximgproc` are available they are used for guided filtering and Gaussian blur. Only `numpy` and `imagecodecs` are required as core dependencies.

## Development

```powershell
# Install all dev dependencies (includes PyInstaller for bundling)
uv sync --group dev --extra cli --extra gui

# Lint
uv run ruff check .

# Type check
uv run mypy src

# Tests
uv run pytest

# Build distributable wheel
uv build

# Build standalone GUI bundle (local test)
uv run pyinstaller --onedir --name uhdr-gui --windowed `
    --add-data "src/ultra_hdr_converter/ui/assets:ultra_hdr_converter/ui/assets" `
    --hidden-import ultra_hdr_converter.ui.assets `
    src/ultra_hdr_converter/ui/gui.py
```

## Architecture

The codebase follows a two-phase workflow:

- **Phase A** — Decode and linearise the SDR JPEG through its embedded ICC profile using `imagecodecs.cms_*`.
- **Phase B** — Generate a highlight-targeted gain map (at half resolution) and compose a standards-aligned Ultra HDR JPEG via API-4 (MPF container with XMP and ISO 21496-1 metadata).

Full design details are in [docs/architecture.md](docs/architecture.md).

```text
src/ultra_hdr_converter/
├── __init__.py               — Public API exports
├── errors.py                 — Custom exception hierarchy
├── core/                     — Image processing pipeline
│   ├── converter.py          — Orchestration and ConversionResult
│   ├── gain_map.py           — Gain map generation and validation
│   ├── color.py              — Simple luminance channel extraction
│   ├── color_cms.py          — ICC-aware linearisation and CIE Y extraction
│   ├── ultrahdr_encoder.py   — Ultra HDR JPEG encoding (MPF + XMP + ISO)
│   └── jpeg_io.py            — JPEG decode/encode and file I/O
└── ui/                       — User interfaces
    ├── cli.py                — Command line interface (uhdr-convert)
    ├── gui.py                — Optional desktop GUI (uhdr-gui, requires PySide6)
    ├── _gui_style.py         — GUI dark theme stylesheet
    └── assets/
        └── icon.png          — Bundled application icon
```

## License

This repository is licensed under the Apache License 2.0.

- Full text: [LICENSE](LICENSE)
- Dependency notices: [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt)

Third-party libraries keep their own licenses. If you redistribute binaries
(for example standalone executables), include applicable third-party notices
and license files in your release assets.
