# Ultra HDR Converter

Convert standard JPEG photos into **Ultra HDR JPEG** files compliant with **ISO 21496-1**. These images display expanded brightness and vivid highlights on HDR-capable screens (iPhone 12 or newer, recent Android, HDR monitors) while remaining fully compatible with SDR devices. No HDR camera needed — the tool automatically synthesizes a gain map from a single SDR JPEG.

## Requirements

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** — fast Python package manager

For standalone GUI releases (no Python required), see [Releases](https://github.com/Zemax27/ultra-hdr-converter/releases).

## Installation

```powershell
git clone https://github.com/Zemax27/ultra-hdr-converter.git
cd ultra-hdr-converter
uv sync                     # core library only (programmatic API)
uv sync --extra cli         # + CLI tool (uhdr-convert)
uv sync --extra cli --extra gui  # + CLI and desktop GUI (uhdr-gui)
```

> **Note:** `imagecodecs` must be built with the `cms` and `ultrahdr` extensions enabled. The default PyPI wheel includes both.

## Usage

### Convert a single photo

```powershell
uv run uhdr-convert input.jpg output_ultrahdr.jpg
```

The output file is a valid JPEG readable everywhere. On HDR displays it lights up; on SDR screens it looks identical to the original. If you omit the output path, the file is saved beside the input as `<name>_ultrahdr.jpg`:

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

### Use an external or embedded gain map

If you have a pre-computed gain map (`.npy` or any image file) you can supply it directly and skip auto-generation:

```powershell
uv run uhdr-convert input.jpg output_ultrahdr.jpg --gain-map gain_map.png
```

Alternatively, if your input file already contains an embedded gain map in Multi-Picture Format (MPF) but lacks the required ISO 21496-1 or Adobe XMP metadata, the pipeline will automatically extract and use that embedded map to encode the final Ultra HDR image.

> **Note**: If you run `uhdr-convert` on a file that is *already* fully encoded as Ultra HDR or ISO 21496-1, the tool will inform you and gracefully skip the file.

### Desktop GUI

A graphical interface is available for users who prefer not to use the command line.

```powershell
uv sync --extra cli --extra gui
uv run uhdr-gui
```

The GUI runs conversions on a background thread so the interface stays responsive, and shows live progress for each pipeline phase. Expand **HDR Tuning** to adjust highlight threshold, expansion gamma, maximum boost, and bloom weight for the whole batch. Each control includes an effect hint, and the documented defaults preserve the standard conversion behavior. These controls affect synthesized gain maps; an existing embedded gain map is reused without regeneration.

For a polished desktop experience, download the standalone executable from the [Releases](https://github.com/Zemax27/ultra-hdr-converter/releases) page.

> **macOS note:** The current app bundle is ad-hoc signed rather than Apple-notarized. On first launch, trusted users must approve it in **System Settings → Privacy & Security → Open Anyway**. See [CONTRIBUTING.md](CONTRIBUTING.md#release-process) for details.

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

## Examples

See the `examples/` directory for programmatic usage and custom workflows:

- [convert_with_custom_gainmap.py](examples/convert_with_custom_gainmap.py) — Using the Python API directly

## Troubleshooting

**"imagecodecs cms extension is unavailable"**
The installed `imagecodecs` package lacks CMS/UltraHDR support. This typically happens when a source build was attempted without required C libraries, or an incomplete wheel was installed.

Fix by reinstalling the official PyPI wheel (includes both extensions):

```bash
# Using uv (recommended)
uv sync --reinstall-package imagecodecs

# Or explicitly via uv pip
uv pip install --upgrade --force-reinstall imagecodecs
```

If building from source is unavoidable (e.g., no wheel available for your platform), install system dependencies first. See the [imagecodecs documentation](https://github.com/cgohlke/imagecodecs/#using) for details. On Ubuntu/Debian:

```bash
sudo apt install build-essential python3-dev cython3 python3-pip \
  python3-setuptools python3-wheel python3-numpy libdeflate-dev libjpeg-dev \
  libjxr-dev liblcms2-dev liblz4-dev liblerc-dev liblzma-dev \
  libopenjp2-7-dev libpng-dev libtiff-dev libwebp-dev libz-dev libzstd-dev
```

**"File … is already an Ultra HDR image"**
This is expected behavior when re-processing converted files. The tool skips these to avoid redundant work. Use a different output filename if you want to force a new conversion.

**GUI won't start / missing PySide6**
Ensure you installed with GUI extras: `uv sync --extra cli --extra gui`. If running a standalone executable, download the latest release from the [Releases](https://github.com/your-org/ultra-hdr-converter/releases) page.

## License

This repository is licensed under the Apache License 2.0.

- Full text: [LICENSE](LICENSE)
- Dependency notices: [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt)

Third-party libraries keep their own licenses. If you redistribute binaries (for example standalone executables), include applicable third-party notices and license files in your release assets.
