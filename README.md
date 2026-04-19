# Ultra HDR Converter

Python toolkit and CLI to package Ultra HDR JPEG files from:

- an SDR base JPEG
- an external gain map (or an automatically generated one)
- optional ICC metadata passthrough

Dependency and build management use `uv`.

## Architecture

The codebase follows the implementation workflow in [starting-point.md](starting-point.md):

- Phase A: Decode and linearize the SDR JPEG through embedded ICC profile using `imagecodecs.cms_*`.
- Phase B: Encode a standards-aligned Ultra HDR JPEG via `imagecodecs.ultrahdr_encode` using SDR base + gain map.

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
|  |- test_gainmap.py
|  `- test_pipeline.py
|- pyproject.toml
`- starting-point.md
```

## Quick Start

```powershell
uv sync --group dev
uv run uhdr-convert input.jpg output_ultrahdr.jpg --gain-map gain_map.png
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

## Notes

- Use the full `imagecodecs` build with both `cms` and `ultrahdr` extensions enabled.
- For math on linearized data, keep `float32` output (`--linear-dtype float32` in CLI).
