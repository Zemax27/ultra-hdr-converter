# Contributing to Ultra HDR Converter

Thank you for your interest in improving Ultra HDR Converter! This document provides guidelines for developers, reviewers, and maintainers.

## Quick Start

### Prerequisites

- **Python 3.12+** (tested on 3.12 and 3.13)
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** — fast Python package manager

### Install Development Dependencies

```bash
# Core library + CLI + GUI + all dev tools (ruff, pytest, mypy, nuitka)
uv sync --group dev --extra cli --extra gui
```

This installs:
- Testing: `pytest`, `pytest-cov`
- Linting: `ruff`
- Type checking: `mypy`
- Build tools: `nuitka` (for standalone executable bundling)
- Runtime: CLI and GUI dependencies

### Verify Setup

```bash
# Lint code
uv run ruff check .

# Type check
uv run mypy src

# Run tests with coverage
uv run pytest --cov

# Build to verify packaging
uv build
```

## Development Workflow

### Code Quality Standards

This project follows the guidelines defined in [.github/copilot-instructions.md](.github/copilot-instructions.md):

- **Type coverage:** 100% — annotate all functions and public members
- **Style:** ruff-compatible PEP 8, line length 120, double quotes, f-strings
- **Docstrings:** Google-style for public APIs (do not repeat type hints)
- **Error handling:** Validate inputs early; raise specific exceptions
- **Performance:** Avoid quadratic loops; prefer vectorized/library operations

### Testing Philosophy

- **Unit tests** mock external I/O (filesystem, imagecodecs library)
- **Integration tests** use real JPEG files when possible
- **Happy path + edge cases + failure cases** must be covered
- **Update tests** whenever you change behavior

All tests live in `tests/` and use `pytest`. Run the full suite locally before pushing:

```bash
uv run pytest -v
```

### Making Changes

1. **Create a branch** (preferably from `develop` or `main`)
2. **Make focused commits** — one logical change per commit
3. **Write tests** for new functionality or bug fixes
4. **Update documentation** — both user-facing (README) and technical (docs/)
5. **Run all checks** locally: `ruff`, `mypy`, `pytest`
6. **Push and open a PR** — CI will run automatically

See [.github/workflows/ci.yml](.github/workflows/ci.yml) for the exact CI checks.

### Commit Messages

Follow conventional commits style:

```
feat: add support for WebP input format
fix: handle malformed JPEG without crashing
docs: clarify tuning tips for bright skies
test: add regression test for gain_map zero division
refactor: extract gain map validation logic
```

This generates clean changelogs and helps reviewers understand your intent.

## Architecture

The codebase follows a two-phase pipeline:

- **Phase A — Decode & Linearize:** Read JPEG, decode raster, extract ICC profile, convert to linear light via CMS.
- **Phase B — Encode Ultra HDR:** Generate or load gain map, then compose MPF container with XMP and ISO 21496-1 metadata.

Detailed design documentation is in [docs/architecture.md](docs/architecture.md):

- Pipeline phases and data contracts
- Public API surface
- Module boundaries and responsibilities
- Gain map generation algorithm
- Error handling strategy

Read [docs/architecture.md](docs/architecture.md) before making significant architectural changes.

## Adding New Features

### Before You Start

1. **Search existing issues** — someone may have requested the same feature
2. **Discuss approach** — open an issue or comment on an existing one
3. **Keep scope minimal** — avoid scope creep; focused fixes are easier to review

### Implementation Guidelines

- **Do not break backward compatibility** without discussion
- **Preserve existing public API** — add new functions rather than modifying signatures
- **Update `__init__.py`** if adding public exports
- **Update type stubs** if necessary (though type hints should be inline)
- **Document configuration options** in both code and README if user-facing

### Performance Considerations

Conversion speed matters for batch workflows. If your change affects performance:

1. **Add benchmarks** (future work: integrate `pytest-benchmark`)
2. **Document performance impact** in your PR description
3. **Avoid unnecessary memory allocations** — use in-place NumPy ops where safe
4. **Profile before optimizing** — don't guess

See [docs/architecture.md](docs/architecture.md#performance-optimizations) for current optimization strategies.

## Building & Distribution

### Build a Python Wheel

```bash
uv build
```

Outputs to `dist/` as both `.whl` and `.tar.gz`.

### Build Standalone Executable (GUI)

For local testing or distribution:

```bash
uv run python -m nuitka --standalone \
    --output-filename=uhdr-gui \
    --windows-console-mode=disable \
    --macos-create-app-bundle \
    --macos-app-mode=gui \
    --enable-plugin=pyside6 \
    --include-data-dir="src/ultra_hdr_converter/ui/assets=ultra_hdr_converter/ui/assets" \
    --include-package=imagecodecs \
    src/ultra_hdr_converter/ui/gui.py
```

See `pyproject.toml` for Nuitka options. The release workflow automates this with cross-platform builds.

## CI/CD

Pull requests and pushes to `main`/`develop` trigger GitHub Actions:

- **Lint** (`ruff check .`)
- **Type check** (`mypy src`)
- **Test** (`pytest -q` across Python 3.12/3.13 matrix)
- **Build** (`uv build` — ensure wheel is valid)

All checks must pass before merging. Fix any failures promptly.

## Release Process

Releases are automated via [.github/workflows/release.yml](.github/workflows/release.yml). To create a new release:

1. Update version in `pyproject.toml` (follow [SemVer](https://semver.org/))
2. Update `CHANGELOG.md` (create if missing)
3. Push a tag: `git tag v0.2.0 && git push origin v0.2.0`
4. GitHub Actions builds wheels, creates release notes, and attaches assets

## Reporting Issues

- **Bugs:** Use GitHub Issues — include steps to reproduce, Python version, OS, error messages
- **Feature requests:** Use GitHub Issues — describe the use case and expected behavior
- **Questions:** Use GitHub Discussions (if enabled) or Stack Overflow with tag `ultra-hdr-converter`

Before reporting a bug:
1. Check if it's already reported
2. Try the latest version (`git pull` + reinstall)
3. Include a minimal test case if possible

## Attribution

This project is based on the ISO 21496-1 standard and Adobe Ultra HDR specifications. See [LICENSE](LICENSE) for licensing details.

---

**Thank you for contributing!** Your work helps make HDR photography accessible to everyone.
