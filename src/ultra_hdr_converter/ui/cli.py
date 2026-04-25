"""Command line interface for Ultra HDR conversion."""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

if TYPE_CHECKING:
    from rich.console import Console
    from rich.progress import Progress, TaskID

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskID,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

from ultra_hdr_converter.core.converter import ConversionResult, convert_jpeg_to_ultrahdr
from ultra_hdr_converter.core.gain_map import GainMapConfig
from ultra_hdr_converter.errors import AlreadyUltraHDRError

JPEG_SUFFIXES = {".jpg", ".jpeg"}
DEFAULT_OUTPUT_SUFFIX = "_ultrahdr.jpg"


@dataclass(frozen=True)
class ConversionJob:
    """Resolved input/output pair for one conversion run."""

    input_path: Path
    output_path: Path


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="uhdr-convert",
        description="Convert SDR JPEG files to Ultra HDR JPEG.",
    )
    parser.add_argument(
        "input_jpeg",
        nargs="?",
        type=Path,
        help="Input SDR JPEG file for single-file mode.",
    )
    parser.add_argument(
        "output_jpeg",
        nargs="?",
        type=Path,
        help="Output Ultra HDR JPEG file for single-file mode.",
    )
    parser.add_argument(
        "--batch-inputs",
        nargs="+",
        type=Path,
        default=None,
        help="One or more SDR JPEG files or directories for batch mode.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for batch mode, or default destination for single-file mode.",
    )
    parser.add_argument(
        "--gain-map",
        type=Path,
        default=None,
        help="Optional gain map path (.npy or image). If omitted, a gain map is generated automatically.",
    )
    parser.add_argument(
        "--highlight-threshold",
        type=float,
        default=0.5,
        help="Linear luminance value where HDR boost begins (0.0-1.0).",
    )
    parser.add_argument(
        "--expansion-gamma",
        type=float,
        default=2.2,
        help="Exponent for non-linear highlight stretch.",
    )
    parser.add_argument(
        "--max-boost-factor",
        type=float,
        default=3.0,
        help="Maximum HDR multiplier (in stops) for the brightest pixels.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality level for the gain map (0-100).",
    )
    parser.add_argument(
        "--guided-radius",
        type=int,
        default=20,
        help="Guided filter radius for edge-aware smoothing.",
    )
    parser.add_argument(
        "--guided-eps",
        type=float,
        default=1e-3,
        help="Guided filter epsilon.",
    )
    parser.add_argument(
        "--bloom-weight",
        type=float,
        default=0.15,
        help="Weight of aesthetic bloom effect (0 to disable).",
    )
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    return _build_parser().parse_args(argv)


def _is_jpeg_path(path: Path) -> bool:
    """Return True when the path looks like a supported JPEG file."""
    return path.suffix.lower() in JPEG_SUFFIXES


def _default_output_path(input_path: Path, out_dir: Path | None) -> Path:
    """Build the default output path for one input file."""
    filename = f"{input_path.stem}{DEFAULT_OUTPUT_SUFFIX}"
    if out_dir is not None:
        return out_dir / filename
    return input_path.with_name(filename)


def _ensure_gain_map_exists(parser: argparse.ArgumentParser, gain_map: Path | None) -> None:
    """Validate the optional external gain map path."""
    if gain_map is not None and not gain_map.is_file():
        parser.error(f"gain map file not found: {gain_map}")


def _ensure_single_input_exists(parser: argparse.ArgumentParser, input_path: Path) -> None:
    """Validate the single-file input path."""
    if not input_path.is_file():
        parser.error(f"input file not found: {input_path}")
    if not _is_jpeg_path(input_path):
        parser.error(f"single-file mode requires a .jpg or .jpeg input: {input_path}")


def _collect_directory_jpegs(directory: Path) -> list[Path]:
    """Collect top-level JPEG files from a directory in deterministic order."""
    return sorted(path for path in directory.iterdir() if path.is_file() and _is_jpeg_path(path))


def _collect_batch_input_paths(parser: argparse.ArgumentParser, raw_input: Path) -> list[Path]:
    """Resolve one batch input path into concrete JPEG files."""
    if raw_input.is_file():
        if not _is_jpeg_path(raw_input):
            parser.error(f"batch input must be a .jpg or .jpeg file: {raw_input}")
        return [raw_input]

    if raw_input.is_dir():
        directory_jpegs = _collect_directory_jpegs(raw_input)
        if not directory_jpegs:
            parser.error(f"directory contains no .jpg or .jpeg files: {raw_input}")
        return directory_jpegs

    parser.error(f"batch input not found: {raw_input}")
    raise AssertionError("unreachable")


def _build_single_job(parser: argparse.ArgumentParser, args: argparse.Namespace) -> list[ConversionJob]:
    """Resolve single-file mode into one conversion job."""
    input_path = args.input_jpeg
    if input_path is None:
        parser.error("single-file mode requires an input JPEG")
    if args.batch_inputs is not None:
        parser.error("use either positional single-file arguments or --batch-inputs, not both")

    _ensure_single_input_exists(parser, input_path)

    if args.output_jpeg is not None and args.out_dir is not None:
        parser.error("single-file mode cannot combine an explicit output file with --out-dir")

    output_path = args.output_jpeg or _default_output_path(input_path, args.out_dir)
    return [ConversionJob(input_path=input_path, output_path=output_path)]


def _build_batch_jobs(parser: argparse.ArgumentParser, args: argparse.Namespace) -> list[ConversionJob]:
    """Resolve batch mode into one job per discovered input JPEG."""
    raw_inputs = args.batch_inputs
    if raw_inputs is None:
        parser.error("batch mode requires at least one value for --batch-inputs")
    if args.input_jpeg is not None or args.output_jpeg is not None:
        parser.error("batch mode cannot be combined with positional single-file arguments")

    out_dir = args.out_dir
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    collected_inputs: list[Path] = []
    seen_inputs: set[Path] = set()

    for raw_input in raw_inputs:
        for input_path in _collect_batch_input_paths(parser, raw_input):
            resolved = input_path.resolve()
            if resolved not in seen_inputs:
                collected_inputs.append(input_path)
                seen_inputs.add(resolved)

    jobs = [
        ConversionJob(
            input_path=input_path,
            output_path=_default_output_path(input_path, out_dir),
        )
        for input_path in collected_inputs
    ]

    seen_outputs: set[Path] = set()
    for job in jobs:
        resolved_output = job.output_path.resolve(strict=False)
        if resolved_output in seen_outputs:
            parser.error(f"batch output collision detected for: {job.output_path}")
        seen_outputs.add(resolved_output)

    return jobs


def _build_jobs(parser: argparse.ArgumentParser, args: argparse.Namespace) -> list[ConversionJob]:
    """Resolve the CLI request into concrete conversion jobs."""
    _ensure_gain_map_exists(parser, args.gain_map)

    if args.batch_inputs is not None:
        return _build_batch_jobs(parser, args)
    if args.input_jpeg is not None:
        return _build_single_job(parser, args)
    parser.error("provide either positional single-file arguments or --batch-inputs")
    raise AssertionError("unreachable")


def _build_gain_map_config(args: argparse.Namespace) -> GainMapConfig:
    """Construct gain map configuration from parsed arguments."""
    return GainMapConfig(
        highlight_threshold=args.highlight_threshold,
        expansion_gamma=args.expansion_gamma,
        max_boost_factor=args.max_boost_factor,
        guided_radius=args.guided_radius,
        guided_eps=args.guided_eps,
        bloom_weight=args.bloom_weight,
    )


def _make_progress_callback(
    progress: Progress,
    task_id: TaskID,
    input_name: str,
) -> Callable[[str, int, int], None]:
    """Build a progress callback that updates the active file task."""

    def _callback(message: str, step: int, total_steps: int) -> None:
        progress.update(
            task_id,
            total=total_steps,
            completed=step,
            description=f"[green]{input_name}[/green] [dim]{message}[/dim]",
        )

    return _callback


def _run_jobs(
    console: Console,
    jobs: Sequence[ConversionJob],
    gain_map_path: Path | None,
    gain_map_config: GainMapConfig,
    jpeg_quality: int,
    external_boost: float,
) -> tuple[list[ConversionResult], list[tuple[ConversionJob, Exception]], list[tuple[ConversionJob, Exception]]]:
    """Execute conversion jobs with Rich progress reporting."""
    successes: list[ConversionResult] = []
    failures: list[tuple[ConversionJob, Exception]] = []
    skipped: list[tuple[ConversionJob, Exception]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        batch_task = progress.add_task(
            "[bold cyan]Batch progress[/bold cyan]",
            total=len(jobs),
            visible=len(jobs) > 1,
        )

        def process_job(job: ConversionJob) -> tuple[ConversionJob, ConversionResult | None, Exception | None]:
            file_task = progress.add_task(
                f"[green]{job.input_path.name}[/green]",
                total=1,
            )
            progress_callback = _make_progress_callback(progress, file_task, job.input_path.name)

            try:
                result = convert_jpeg_to_ultrahdr(
                    input_jpeg=job.input_path,
                    output_jpeg=job.output_path,
                    gain_map_path=gain_map_path,
                    gain_map_config=gain_map_config,
                    progress_callback=progress_callback,
                    jpeg_quality=jpeg_quality,
                    max_content_boost=external_boost if gain_map_path is not None else None,
                )
                return job, result, None
            except Exception as exc:
                return job, None, exc
            finally:
                progress.update(batch_task, advance=1)
                progress.remove_task(file_task)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_job, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                job, result, exc = future.result()
                if isinstance(exc, AlreadyUltraHDRError):
                    skipped.append((job, exc))
                    progress.console.print(f"[yellow]Skipped[/yellow] {job.input_path}: Already an Ultra HDR image.")
                elif exc is not None:
                    failures.append((job, exc))
                elif result is not None:
                    successes.append(result)

    return successes, failures, skipped


def _print_results(
    console: Console,
    successes: Sequence[ConversionResult],
    failures: Sequence[tuple[ConversionJob, Exception]],
    skipped: Sequence[tuple[ConversionJob, Exception]],
    is_batch_mode: bool,
) -> None:
    """Render a compact post-run summary."""
    if not is_batch_mode and len(successes) == 1 and not failures:
        result = successes[0]
        console.print(f"Wrote Ultra HDR JPEG: {result.output_path}")
        console.print(f"Gain map source: {result.gain_map_source}")
        console.print(f"ICC profile found: {result.has_icc}")
        return

    for result in successes:
        console.print(f"[green]Converted[/green] {result.output_path}")

    for job, exc in failures:
        console.print(f"[bold red]Failed[/bold red] {job.input_path}: {exc}")

    console.print(f"Completed batch: {len(successes)} succeeded, {len(failures)} failed, {len(skipped)} skipped.")


def main(argv: Sequence[str] | None = None) -> None:
    """Run the Ultra HDR conversion CLI."""
    if not _RICH_AVAILABLE:
        raise SystemExit(
            "The CLI requires the 'rich' package. Install it with:\n"
            "    pip install 'ultra-hdr-converter[cli]'\n"
            "    uv sync --extra cli"
        )
    console = Console()
    parser = _build_parser()
    args = parser.parse_args(argv)

    jobs = _build_jobs(parser, args)
    gain_map_config = _build_gain_map_config(args)
    successes, failures, skipped = _run_jobs(
        console, jobs, args.gain_map, gain_map_config, args.jpeg_quality, args.max_boost_factor
    )

    _print_results(console, successes, failures, skipped, is_batch_mode=args.batch_inputs is not None)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
