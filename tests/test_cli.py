from pathlib import Path

from ultra_hdr_converter.ui import cli


def test_build_jobs_preserves_single_file_mode(tmp_path: Path) -> None:
    input_file = tmp_path / "input.jpg"
    output_file = tmp_path / "output.jpg"
    input_file.write_bytes(b"jpeg")

    args = cli._parse_args([str(input_file), str(output_file)])
    jobs = cli._build_jobs(cli._build_parser(), args)

    assert jobs == [cli.ConversionJob(input_path=input_file, output_path=output_file)]


def test_build_jobs_defaults_single_output_path(tmp_path: Path) -> None:
    input_file = tmp_path / "input.jpg"
    input_file.write_bytes(b"jpeg")

    args = cli._parse_args([str(input_file)])
    jobs = cli._build_jobs(cli._build_parser(), args)

    assert jobs == [
        cli.ConversionJob(
            input_path=input_file,
            output_path=tmp_path / "input_ultrahdr.jpg",
        )
    ]


def test_build_jobs_collects_batch_inputs_in_sorted_order(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch"
    out_dir = tmp_path / "out"
    batch_dir.mkdir()

    first = batch_dir / "b.jpg"
    second = batch_dir / "a.jpeg"
    ignored = batch_dir / "ignore.png"
    first.write_bytes(b"jpeg")
    second.write_bytes(b"jpeg")
    ignored.write_bytes(b"png")

    args = cli._parse_args(["--batch-inputs", str(batch_dir), "--out-dir", str(out_dir)])
    jobs = cli._build_jobs(cli._build_parser(), args)

    assert jobs == [
        cli.ConversionJob(input_path=second, output_path=out_dir / "a_ultrahdr.jpg"),
        cli.ConversionJob(input_path=first, output_path=out_dir / "b_ultrahdr.jpg"),
    ]
