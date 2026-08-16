import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # type: ignore[import-not-found]  # noqa: E402

from ultra_hdr_converter.core.formats import ImageFormat  # noqa: E402
from ultra_hdr_converter.core.gain_map import GainMapConfig  # noqa: E402
from ultra_hdr_converter.ui import gui  # noqa: E402


@pytest.fixture(scope="module")
def application() -> QApplication:
    """Return the shared Qt application required to construct widgets."""
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(application: QApplication) -> gui.UltraHdrGui:
    """Create and dispose an offscreen converter window."""
    converter_window = gui.UltraHdrGui()
    yield converter_window
    converter_window.close()


def test_tuning_controls_expose_defaults_ranges_and_hints(window: gui.UltraHdrGui) -> None:
    threshold_slider_value = 35
    expected_threshold = 0.35
    gamma_value = 3.4
    expected_gamma_slider_value = 340
    controls = [
        (window.highlight_threshold_spinbox, window.highlight_threshold_slider, 0.01, 0.99, 0.50),
        (window.expansion_gamma_spinbox, window.expansion_gamma_slider, 0.10, 5.00, 2.20),
        (window.max_boost_factor_spinbox, window.max_boost_factor_slider, 0.10, 10.00, 3.00),
        (window.bloom_weight_spinbox, window.bloom_weight_slider, 0.00, 1.00, 0.15),
    ]

    assert not window.tuning_panel.isVisible()
    for spinbox, slider, minimum, maximum, default in controls:
        assert spinbox.minimum() == minimum
        assert spinbox.maximum() == maximum
        assert spinbox.value() == default
        assert slider.value() == round(default * 100)
        assert spinbox.toolTip()
        assert slider.toolTip() == spinbox.toolTip()

    window.highlight_threshold_slider.setValue(threshold_slider_value)
    assert window.highlight_threshold_spinbox.value() == expected_threshold

    window.expansion_gamma_spinbox.setValue(gamma_value)
    assert window.expansion_gamma_slider.value() == expected_gamma_slider_value


def test_redesigned_queue_switches_between_empty_and_populated_states(
    window: gui.UltraHdrGui,
    tmp_path: Path,
) -> None:
    assert window._queue_stack.currentIndex() == 0
    assert window._drop_hint.text() == "Drop photos here"
    assert window.lbl_queue_count.text() == "0 photos"
    assert window.btn_add.toolTip()
    assert window.btn_start.text() == "Start conversion"

    input_path = tmp_path / "photo.jpg"
    input_path.write_bytes(b"jpeg")
    window._add_file_to_table(input_path)

    assert window._queue_stack.currentIndex() == 1
    assert window.lbl_queue_count.text() == "1 photo"

    window._clear_queue()

    assert window._queue_stack.currentIndex() == 0
    assert window.lbl_queue_count.text() == "0 photos"


def test_tuning_toggle_communicates_expansion_state(window: gui.UltraHdrGui) -> None:
    assert window.btn_tuning.text() == "HDR tuning  |  Show"

    window.btn_tuning.setChecked(True)
    assert window.btn_tuning.text() == "HDR tuning  |  Hide"

    window.btn_tuning.setChecked(False)
    assert window.btn_tuning.text() == "HDR tuning  |  Show"


def test_start_conversion_forwards_selected_gain_map_config(
    window: gui.UltraHdrGui,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "photo.jpg"
    input_path.write_bytes(b"jpeg")
    window._add_file_to_table(input_path)
    window.highlight_threshold_spinbox.setValue(0.35)
    window.expansion_gamma_spinbox.setValue(3.4)
    window.max_boost_factor_spinbox.setValue(6.0)
    window.bloom_weight_spinbox.setValue(0.0)

    monkeypatch.setattr(gui.WorkerThread, "start", lambda self: None)
    window._start_conversion()

    assert window.worker is not None
    assert window.worker.gain_map_config == GainMapConfig(
        highlight_threshold=0.35,
        expansion_gamma=3.4,
        max_boost_factor=6.0,
        bloom_weight=0.0,
    )
    assert not window.tuning_panel.isEnabled()

    window._on_finished(1, 0)
    assert window.tuning_panel.isEnabled()


def test_queue_accepts_avif_and_rejects_unsupported_suffixes(
    window: gui.UltraHdrGui,
    tmp_path: Path,
) -> None:
    """The queue is driven by the shared format registry, not a hardcoded JPEG list."""
    for name in ("photo.jpg", "photo.avif"):
        path = tmp_path / name
        path.write_bytes(b"data")
        window._add_file_to_table(path)

    expected_queued = 2
    assert window.table.rowCount() == expected_queued
    assert gui.is_supported_path(tmp_path / "photo.avif") is True
    assert gui.is_supported_path(tmp_path / "photo.png") is False


def test_format_selector_offers_every_container(window: gui.UltraHdrGui) -> None:
    labels = [window.combo_format.itemText(i) for i in range(window.combo_format.count())]
    values = [window.combo_format.itemData(i) for i in range(window.combo_format.count())]

    assert labels[0] == "Same as input"
    assert values[0] is None
    assert set(values[1:]) == set(ImageFormat)


def test_start_conversion_forwards_selected_output_format(
    window: gui.UltraHdrGui,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "photo.jpg"
    input_path.write_bytes(b"jpeg")
    window._add_file_to_table(input_path)
    window.combo_format.setCurrentIndex(window.combo_format.findData(ImageFormat.AVIF))

    monkeypatch.setattr(gui.WorkerThread, "start", lambda self: None)
    window._start_conversion()

    assert window.worker is not None
    assert window.worker.output_format is ImageFormat.AVIF
    assert not window.combo_format.isEnabled()

    window._on_finished(1, 0)
    assert window.combo_format.isEnabled()


@pytest.mark.parametrize(
    ("input_name", "output_format", "expected_name"),
    [
        ("photo.jpg", None, "photo_ultrahdr.jpg"),
        ("photo.avif", None, "photo_ultrahdr.avif"),
        ("photo.jpg", ImageFormat.AVIF, "photo_ultrahdr.avif"),
        ("photo.avif", ImageFormat.JPEG, "photo_ultrahdr.jpg"),
    ],
)
def test_worker_output_name_follows_selected_container(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    input_name: str,
    output_format: ImageFormat | None,
    expected_name: str,
) -> None:
    """Output files take the resolved container's extension, not always .jpg."""
    input_path = tmp_path / input_name
    captured: list[Path] = []
    monkeypatch.setattr(gui, "convert_to_ultrahdr", lambda **kwargs: captured.append(kwargs["output_path"]))

    worker = gui.WorkerThread([input_path], tmp_path, GainMapConfig(), output_format)
    worker._process_file(0, input_path, 1, [0.0], [0.0], gui.Lock(), GainMapConfig())

    assert captured == [tmp_path / expected_name]
