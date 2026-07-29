"""Optional desktop GUI for batch Ultra HDR conversion."""

from __future__ import annotations

import concurrent.futures
import sys
import time
import warnings
from importlib import resources
from pathlib import Path
from threading import Lock

from ultra_hdr_converter.core.converter import convert_jpeg_to_ultrahdr
from ultra_hdr_converter.core.gain_map import GainMapConfig
from ultra_hdr_converter.errors import AlreadyUltraHDRError
from ultra_hdr_converter.ui._gui_style import C_TEXT_DIM, STATUS_COLORS, STYLESHEET

# Minimum seconds between progress signal emissions to avoid flooding the UI event loop.
_PROGRESS_THROTTLE_SECONDS: float = 0.05

try:
    from PySide6.QtCore import Qt, QThread, Signal, Slot  # type: ignore[import-not-found]
    from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QIcon  # type: ignore[import-not-found]
    from PySide6.QtWidgets import (  # type: ignore[import-not-found]
        QAbstractItemView,
        QApplication,
        QDoubleSpinBox,
        QFileDialog,
        QGridLayout,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QSizePolicy,
        QSlider,
        QStackedWidget,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYSIDE = True
except ImportError:
    HAS_PYSIDE = False


# Keep the context manager alive for the whole process so any temp file is never
# cleaned up while the app is running. Degrades to None with a warning if missing.
_ICON_PATH: Path | None
try:
    _ICON_CTX = resources.as_file(resources.files("ultra_hdr_converter.ui.assets").joinpath("icon.png"))
    _ICON_PATH = _ICON_CTX.__enter__()
except Exception as _icon_exc:  # noqa: BLE001
    _ICON_PATH = None
    warnings.warn(
        f"ultra_hdr_converter: could not resolve bundled icon — the window will use the OS default. Cause: {_icon_exc}",
        stacklevel=1,
    )


if HAS_PYSIDE:
    from typing import Callable

    ProgressCallbackType = Callable[[str, int, int], None]

    class WorkerThread(QThread):  # type: ignore[misc]
        """Background thread that converts a batch of JPEG files to Ultra HDR.

        Signals:
            progress: Emits (status_message, 0.0–1.0 fraction) during conversion.
            log: Emits a single log line string.
            status_update: Emits (row_index, status_label) for per-file status.
            finished: Emits (successes, failures) when the batch is done.
        """

        progress = Signal(str, float)
        log = Signal(str)
        status_update = Signal(int, str)
        finished = Signal(int, int)

        def __init__(
            self,
            input_files: list[Path],
            output_dir: Path | None,
            gain_map_config: GainMapConfig,
        ) -> None:
            super().__init__()
            self.input_files = input_files
            self.output_dir = output_dir
            self.gain_map_config = gain_map_config
            self.is_cancelled = False

        def cancel(self) -> None:
            """Request cancellation. The worker stops after the current file."""
            self.is_cancelled = True

        def _process_file(
            self,
            index: int,
            input_path: Path,
            total_files: int,
            file_progress: list[float],
            last_emit_time: list[float],
            emit_lock: Lock,
            gain_map_config: GainMapConfig,
        ) -> tuple[int, Path, bool, Exception | None]:
            if self.is_cancelled:
                return index, input_path, False, RuntimeError("Cancelled")

            output_name = f"{input_path.stem}_ultrahdr.jpg"
            output_path = (
                self.output_dir / output_name if self.output_dir is not None else input_path.with_name(output_name)
            )
            self.status_update.emit(index, "Processing")

            def progress_cb(message: str, step: int, total_steps: int) -> None:
                if self.is_cancelled:
                    raise RuntimeError("Cancelled")

                file_progress[index] = step / total_steps
                overall_val = sum(file_progress) / total_files

                now = time.monotonic()
                if now - last_emit_time[0] > _PROGRESS_THROTTLE_SECONDS or step == total_steps:
                    with emit_lock:
                        if time.monotonic() - last_emit_time[0] > _PROGRESS_THROTTLE_SECONDS or step == total_steps:
                            self.progress.emit(f"[{input_path.name}] {message}", overall_val)
                            last_emit_time[0] = time.monotonic()

            try:
                convert_jpeg_to_ultrahdr(
                    input_jpeg=input_path,
                    output_jpeg=output_path,
                    gain_map_config=gain_map_config,
                    progress_callback=progress_cb,
                )
                return index, input_path, True, None
            except AlreadyUltraHDRError as exc:
                file_progress[index] = 1.0
                return index, input_path, False, exc
            except Exception as exc:
                return index, input_path, False, exc

        def run(self) -> None:
            """Execute the conversion batch. Called by Qt on the worker thread."""
            total_files = len(self.input_files)
            failures = 0
            successes = 0

            file_progress = [0.0] * total_files
            last_emit_time = [0.0]
            emit_lock = Lock()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        self._process_file,
                        i,
                        p,
                        total_files,
                        file_progress,
                        last_emit_time,
                        emit_lock,
                        self.gain_map_config,
                    ): i
                    for i, p in enumerate(self.input_files)
                }

                for future in concurrent.futures.as_completed(futures):
                    if self.is_cancelled:
                        for f in futures:
                            f.cancel()
                        break

                    index, input_path, success, exc = future.result()
                    if success:
                        successes += 1
                        self.status_update.emit(index, "OK")
                        self.log.emit(f"[{input_path.name}] Wrote output")
                    elif isinstance(exc, AlreadyUltraHDRError):
                        self.status_update.emit(index, "Skipped")
                        self.log.emit(f"[{input_path.name}] Skipped: Already an Ultra HDR image.")
                    elif isinstance(exc, RuntimeError) and str(exc) == "Cancelled":
                        self.status_update.emit(index, "Cancelled")
                    else:
                        failures += 1
                        self.status_update.emit(index, "Failed")
                        self.log.emit(f"[{input_path.name}] Failed: {exc}")

            if self.is_cancelled:
                self.log.emit("Conversion cancelled by user.")

            self.finished.emit(successes, failures)

    # ── Main window ────────────────────────────────────────────────────────────

    class UltraHdrGui(QMainWindow):  # type: ignore[misc]
        """Premium dark-themed desktop application for batch Ultra HDR conversion."""

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Ultra HDR Converter")
            self.resize(1080, 720)
            self.setMinimumSize(720, 520)
            self.setAcceptDrops(True)

            self.output_dir: Path | None = None
            self.worker: WorkerThread | None = None
            self._queued_paths: set[Path] = set()
            self.highlight_threshold_spinbox: QDoubleSpinBox
            self.expansion_gamma_spinbox: QDoubleSpinBox
            self.max_boost_factor_spinbox: QDoubleSpinBox
            self.bloom_weight_spinbox: QDoubleSpinBox

            self._apply_icon()
            self._setup_ui()
            self.setStyleSheet(STYLESHEET)

        # ── Icon ───────────────────────────────────────────────────────────────

        def _apply_icon(self) -> None:
            """Set the window icon from the bundled PNG asset.

            Emits a warning (not a silent pass) if the asset is unavailable,
            satisfying the project's error-handling policy.
            """
            if _ICON_PATH is None:
                warnings.warn(
                    "ultra_hdr_converter: window icon not set because the bundled asset is unavailable.",
                    stacklevel=2,
                )
                return
            self.setWindowIcon(QIcon(str(_ICON_PATH)))

        # ── UI construction ────────────────────────────────────────────────────

        def _setup_ui(self) -> None:
            central = QWidget()
            central.setObjectName("central")
            self.setCentralWidget(central)

            root = QVBoxLayout(central)
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(0)

            root.addWidget(self._build_header())

            body = QWidget()
            body.setObjectName("body")
            body_layout = QVBoxLayout(body)
            body_layout.setContentsMargins(24, 20, 24, 20)
            body_layout.setSpacing(14)

            body_layout.addWidget(self._build_toolbar())
            body_layout.addWidget(self._build_tuning_section())
            body_layout.addWidget(self._build_queue_section(), stretch=3)
            body_layout.addWidget(self._build_log_section(), stretch=1)
            body_layout.addWidget(self._build_bottom_bar())

            root.addWidget(body)

        def _build_header(self) -> QWidget:
            header = QWidget()
            header.setObjectName("header")
            row = QHBoxLayout(header)
            row.setContentsMargins(24, 0, 24, 0)
            row.setSpacing(12)

            if _ICON_PATH is not None:
                icon_lbl = QLabel()
                icon_lbl.setObjectName("app_icon")
                icon_lbl.setPixmap(QIcon(str(_ICON_PATH)).pixmap(32, 32))
                row.addWidget(icon_lbl)

            brand = QWidget()
            brand.setObjectName("brand")
            brand_layout = QVBoxLayout(brand)
            brand_layout.setContentsMargins(0, 0, 0, 0)
            brand_layout.setSpacing(1)

            title = QLabel("Ultra HDR Converter")
            title.setObjectName("app_title")
            brand_layout.addWidget(title)

            subtitle = QLabel("Create JPEGs with gain maps for HDR displays")
            subtitle.setObjectName("app_subtitle")
            brand_layout.addWidget(subtitle)
            row.addWidget(brand)

            row.addStretch()

            return header

        def _build_toolbar(self) -> QWidget:
            toolbar = QWidget()
            toolbar.setObjectName("toolbar")
            bar = QHBoxLayout(toolbar)
            bar.setContentsMargins(10, 8, 10, 8)
            bar.setSpacing(8)

            self.btn_add = QPushButton("Add photos")
            self.btn_add.setObjectName("btn_add")
            self.btn_add.setToolTip("Add JPEG images to the conversion queue")
            self.btn_add.clicked.connect(self._add_photos)

            self.btn_remove = QPushButton("Remove")
            self.btn_remove.setToolTip("Remove the selected photos from the queue")
            self.btn_remove.clicked.connect(self._remove_selected)

            self.btn_clear = QPushButton("Clear")
            self.btn_clear.setToolTip("Remove all photos from the queue")
            self.btn_clear.clicked.connect(self._clear_queue)

            bar.addWidget(self.btn_add)
            bar.addWidget(self.btn_remove)
            bar.addWidget(self.btn_clear)
            bar.addStretch()

            output_group = QWidget()
            output_group.setObjectName("output_group")
            output_layout = QVBoxLayout(output_group)
            output_layout.setContentsMargins(0, 0, 0, 0)
            output_layout.setSpacing(0)

            output_caption = QLabel("OUTPUT LOCATION")
            output_caption.setObjectName("output_caption")
            output_layout.addWidget(output_caption)

            self.lbl_output = QLabel("Alongside original files")
            self.lbl_output.setObjectName("output_label")
            self.lbl_output.setToolTip("Converted files will be saved beside each source image")
            output_layout.addWidget(self.lbl_output)
            bar.addWidget(output_group)

            self.btn_output = QPushButton("Choose folder")
            self.btn_output.setObjectName("btn_output")
            self.btn_output.setToolTip("Choose a folder for all converted images")
            self.btn_output.clicked.connect(self._set_output)
            bar.addWidget(self.btn_output)

            return toolbar

        def _build_tuning_section(self) -> QWidget:
            section = QWidget()
            section.setObjectName("tuning_section")
            layout = QVBoxLayout(section)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)

            self.btn_tuning = QPushButton("HDR tuning  |  Show")
            self.btn_tuning.setObjectName("btn_tuning")
            self.btn_tuning.setCheckable(True)
            self.btn_tuning.setToolTip("Adjust how highlights are expanded into the Ultra HDR gain map")
            self.btn_tuning.toggled.connect(self._toggle_tuning_panel)
            layout.addWidget(self.btn_tuning)

            self.tuning_panel = QWidget()
            self.tuning_panel.setObjectName("tuning_panel")
            controls = QGridLayout(self.tuning_panel)
            controls.setContentsMargins(12, 10, 12, 10)
            controls.setHorizontalSpacing(12)
            controls.setVerticalSpacing(10)

            controls.addWidget(
                self._build_tuning_control(
                    "highlight_threshold",
                    "Highlight threshold",
                    "Lower values begin the HDR boost earlier, affecting more of the image.",
                    0.01,
                    0.99,
                    0.50,
                    0.01,
                ),
                0,
                0,
            )
            controls.addWidget(
                self._build_tuning_control(
                    "expansion_gamma",
                    "Expansion gamma",
                    "Higher values stretch highlights more aggressively.",
                    0.10,
                    5.00,
                    2.20,
                    0.10,
                ),
                0,
                1,
            )
            controls.addWidget(
                self._build_tuning_control(
                    "max_boost_factor",
                    "Maximum boost factor",
                    "Sets the maximum brightness amplification for the brightest pixels.",
                    0.10,
                    10.00,
                    3.00,
                    0.10,
                ),
                1,
                0,
            )
            controls.addWidget(
                self._build_tuning_control(
                    "bloom_weight",
                    "Bloom weight",
                    "Controls the soft halo around bright areas; set to 0 to disable bloom.",
                    0.00,
                    1.00,
                    0.15,
                    0.01,
                ),
                1,
                1,
            )
            controls.setColumnStretch(0, 1)
            controls.setColumnStretch(1, 1)
            self.tuning_panel.setVisible(False)
            layout.addWidget(self.tuning_panel)
            return section

        def _build_tuning_control(
            self,
            name: str,
            label_text: str,
            hint_text: str,
            minimum: float,
            maximum: float,
            default: float,
            step: float,
        ) -> QWidget:
            control = QWidget()
            control.setObjectName("tuning_control")
            layout = QVBoxLayout(control)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)

            heading = QHBoxLayout()
            label = QLabel(label_text)
            label.setObjectName("tuning_label")
            label.setToolTip(hint_text)
            heading.addWidget(label)
            heading.addStretch()

            spinbox = QDoubleSpinBox()
            spinbox.setObjectName(f"{name}_spinbox")
            spinbox.setRange(minimum, maximum)
            spinbox.setDecimals(2)
            spinbox.setSingleStep(step)
            spinbox.setValue(default)
            spinbox.setToolTip(hint_text)
            heading.addWidget(spinbox)
            layout.addLayout(heading)

            scale = 100
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setObjectName(f"{name}_slider")
            slider.setRange(round(minimum * scale), round(maximum * scale))
            slider.setSingleStep(max(1, round(step * scale)))
            slider.setValue(round(default * scale))
            slider.setToolTip(hint_text)
            slider.valueChanged.connect(lambda value, target=spinbox: target.setValue(value / scale))
            spinbox.valueChanged.connect(lambda value, target=slider: target.setValue(round(value * scale)))
            layout.addWidget(slider)

            hint = QLabel(hint_text)
            hint.setObjectName("tuning_hint")
            hint.setWordWrap(True)
            layout.addWidget(hint)

            setattr(self, f"{name}_spinbox", spinbox)
            setattr(self, f"{name}_slider", slider)
            return control

        def _toggle_tuning_panel(self, is_expanded: bool) -> None:
            self.tuning_panel.setVisible(is_expanded)
            self.btn_tuning.setText("HDR tuning  |  Hide" if is_expanded else "HDR tuning  |  Show")

        def _build_queue_section(self) -> QWidget:
            container = QWidget()
            container.setObjectName("queue_section")
            layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)

            heading = QHBoxLayout()
            lbl = QLabel("Conversion queue")
            lbl.setObjectName("section_title")
            heading.addWidget(lbl)
            heading.addStretch()
            self.lbl_queue_count = QLabel("0 photos")
            self.lbl_queue_count.setObjectName("section_meta")
            heading.addWidget(self.lbl_queue_count)
            layout.addLayout(heading)

            # Stacked widget: index 0 = drop hint, index 1 = table
            self._queue_stack = QStackedWidget()

            drop_zone = QWidget()
            drop_zone.setObjectName("drop_zone")
            drop_layout = QVBoxLayout(drop_zone)
            drop_layout.setContentsMargins(20, 20, 20, 20)
            drop_layout.setSpacing(5)
            drop_layout.addStretch()

            self._drop_hint = QLabel("Drop photos here")
            self._drop_hint.setObjectName("drop_hint")
            self._drop_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            drop_layout.addWidget(self._drop_hint)

            drop_detail = QLabel("JPEG files only, or use Add photos")
            drop_detail.setObjectName("drop_detail")
            drop_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
            drop_layout.addWidget(drop_detail)
            drop_layout.addStretch()
            self._queue_stack.addWidget(drop_zone)  # index 0

            self.table = QTableWidget(0, 3)
            self.table.setHorizontalHeaderLabels(["Filename", "Path", "Status"])
            self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
            self.table.setColumnWidth(2, 110)
            self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.table.setAlternatingRowColors(True)
            self.table.verticalHeader().setVisible(False)
            self.table.setShowGrid(True)
            self._queue_stack.addWidget(self.table)  # index 1

            layout.addWidget(self._queue_stack)
            self._update_empty_state()
            return container

        def _build_log_section(self) -> QWidget:
            container = QWidget()
            container.setObjectName("log_section")
            layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)

            hdr = QHBoxLayout()
            lbl = QLabel("Activity")
            lbl.setObjectName("section_title")
            hdr.addWidget(lbl)
            hdr.addStretch()

            btn_clear_log = QPushButton("Clear")
            btn_clear_log.setObjectName("btn_quiet")
            btn_clear_log.setToolTip("Clear the activity log")
            btn_clear_log.setFixedWidth(64)
            btn_clear_log.clicked.connect(self._clear_log)
            hdr.addWidget(btn_clear_log)
            layout.addLayout(hdr)

            self.log_text = QTextEdit()
            self.log_text.setReadOnly(True)
            layout.addWidget(self.log_text)
            return container

        def _build_bottom_bar(self) -> QWidget:
            container = QWidget()
            container.setObjectName("bottom_bar")
            layout = QVBoxLayout(container)
            layout.setContentsMargins(14, 12, 14, 12)
            layout.setSpacing(9)

            # Progress row
            prog_row = QHBoxLayout()
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 1000)
            self.progress_bar.setValue(0)
            prog_row.addWidget(self.progress_bar)

            self.lbl_pct = QLabel("0%")
            self.lbl_pct.setObjectName("pct_label")
            prog_row.addWidget(self.lbl_pct)
            layout.addLayout(prog_row)

            # Action row
            action_row = QHBoxLayout()
            self.lbl_status = QLabel("Ready")
            self.lbl_status.setObjectName("status_label")
            self.lbl_status.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            action_row.addWidget(self.lbl_status)

            self.btn_cancel = QPushButton("Cancel")
            self.btn_cancel.setObjectName("btn_cancel")
            self.btn_cancel.clicked.connect(self._cancel_conversion)
            self.btn_cancel.setEnabled(False)
            action_row.addWidget(self.btn_cancel)

            self.btn_start = QPushButton("Start conversion")
            self.btn_start.setObjectName("btn_start")
            self.btn_start.setToolTip("Convert every photo in the queue")
            self.btn_start.clicked.connect(self._start_conversion)
            action_row.addWidget(self.btn_start)

            layout.addLayout(action_row)
            return container

        # ── Helpers ────────────────────────────────────────────────────────────

        def _update_empty_state(self) -> None:
            """Switch the stacked widget between drop hint (empty) and table."""
            photo_count = self.table.rowCount()
            self._queue_stack.setCurrentIndex(0 if photo_count == 0 else 1)
            self.lbl_queue_count.setText(f"{photo_count} photo{'s' if photo_count != 1 else ''}")

        def _clear_log(self) -> None:
            self.log_text.clear()

        # ── Drag & Drop ────────────────────────────────────────────────────────

        def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
            if event.mimeData().hasUrls():
                event.acceptProposedAction()

        def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
            for url in event.mimeData().urls():
                path = Path(url.toLocalFile())
                if path.suffix.lower() in {".jpg", ".jpeg"}:
                    self._add_file_to_table(path)
            event.acceptProposedAction()

        # ── Queue management ───────────────────────────────────────────────────

        def _add_file_to_table(self, path: Path) -> None:
            """Add a single JPEG path to the queue, ignoring duplicates.

            Args:
                path: Absolute path to the JPEG file to enqueue.
            """
            resolved = path.resolve()
            if resolved in self._queued_paths:
                return
            self._queued_paths.add(resolved)

            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(path.name))
            self.table.setItem(row, 1, QTableWidgetItem(str(resolved)))
            self._set_status_item(row, "Queued")
            self._update_empty_state()

        def _set_status_item(self, row: int, status: str) -> None:
            """Write a coloured status cell into the given table row.

            Args:
                row: Zero-based table row index.
                status: Status label string (e.g. ``"OK"``, ``"Failed"``).
            """
            item = QTableWidgetItem(status)
            color = STATUS_COLORS.get(status, C_TEXT_DIM)
            item.setForeground(QColor(color))
            self.table.setItem(row, 2, item)

        def _add_photos(self) -> None:
            files, _ = QFileDialog.getOpenFileNames(self, "Select JPEG files", "", "JPEG (*.jpg *.jpeg)")
            for f in files:
                self._add_file_to_table(Path(f))

        def _set_output(self) -> None:
            dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
            if dir_path:
                self.output_dir = Path(dir_path)
                self.lbl_output.setText(str(self.output_dir))
                self.lbl_output.setToolTip(str(self.output_dir))

        def _remove_selected(self) -> None:
            rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
            for row in rows:
                item = self.table.item(row, 1)
                if item is not None:
                    self._queued_paths.discard(Path(item.text()))
                self.table.removeRow(row)
            self._update_empty_state()

        def _clear_queue(self) -> None:
            self._queued_paths.clear()
            self.table.setRowCount(0)
            self._update_empty_state()

        # ── Conversion control ─────────────────────────────────────────────────

        def _start_conversion(self) -> None:
            if self.table.rowCount() == 0:
                QMessageBox.warning(self, "Empty Queue", "Please add JPEG files to convert.")
                return

            input_files: list[Path] = []
            for row in range(self.table.rowCount()):
                path_item = self.table.item(row, 1)
                if path_item is not None:
                    input_files.append(Path(path_item.text()))
                self._set_status_item(row, "Queued")

            self.btn_start.setEnabled(False)
            self.btn_cancel.setEnabled(True)
            self.btn_add.setEnabled(False)
            self.btn_remove.setEnabled(False)
            self.btn_clear.setEnabled(False)
            self.tuning_panel.setEnabled(False)
            self.btn_tuning.setEnabled(False)
            self.progress_bar.setValue(0)
            self.lbl_pct.setText("0%")
            self.log_text.clear()
            self.lbl_status.setText("Starting…")

            gain_map_config = GainMapConfig(
                highlight_threshold=self.highlight_threshold_spinbox.value(),
                expansion_gamma=self.expansion_gamma_spinbox.value(),
                max_boost_factor=self.max_boost_factor_spinbox.value(),
                bloom_weight=self.bloom_weight_spinbox.value(),
            )
            self.worker = WorkerThread(input_files, self.output_dir, gain_map_config)
            self.worker.progress.connect(self._on_progress)
            self.worker.log.connect(self._on_log)
            self.worker.status_update.connect(self._on_status_update)
            self.worker.finished.connect(self._on_finished)
            self.worker.start()

        def _cancel_conversion(self) -> None:
            if self.worker and self.worker.isRunning():
                self.worker.cancel()
                self.btn_cancel.setEnabled(False)
                self.log_text.append("Cancelling…")
                self.lbl_status.setText("Cancelling…")

        # ── Signal slots ───────────────────────────────────────────────────────

        @Slot(str, float)  # type: ignore[untyped-decorator, misc]
        def _on_progress(self, msg: str, val: float) -> None:
            self.progress_bar.setValue(int(val * 1000))
            self.lbl_pct.setText(f"{int(val * 100)}%")
            self.lbl_status.setText(msg)

        @Slot(str)  # type: ignore[untyped-decorator, misc]
        def _on_log(self, msg: str) -> None:
            self.log_text.append(msg)

        @Slot(int, str)  # type: ignore[untyped-decorator, misc]
        def _on_status_update(self, row: int, status: str) -> None:
            self._set_status_item(row, status)
            item = self.table.item(row, 0)
            if item is not None:
                self.table.scrollToItem(item)

        @Slot(int, int)  # type: ignore[untyped-decorator, misc]
        def _on_finished(self, successes: int, failures: int) -> None:
            self.btn_start.setEnabled(True)
            self.btn_cancel.setEnabled(False)
            self.btn_add.setEnabled(True)
            self.btn_remove.setEnabled(True)
            self.btn_clear.setEnabled(True)
            self.tuning_panel.setEnabled(True)
            self.btn_tuning.setEnabled(True)

            if self.worker and not self.worker.is_cancelled:
                self.progress_bar.setValue(1000)
                self.lbl_pct.setText("100%")
                self.lbl_status.setText("Done")
            else:
                self.lbl_status.setText("Cancelled")

            self.log_text.append(f"Batch complete — {successes} succeeded, {failures} failed.")


def main() -> None:
    """Launch the optional desktop GUI."""
    if not HAS_PYSIDE:
        raise SystemExit("The GUI requires PySide6. Install it with: uv sync --extra gui")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = UltraHdrGui()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
