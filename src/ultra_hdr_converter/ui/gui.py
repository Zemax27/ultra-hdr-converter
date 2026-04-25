"""Optional desktop GUI for batch Ultra HDR conversion."""

from __future__ import annotations

import sys
import time
import warnings
from importlib import resources
from pathlib import Path

from ultra_hdr_converter.core.converter import convert_jpeg_to_ultrahdr
from ultra_hdr_converter.core.gain_map import GainMapConfig
from ultra_hdr_converter.errors import AlreadyUltraHDRError
from ultra_hdr_converter.ui._gui_style import C_TEXT_DIM, STATUS_COLORS, STYLESHEET

# Minimum seconds between progress signal emissions to avoid flooding the UI event loop.
_PROGRESS_THROTTLE_SECONDS: float = 0.05

try:
    from PySide6.QtCore import QThread, Signal, Slot
    from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QIcon
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QFileDialog,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QSizePolicy,
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


def _make_progress_callback(
    worker: "WorkerThread",
    index: int,
    total_files: int,
    file_name: str,
) -> "ProgressCallbackType":
    """Return a closure that captures index by value, avoiding late-binding bugs.

    Args:
        worker: The running WorkerThread instance.
        index: Zero-based index of the current file in the batch (captured by value).
        total_files: Total number of files in the batch.
        file_name: Display name of the current file.

    Returns:
        A progress callback compatible with ``convert_jpeg_to_ultrahdr``.
    """
    last_update: list[float] = [0.0]

    def _callback(message: str, step: int, total_steps: int) -> None:
        if worker.is_cancelled:
            raise RuntimeError("Cancelled")
        now = time.monotonic()
        if now - last_update[0] > _PROGRESS_THROTTLE_SECONDS or step == total_steps:
            progress_value = (index + (step / total_steps)) / total_files
            worker.progress.emit(f"[{file_name}] {message}", progress_value)
            last_update[0] = now

    return _callback


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

    class WorkerThread(QThread):
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

        def __init__(self, input_files: list[Path], output_dir: Path | None) -> None:
            super().__init__()
            self.input_files = input_files
            self.output_dir = output_dir
            self.is_cancelled = False

        def cancel(self) -> None:
            """Request cancellation. The worker stops after the current file."""
            self.is_cancelled = True

        def run(self) -> None:
            """Execute the conversion batch. Called by Qt on the worker thread."""
            total_files = len(self.input_files)
            gain_map_config = GainMapConfig()
            failures = 0
            successes = 0

            for index, input_path in enumerate(self.input_files):
                if self.is_cancelled:
                    self.log.emit("Conversion cancelled by user.")
                    break

                output_name = f"{input_path.stem}_ultrahdr.jpg"
                output_path = (
                    self.output_dir / output_name if self.output_dir is not None else input_path.with_name(output_name)
                )
                self.status_update.emit(index, "Processing")
                progress_cb = _make_progress_callback(self, index, total_files, input_path.name)

                try:
                    convert_jpeg_to_ultrahdr(
                        input_jpeg=input_path,
                        output_jpeg=output_path,
                        gain_map_config=gain_map_config,
                        progress_callback=progress_cb,
                    )
                    successes += 1
                    self.status_update.emit(index, "OK")
                    self.log.emit(f"[{input_path.name}] Wrote {output_path}")
                except AlreadyUltraHDRError:
                    self.status_update.emit(index, "Skipped")
                    self.log.emit(f"[{input_path.name}] Skipped: Already an Ultra HDR image.")
                except Exception as exc:
                    if self.is_cancelled and str(exc) == "Cancelled":
                        self.status_update.emit(index, "Cancelled")
                        break
                    failures += 1
                    self.status_update.emit(index, "Failed")
                    self.log.emit(f"[{input_path.name}] Failed: {exc}")

            self.finished.emit(successes, failures)

    # ── Main window ────────────────────────────────────────────────────────────

    class UltraHdrGui(QMainWindow):
        """Premium dark-themed desktop application for batch Ultra HDR conversion."""

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Ultra HDR Studio")
            self.resize(1080, 720)
            self.setMinimumSize(720, 520)
            self.setAcceptDrops(True)

            self.output_dir: Path | None = None
            self.worker: WorkerThread | None = None
            self._queued_paths: set[Path] = set()

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
            body_layout.setContentsMargins(18, 14, 18, 14)
            body_layout.setSpacing(10)

            body_layout.addLayout(self._build_toolbar())
            body_layout.addWidget(self._build_queue_section(), stretch=3)
            body_layout.addWidget(self._build_log_section(), stretch=1)
            body_layout.addWidget(self._build_bottom_bar())

            root.addWidget(body)

        def _build_header(self) -> QWidget:
            header = QWidget()
            header.setObjectName("header")
            row = QHBoxLayout(header)
            row.setContentsMargins(18, 0, 18, 0)
            row.setSpacing(10)

            if _ICON_PATH is not None:
                icon_lbl = QLabel()
                icon_lbl.setPixmap(QIcon(str(_ICON_PATH)).pixmap(24, 24))
                row.addWidget(icon_lbl)

            title = QLabel("Ultra HDR Studio")
            title.setObjectName("app_title")
            row.addWidget(title)

            row.addStretch()

            version = QLabel("v0.1.0")
            version.setObjectName("version_badge")
            row.addWidget(version)

            return header

        def _build_toolbar(self) -> QHBoxLayout:
            bar = QHBoxLayout()
            bar.setSpacing(6)

            self.btn_add = QPushButton("＋  Add Photos")
            self.btn_add.clicked.connect(self._add_photos)

            self.btn_remove = QPushButton("✕  Remove")
            self.btn_remove.clicked.connect(self._remove_selected)

            self.btn_clear = QPushButton("⊘  Clear")
            self.btn_clear.clicked.connect(self._clear_queue)

            bar.addWidget(self.btn_add)
            bar.addWidget(self.btn_remove)
            bar.addWidget(self.btn_clear)
            bar.addStretch()

            self.lbl_output = QLabel("Output: alongside original files")
            self.lbl_output.setObjectName("output_label")
            bar.addWidget(self.lbl_output)

            self.btn_output = QPushButton("📁  Output Folder")
            self.btn_output.clicked.connect(self._set_output)
            bar.addWidget(self.btn_output)

            return bar

        def _build_queue_section(self) -> QWidget:
            container = QWidget()
            layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)

            lbl = QLabel("QUEUE")
            lbl.setObjectName("section_label")
            layout.addWidget(lbl)

            # Stacked widget: index 0 = drop hint, index 1 = table
            self._queue_stack = QStackedWidget()

            self._drop_hint = QLabel("Drop JPEG files here  ·  or click  ＋ Add Photos")
            self._drop_hint.setObjectName("drop_hint")
            self._drop_hint.setAlignment(self._drop_hint.alignment())
            from PySide6.QtCore import Qt as _Qt  # noqa: PLC0415 — local import inside guard

            self._drop_hint.setAlignment(_Qt.AlignmentFlag.AlignCenter)
            self._queue_stack.addWidget(self._drop_hint)  # index 0

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
            layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)

            hdr = QHBoxLayout()
            lbl = QLabel("CONSOLE")
            lbl.setObjectName("section_label")
            hdr.addWidget(lbl)
            hdr.addStretch()

            btn_clear_log = QPushButton("Clear")
            btn_clear_log.setFixedWidth(60)
            btn_clear_log.clicked.connect(self._clear_log)
            hdr.addWidget(btn_clear_log)
            layout.addLayout(hdr)

            self.log_text = QTextEdit()
            self.log_text.setReadOnly(True)
            layout.addWidget(self.log_text)
            return container

        def _build_bottom_bar(self) -> QWidget:
            container = QWidget()
            layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 4, 0, 0)
            layout.setSpacing(6)

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

            self.btn_start = QPushButton("▶  Start Conversion")
            self.btn_start.setObjectName("btn_start")
            self.btn_start.clicked.connect(self._start_conversion)
            action_row.addWidget(self.btn_start)

            layout.addLayout(action_row)
            return container

        # ── Helpers ────────────────────────────────────────────────────────────

        def _update_empty_state(self) -> None:
            """Switch the stacked widget between drop hint (empty) and table."""
            self._queue_stack.setCurrentIndex(0 if self.table.rowCount() == 0 else 1)

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
                self.lbl_output.setText(f"Output: {self.output_dir}")

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
            self.progress_bar.setValue(0)
            self.lbl_pct.setText("0%")
            self.log_text.clear()
            self.lbl_status.setText("Starting…")

            self.worker = WorkerThread(input_files, self.output_dir)
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

        @Slot(str, float)
        def _on_progress(self, msg: str, val: float) -> None:
            self.progress_bar.setValue(int(val * 1000))
            self.lbl_pct.setText(f"{int(val * 100)}%")
            self.lbl_status.setText(msg)

        @Slot(str)
        def _on_log(self, msg: str) -> None:
            self.log_text.append(msg)

        @Slot(int, str)
        def _on_status_update(self, row: int, status: str) -> None:
            self._set_status_item(row, status)
            item = self.table.item(row, 0)
            if item is not None:
                self.table.scrollToItem(item)

        @Slot(int, int)
        def _on_finished(self, successes: int, failures: int) -> None:
            self.btn_start.setEnabled(True)
            self.btn_cancel.setEnabled(False)
            self.btn_add.setEnabled(True)
            self.btn_remove.setEnabled(True)
            self.btn_clear.setEnabled(True)

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
