"""Optional desktop GUI for batch Ultra HDR conversion."""

from __future__ import annotations

import importlib
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

from ultra_hdr_converter.gainmap import GainMapConfig
from ultra_hdr_converter.pipeline import convert_jpeg_to_ultrahdr

POLL_INTERVAL_MS = 50
WINDOW_WIDTH = 920
WINDOW_HEIGHT = 620


@dataclass(frozen=True)
class GuiEvent:
    """Message produced by the worker thread for the UI thread."""

    kind: str
    message: str | None = None
    value: float | None = None


def _load_customtkinter() -> Any:
    """Import CustomTkinter lazily so the package stays importable without GUI extras."""
    try:
        return importlib.import_module("customtkinter")
    except ModuleNotFoundError as exc:
        raise SystemExit("The GUI requires the optional 'gui' extra. Install it with `uv sync --extra gui`.") from exc


class UltraHdrGui:
    """Desktop application for batch Ultra HDR conversion."""

    def __init__(self, ctk: Any) -> None:
        self._ctk = ctk
        self._events: queue.SimpleQueue[GuiEvent] = queue.SimpleQueue()
        self._worker: threading.Thread | None = None
        self._input_files: list[Path] = []
        self._output_dir: Path | None = None

        self._ctk.set_appearance_mode("System")
        self._ctk.set_default_color_theme("blue")

        self._root = self._ctk.CTk()
        self._root.title("Ultra HDR Converter")
        self._root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self._root.minsize(820, 560)
        self._root.grid_columnconfigure(1, weight=1)
        self._root.grid_rowconfigure(0, weight=1)

        self._sidebar = self._ctk.CTkFrame(self._root, width=220, corner_radius=0)
        self._sidebar.grid(row=0, column=0, sticky="nsew")
        self._sidebar.grid_columnconfigure(0, weight=1)

        self._content = self._ctk.CTkFrame(self._root)
        self._content.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self._content.grid_columnconfigure(0, weight=1)
        self._content.grid_rowconfigure(1, weight=1)
        self._content.grid_rowconfigure(3, weight=1)

        self._build_sidebar()
        self._build_content()

        self._root.after(POLL_INTERVAL_MS, self._drain_events)

    def _build_sidebar(self) -> None:
        """Create the static sidebar controls."""
        title_label = self._ctk.CTkLabel(
            self._sidebar,
            text="Ultra HDR",
            font=self._ctk.CTkFont(size=22, weight="bold"),
        )
        title_label.grid(row=0, column=0, padx=20, pady=(24, 12), sticky="ew")

        subtitle_label = self._ctk.CTkLabel(
            self._sidebar,
            text="Batch conversion with coarse progress updates.",
            justify="left",
            wraplength=180,
        )
        subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="ew")

        self._select_files_button = self._ctk.CTkButton(
            self._sidebar,
            text="Select Files",
            command=self._select_files,
        )
        self._select_files_button.grid(row=2, column=0, padx=20, pady=8, sticky="ew")

        self._select_output_button = self._ctk.CTkButton(
            self._sidebar,
            text="Select Output Folder",
            command=self._select_output_dir,
        )
        self._select_output_button.grid(row=3, column=0, padx=20, pady=8, sticky="ew")

        self._convert_button = self._ctk.CTkButton(
            self._sidebar,
            text="Start Conversion",
            command=self._start_conversion,
        )
        self._convert_button.grid(row=4, column=0, padx=20, pady=8, sticky="ew")

        self._appearance_menu = self._ctk.CTkOptionMenu(
            self._sidebar,
            values=["Light", "Dark", "System"],
            command=self._ctk.set_appearance_mode,
        )
        self._appearance_menu.set("System")
        self._appearance_menu.grid(row=5, column=0, padx=20, pady=(24, 20), sticky="ew")

        self._output_label = self._ctk.CTkLabel(
            self._sidebar,
            text="Output: next to each input file",
            justify="left",
            wraplength=180,
        )
        self._output_label.grid(row=6, column=0, padx=20, pady=(0, 20), sticky="ew")

    def _build_content(self) -> None:
        """Create the main content area."""
        files_label = self._ctk.CTkLabel(
            self._content,
            text="Selected Files",
            anchor="w",
            font=self._ctk.CTkFont(size=16, weight="bold"),
        )
        files_label.grid(row=0, column=0, padx=20, pady=(20, 8), sticky="ew")

        self._files_textbox = self._ctk.CTkTextbox(self._content, height=160)
        self._files_textbox.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")

        log_label = self._ctk.CTkLabel(
            self._content,
            text="Conversion Log",
            anchor="w",
            font=self._ctk.CTkFont(size=16, weight="bold"),
        )
        log_label.grid(row=2, column=0, padx=20, pady=(0, 8), sticky="ew")

        self._log_textbox = self._ctk.CTkTextbox(self._content)
        self._log_textbox.grid(row=3, column=0, padx=20, pady=(0, 20), sticky="nsew")

        self._progress_bar = self._ctk.CTkProgressBar(self._content)
        self._progress_bar.grid(row=4, column=0, padx=20, pady=(0, 20), sticky="ew")
        self._progress_bar.set(0.0)

    def _select_files(self) -> None:
        """Let the user choose one or more JPEG files."""
        selected = filedialog.askopenfilenames(
            title="Select SDR JPEG files",
            filetypes=[("JPEG files", "*.jpg *.jpeg")],
        )
        if not selected:
            return

        self._input_files = [Path(path) for path in selected]
        self._refresh_file_list()
        self._append_log(f"Selected {len(self._input_files)} input files.")

    def _select_output_dir(self) -> None:
        """Let the user choose an output directory."""
        selected = filedialog.askdirectory(title="Select Output Directory")
        if not selected:
            return

        self._output_dir = Path(selected)
        self._output_label.configure(text=f"Output: {self._output_dir}")
        self._append_log(f"Selected output directory: {self._output_dir}")

    def _refresh_file_list(self) -> None:
        """Render the selected input files."""
        self._files_textbox.delete("1.0", "end")
        for path in self._input_files:
            self._files_textbox.insert("end", f"{path}\n")
        self._files_textbox.see("end")

    def _append_log(self, message: str) -> None:
        """Append one line to the GUI log."""
        self._log_textbox.insert("end", f"{message}\n")
        self._log_textbox.see("end")

    def _set_busy(self, busy: bool) -> None:
        """Toggle the conversion controls while a worker is active."""
        state = "disabled" if busy else "normal"
        self._convert_button.configure(state=state)
        self._select_files_button.configure(state=state)
        self._select_output_button.configure(state=state)

    def _start_conversion(self) -> None:
        """Kick off batch conversion in a background thread."""
        if not self._input_files:
            messagebox.showerror("Ultra HDR Converter", "Select at least one JPEG file before converting.")
            return

        if self._worker is not None and self._worker.is_alive():
            messagebox.showinfo("Ultra HDR Converter", "A batch conversion is already running.")
            return

        self._progress_bar.set(0.0)
        self._append_log(f"Starting batch conversion for {len(self._input_files)} files.")
        self._set_busy(True)

        self._worker = threading.Thread(target=self._process_batch, daemon=True)
        self._worker.start()

    def _process_batch(self) -> None:
        """Convert the selected files without blocking the UI thread."""
        total_files = len(self._input_files)
        gain_map_config = GainMapConfig()
        failures = 0

        for index, input_path in enumerate(self._input_files):
            output_path = self._build_output_path(input_path)

            def _progress_callback(message: str, step: int, total_steps: int) -> None:
                progress_value = (index + (step / total_steps)) / total_files
                self._events.put(GuiEvent(kind="progress", value=progress_value))
                self._events.put(GuiEvent(kind="log", message=f"[{input_path.name}] {message}"))

            try:
                convert_jpeg_to_ultrahdr(
                    input_jpeg=input_path,
                    output_jpeg=output_path,
                    gain_map_config=gain_map_config,
                    progress_callback=_progress_callback,
                )
            except Exception as exc:  # pragma: no cover - depends on GUI interactions
                failures += 1
                self._events.put(GuiEvent(kind="log", message=f"[{input_path.name}] Failed: {exc}"))
            else:
                self._events.put(GuiEvent(kind="log", message=f"[{input_path.name}] Wrote {output_path}"))

        self._events.put(GuiEvent(kind="progress", value=1.0))
        self._events.put(
            GuiEvent(
                kind="log",
                message=f"Batch conversion complete. {total_files - failures} succeeded, {failures} failed.",
            )
        )
        self._events.put(GuiEvent(kind="idle"))

    def _build_output_path(self, input_path: Path) -> Path:
        """Resolve the output path for one GUI conversion job."""
        output_name = f"{input_path.stem}_ultrahdr.jpg"
        if self._output_dir is not None:
            return self._output_dir / output_name
        return input_path.with_name(output_name)

    def _drain_events(self) -> None:
        """Apply worker-thread events on the Tk main loop."""
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break

            if event.kind == "log" and event.message is not None:
                self._append_log(event.message)
            elif event.kind == "progress" and event.value is not None:
                self._progress_bar.set(event.value)
            elif event.kind == "idle":
                self._set_busy(False)

        self._root.after(POLL_INTERVAL_MS, self._drain_events)

    def run(self) -> None:
        """Start the GUI event loop."""
        self._root.mainloop()


def main() -> None:
    """Launch the optional desktop GUI."""
    ctk = _load_customtkinter()
    app = UltraHdrGui(ctk)
    app.run()


if __name__ == "__main__":
    main()
