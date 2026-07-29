"""Dark-theme palette and QSS stylesheet for the Ultra HDR Converter GUI.

All colour constants are module-level strings so they can be referenced both
inside the QSS f-string and from Python code (e.g. for item foreground colours).
This module has no Qt imports so it can be imported without PySide6 present.
"""

from __future__ import annotations

# ── Palette ───────────────────────────────────────────────────────────────────

C_BG: str = "#0D0F18"
C_SURFACE: str = "#131622"
C_ELEVATED: str = "#1A1E30"
C_CARD: str = "#1E2238"
C_BORDER: str = "#22263E"

C_ACCENT: str = "#E89528"  # warm amber  — "HDR highlights"
C_ACCENT_ALT: str = "#F26C3A"  # burnt orange — gradient end

C_TEXT: str = "#E2E6F3"
C_TEXT_MID: str = "#7A87A8"
C_TEXT_DIM: str = "#4A5270"

C_OK: str = "#22C55E"
C_FAIL: str = "#F43F5E"
C_PROCESSING: str = "#60A5FA"
C_CANCELLED: str = "#F59E0B"

# Maps status label text → foreground colour used on table items.
STATUS_COLORS: dict[str, str] = {
    "OK": C_OK,
    "Failed": C_FAIL,
    "Processing": C_PROCESSING,
    "Queued": C_TEXT_DIM,
    "Skipped": C_CANCELLED,
    "Cancelled": C_CANCELLED,
}

# ── QSS ──────────────────────────────────────────────────────────────────────

STYLESHEET: str = f"""
/* Base */
QMainWindow, QWidget {{
    background-color: {C_BG};
    color: {C_TEXT};
    font-family: "Segoe UI", "SF Pro Display", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}}

/* Header bar */
QWidget#header {{
    background-color: {C_SURFACE};
    border-bottom: 1px solid {C_BORDER};
    min-height: 54px;
    max-height: 54px;
}}
QLabel#app_title {{
    color: {C_TEXT};
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.4px;
}}
QLabel#version_badge {{
    color: {C_TEXT_DIM};
    font-size: 10px;
    font-weight: 600;
    padding: 2px 8px;
    background-color: {C_ELEVATED};
    border: 1px solid {C_BORDER};
    border-radius: 9px;
}}
QLabel#section_label {{
    color: {C_TEXT_DIM};
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.4px;
}}
QLabel#output_label {{
    color: {C_TEXT_DIM};
    font-size: 11px;
}}
QLabel#status_label {{
    color: {C_TEXT_MID};
    font-size: 12px;
}}
QLabel#pct_label {{
    color: {C_ACCENT};
    font-size: 12px;
    font-weight: 700;
    min-width: 42px;
}}
QLabel#drop_hint {{
    color: {C_TEXT_DIM};
    font-size: 14px;
    background-color: {C_SURFACE};
    border: 2px dashed {C_BORDER};
    border-radius: 10px;
}}

/* Generic buttons */
QPushButton {{
    background-color: {C_ELEVATED};
    color: {C_TEXT};
    border: 1px solid {C_BORDER};
    border-radius: 6px;
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 500;
    min-height: 28px;
}}
QPushButton:hover {{
    background-color: #1E2340;
    border-color: #3A4060;
}}
QPushButton:pressed {{ background-color: {C_SURFACE}; }}
QPushButton:disabled {{
    background-color: {C_SURFACE};
    color: {C_TEXT_DIM};
    border-color: {C_BORDER};
}}

/* HDR tuning */
QPushButton#btn_tuning {{
    background-color: transparent;
    color: {C_TEXT_MID};
    border: none;
    border-radius: 0;
    padding: 3px 0;
    text-align: left;
    font-size: 11px;
    font-weight: 700;
}}
QPushButton#btn_tuning:hover {{ color: {C_ACCENT}; background-color: transparent; }}
QPushButton#btn_tuning:disabled {{ color: {C_TEXT_DIM}; background-color: transparent; }}
QWidget#tuning_panel {{
    background-color: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-radius: 8px;
}}
QWidget#tuning_control {{ background-color: transparent; border: none; }}
QLabel#tuning_label {{
    background-color: transparent;
    color: {C_TEXT};
    font-size: 12px;
    font-weight: 600;
}}
QLabel#tuning_hint {{
    background-color: transparent;
    color: {C_TEXT_DIM};
    font-size: 10px;
}}
QDoubleSpinBox {{
    background-color: {C_ELEVATED};
    color: {C_TEXT};
    border: 1px solid {C_BORDER};
    border-radius: 5px;
    padding: 3px 6px;
    min-width: 64px;
}}
QDoubleSpinBox:focus {{ border-color: {C_ACCENT}; }}
QDoubleSpinBox:disabled {{ color: {C_TEXT_DIM}; background-color: {C_SURFACE}; }}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    width: 14px;
    background-color: {C_CARD};
    border: none;
}}
QSlider::groove:horizontal {{
    height: 4px;
    background-color: {C_ELEVATED};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{ background-color: {C_ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 14px;
    margin: -5px 0;
    background-color: {C_TEXT};
    border: 2px solid {C_ACCENT};
    border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ background-color: #FFFFFF; }}
QSlider::handle:horizontal:disabled {{ background-color: {C_TEXT_DIM}; border-color: {C_BORDER}; }}

/* Start — accent gradient */
QPushButton#btn_start {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {C_ACCENT}, stop:1 {C_ACCENT_ALT});
    color: #0D0F18;
    border: none;
    font-size: 13px;
    font-weight: 700;
    padding: 7px 26px;
    border-radius: 7px;
    min-height: 34px;
}}
QPushButton#btn_start:hover {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #F5A840, stop:1 #F57F48);
}}
QPushButton#btn_start:pressed {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #C87820, stop:1 #D05828);
}}
QPushButton#btn_start:disabled {{
    background: {C_ELEVATED};
    color: {C_TEXT_DIM};
    border: 1px solid {C_BORDER};
}}

/* Cancel */
QPushButton#btn_cancel {{
    background-color: transparent;
    border: 1px solid {C_FAIL};
    color: {C_FAIL};
    border-radius: 7px;
    padding: 7px 18px;
    min-height: 34px;
    font-weight: 500;
}}
QPushButton#btn_cancel:hover {{ background-color: rgba(244,63,94,0.10); }}
QPushButton#btn_cancel:disabled {{
    border-color: {C_BORDER};
    color: {C_TEXT_DIM};
    background: transparent;
}}

/* File table */
QTableWidget {{
    background-color: {C_SURFACE};
    alternate-background-color: {C_CARD};
    gridline-color: {C_BORDER};
    border: 1px solid {C_BORDER};
    border-radius: 8px;
    selection-background-color: #1D2446;
    selection-color: {C_TEXT};
    outline: none;
}}
QTableWidget::item {{ padding: 5px 10px; border: none; }}
QTableWidget::item:selected {{ background-color: #1D2446; }}
QHeaderView::section {{
    background-color: {C_ELEVATED};
    color: {C_TEXT_DIM};
    border: none;
    border-right: 1px solid {C_BORDER};
    border-bottom: 1px solid {C_BORDER};
    padding: 7px 10px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.0px;
    text-transform: uppercase;
}}
QHeaderView::section:last-section {{ border-right: none; }}
QHeaderView {{ background-color: transparent; }}
QTableCornerButton::section {{
    background-color: {C_ELEVATED};
    border-right: 1px solid {C_BORDER};
    border-bottom: 1px solid {C_BORDER};
}}

/* Log console */
QTextEdit {{
    background-color: {C_SURFACE};
    color: #68D4A0;
    border: 1px solid {C_BORDER};
    border-radius: 8px;
    font-family: "Cascadia Code","JetBrains Mono","Fira Code","Consolas","Courier New",monospace;
    font-size: 11.5px;
    padding: 8px;
    selection-background-color: #1D2446;
}}

/* Progress bar */
QProgressBar {{
    background-color: {C_ELEVATED};
    border: 1px solid {C_BORDER};
    border-radius: 4px;
    min-height: 5px;
    max-height: 5px;
    color: transparent;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {C_ACCENT}, stop:1 {C_ACCENT_ALT});
    border-radius: 3px;
}}

/* Scrollbars */
QScrollBar:vertical {{
    background: transparent; width: 6px; margin: 2px 0;
}}
QScrollBar::handle:vertical {{
    background: {C_BORDER}; border-radius: 3px; min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{ background: {C_TEXT_DIM}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: transparent; height: 6px; margin: 0 2px;
}}
QScrollBar::handle:horizontal {{
    background: {C_BORDER}; border-radius: 3px; min-width: 20px;
}}
QScrollBar::handle:horizontal:hover {{ background: {C_TEXT_DIM}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
"""
