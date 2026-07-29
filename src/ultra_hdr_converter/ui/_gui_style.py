"""Visual design tokens and QSS for the Ultra HDR Converter desktop UI."""

from __future__ import annotations

C_BG: str = "#101112"
C_SURFACE: str = "#17191B"
C_ELEVATED: str = "#1E2023"
C_CARD: str = "#222529"
C_BORDER: str = "#303338"

C_ACCENT: str = "#F5A524"
C_ACCENT_ALT: str = "#FFCA55"

C_TEXT: str = "#F4F3EF"
C_TEXT_MID: str = "#A9ADB3"
C_TEXT_DIM: str = "#737880"

C_OK: str = "#55C98A"
C_FAIL: str = "#FF6B73"
C_PROCESSING: str = "#70B7FF"
C_CANCELLED: str = "#F2B84B"

STATUS_COLORS: dict[str, str] = {
    "OK": C_OK,
    "Failed": C_FAIL,
    "Processing": C_PROCESSING,
    "Queued": C_TEXT_DIM,
    "Skipped": C_CANCELLED,
    "Cancelled": C_CANCELLED,
}

STYLESHEET: str = f"""
/* Application shell */
QMainWindow, QWidget {{
    background-color: {C_BG};
    color: {C_TEXT};
    font-family: "Aptos", "Segoe UI Variable", "Segoe UI", sans-serif;
    font-size: 13px;
}}
QWidget#central, QWidget#body, QWidget#brand, QWidget#output_group,
QWidget#queue_section, QWidget#log_section, QWidget#tuning_control {{
    background-color: transparent;
}}

/* Brand header */
QWidget#header {{
    background-color: #141517;
    border-bottom: 1px solid {C_BORDER};
    min-height: 70px;
    max-height: 70px;
}}
QLabel#app_icon {{ background-color: transparent; }}
QLabel#app_title {{
    background-color: transparent;
    color: {C_TEXT};
    font-size: 17px;
    font-weight: 700;
}}
QLabel#app_subtitle {{
    background-color: transparent;
    color: {C_TEXT_DIM};
    font-size: 11px;
}}
QLabel#version_badge {{
    color: {C_ACCENT_ALT};
    font-size: 10px;
    font-weight: 700;
    padding: 4px 9px;
    background-color: #292317;
    border: 1px solid #4B3B1B;
    border-radius: 6px;
}}

/* Section typography */
QLabel#section_title {{
    background-color: transparent;
    color: {C_TEXT};
    font-size: 13px;
    font-weight: 700;
}}
QLabel#section_meta, QLabel#output_caption {{
    background-color: transparent;
    color: {C_TEXT_DIM};
    font-size: 10px;
    font-weight: 700;
}}
QLabel#output_label {{
    background-color: transparent;
    color: {C_TEXT_MID};
    font-size: 11px;
}}
QLabel#status_label {{
    background-color: transparent;
    color: {C_TEXT_MID};
    font-size: 12px;
}}
QLabel#pct_label {{
    background-color: transparent;
    color: {C_ACCENT_ALT};
    font-size: 12px;
    font-weight: 700;
    min-width: 40px;
}}

/* Toolbar */
QWidget#toolbar {{
    background-color: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-radius: 8px;
}}
QPushButton {{
    background-color: {C_ELEVATED};
    color: {C_TEXT_MID};
    border: 1px solid {C_BORDER};
    border-radius: 6px;
    padding: 6px 13px;
    font-size: 12px;
    font-weight: 600;
    min-height: 28px;
}}
QPushButton:hover {{
    background-color: #292C30;
    color: {C_TEXT};
    border-color: #484C52;
}}
QPushButton:pressed {{ background-color: #121416; }}
QPushButton:focus {{ border-color: {C_ACCENT}; }}
QPushButton:disabled {{
    background-color: #181A1C;
    color: #5C6066;
    border-color: #282B2F;
}}
QPushButton#btn_add {{
    background-color: #2D281D;
    color: {C_ACCENT_ALT};
    border-color: #55451F;
}}
QPushButton#btn_add:hover {{
    background-color: #3A311E;
    color: #FFD478;
    border-color: #82652B;
}}
QPushButton#btn_output {{ padding-left: 15px; padding-right: 15px; }}
QPushButton#btn_quiet {{
    background-color: transparent;
    border-color: transparent;
    color: {C_TEXT_DIM};
    padding: 2px 8px;
    min-height: 22px;
}}
QPushButton#btn_quiet:hover {{ color: {C_TEXT}; background-color: {C_ELEVATED}; }}

/* HDR tuning */
QPushButton#btn_tuning {{
    background-color: transparent;
    color: {C_TEXT_MID};
    border: none;
    border-radius: 4px;
    padding: 4px 2px;
    text-align: left;
    font-size: 12px;
    font-weight: 700;
}}
QPushButton#btn_tuning:hover {{ color: {C_ACCENT_ALT}; background-color: transparent; }}
QPushButton#btn_tuning:checked {{ color: {C_ACCENT_ALT}; }}
QPushButton#btn_tuning:disabled {{ color: {C_TEXT_DIM}; background-color: transparent; }}
QWidget#tuning_panel {{
    background-color: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-radius: 8px;
}}
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
    padding: 4px 7px;
    min-width: 66px;
    selection-background-color: #664710;
}}
QDoubleSpinBox:focus {{ border-color: {C_ACCENT}; }}
QDoubleSpinBox:disabled {{ color: {C_TEXT_DIM}; background-color: {C_SURFACE}; }}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    width: 15px;
    background-color: {C_CARD};
    border: none;
}}
QSlider::groove:horizontal {{
    height: 4px;
    background-color: #303338;
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

/* Empty queue */
QWidget#drop_zone {{
    background-color: {C_SURFACE};
    border: 1px dashed #44484E;
    border-radius: 8px;
}}
QLabel#drop_hint {{
    background-color: transparent;
    color: {C_TEXT};
    font-size: 15px;
    font-weight: 700;
}}
QLabel#drop_detail {{
    background-color: transparent;
    color: {C_TEXT_DIM};
    font-size: 11px;
}}

/* File table */
QTableWidget {{
    background-color: {C_SURFACE};
    alternate-background-color: #1B1D20;
    gridline-color: transparent;
    border: 1px solid {C_BORDER};
    border-radius: 8px;
    selection-background-color: #332C1E;
    selection-color: {C_TEXT};
    outline: none;
}}
QTableWidget::item {{
    padding: 7px 10px;
    border-bottom: 1px solid #24272A;
}}
QTableWidget::item:selected {{ background-color: #332C1E; }}
QHeaderView::section {{
    background-color: {C_ELEVATED};
    color: {C_TEXT_DIM};
    border: none;
    border-bottom: 1px solid {C_BORDER};
    padding: 8px 10px;
    font-size: 10px;
    font-weight: 700;
}}
QHeaderView {{ background-color: transparent; }}
QTableCornerButton::section {{
    background-color: {C_ELEVATED};
    border: none;
    border-bottom: 1px solid {C_BORDER};
}}

/* Activity log */
QTextEdit {{
    background-color: #131516;
    color: #9FCBAD;
    border: 1px solid {C_BORDER};
    border-radius: 8px;
    font-family: "Cascadia Mono", "Cascadia Code", Consolas, monospace;
    font-size: 11px;
    padding: 9px;
    selection-background-color: #3B3526;
}}

/* Conversion footer */
QWidget#bottom_bar {{
    background-color: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-radius: 8px;
}}
QPushButton#btn_start {{
    background-color: {C_ACCENT};
    color: #17130C;
    border: 1px solid {C_ACCENT};
    font-size: 13px;
    font-weight: 800;
    padding: 7px 24px;
    border-radius: 6px;
    min-height: 34px;
}}
QPushButton#btn_start:hover {{
    background-color: {C_ACCENT_ALT};
    border-color: {C_ACCENT_ALT};
}}
QPushButton#btn_start:pressed {{ background-color: #D78A12; border-color: #D78A12; }}
QPushButton#btn_start:disabled {{
    background-color: {C_ELEVATED};
    color: {C_TEXT_DIM};
    border-color: {C_BORDER};
}}
QPushButton#btn_cancel {{
    background-color: transparent;
    border: 1px solid #6D3438;
    color: #FF8A91;
    border-radius: 6px;
    padding: 7px 17px;
    min-height: 34px;
}}
QPushButton#btn_cancel:hover {{ background-color: #321D20; border-color: {C_FAIL}; }}
QPushButton#btn_cancel:disabled {{
    border-color: {C_BORDER};
    color: {C_TEXT_DIM};
    background-color: transparent;
}}
QProgressBar {{
    background-color: #2A2D31;
    border: none;
    border-radius: 3px;
    min-height: 6px;
    max-height: 6px;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: {C_ACCENT};
    border-radius: 3px;
}}

/* Scrollbars */
QScrollBar:vertical {{ background: transparent; width: 7px; margin: 2px 0; }}
QScrollBar::handle:vertical {{ background: #3A3D42; border-radius: 3px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: #555A61; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 7px; margin: 0 2px; }}
QScrollBar::handle:horizontal {{ background: #3A3D42; border-radius: 3px; min-width: 24px; }}
QScrollBar::handle:horizontal:hover {{ background: #555A61; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
"""
