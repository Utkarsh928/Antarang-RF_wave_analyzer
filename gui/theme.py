"""Antarang's single, application-wide visual system."""
import sys
from pathlib import Path


# Qt stylesheets handle normalized filesystem paths on Windows more reliably
# than file:/// URLs, which can be resolved relative to the working directory.
CHECKMARK_ICON = str(
    Path(__file__).resolve().parent.parent / "assets" / "icons" / "check.svg"
).replace("\\", "/")


def _theme(*, dark: bool) -> str:
    c = ({
        "window": "#080809", "surface": "#101011", "raised": "#18181A", "active": "#242426",
        "border": "#2C2C2E", "strong": "#38383A", "text": "#F2F2F7", "muted": "#8E8E93",
        "faint": "#636366", "primary": "#F2F2F7", "primary_text": "#000000", "check": "#1677E8", "selection": "#38383A",
    } if dark else {
        "window": "#F5F6F8", "surface": "#FFFFFF", "raised": "#F8F9FB", "active": "#F1F3F6",
        "border": "#D9DDE3", "strong": "#C8CDD5", "text": "#111318", "muted": "#4B5563",
        "faint": "#6B7280", "primary": "#1677E8", "primary_text": "#FFFFFF", "check": "#1677E8", "selection": "#E7F0FD",
    })
    focus = "#AEAEB2" if dark else "#1677E8"
    hover_primary = "#FFFFFF" if dark else "#1468D5"
    pipeline_surface = c["raised"] if dark else c["surface"]
    pipeline_actions = "" if dark else f"""
QWidget#pipeline_container QPushButton#btn_step, QWidget#pipeline_container QPushButton#btn_step:disabled {{ background: {c['primary']}; color: {c['primary_text']}; border-color: {c['primary']}; }}
QWidget#pipeline_container QPushButton#btn_step:hover {{ background: {hover_primary}; border-color: {hover_primary}; }}
"""
    return f"""
QMainWindow, QDialog {{ background: {c['window']}; color: {c['text']}; }}
QWidget {{ color: {c['text']}; font-family: \"Segoe UI\", \"SF Pro Display\", Arial, sans-serif; font-size: 13px; }}
QWidget#central_root {{ background: {c['window']}; }}
QToolTip, QMenu {{ background: {c['raised']}; color: {c['text']}; border: 1px solid {c['border']}; border-radius: 8px; padding: 6px 8px; }}
QWidget#toolbar_container, QWidget#toolbar_row1, QStatusBar {{ background: {c['surface']}; border-color: {c['border']}; }}
QWidget#toolbar_container {{ border-bottom: 1px solid {c['border']}; }} QWidget#toolbar_row2 {{ background: {c['raised']}; border-top: 1px solid {c['border']}; }}
QLabel#label_appname, QLabel#label_title {{ color: {c['text']}; font-weight: 600; }} QLabel#label_appname {{ font-size: 16px; letter-spacing: .4px; }} QLabel#label_title {{ font-size: 20px; }}
QLabel#label_key {{ color: {c['muted']}; font-size: 11px; font-weight: 500; background: transparent; }} QLabel#label_value {{ color: {c['text']}; font-size: 12px; background: transparent; }} QLabel#label_param_big {{ color: {c['text']}; font-size: 22px; font-weight: 600; background: transparent; }}
QLabel#status_op {{ color: {c['muted']}; font-size: 11px; }} QLabel#status_sig {{ color: {c['text']}; font-size: 11px; }} QLabel#status_time, QLabel#status_sep {{ color: {c['faint']}; font-size: 11px; }}
QSplitter::handle {{ background: {c['border']}; width: 1px; }} QSplitter::handle:hover {{ background: {c['strong']}; }}
QFrame#separator {{ background: {c['border']}; max-height: 1px; }} QFrame#workspace_separator {{ background: {c['border']}; min-height: 1px; max-height: 1px; }}
QPushButton {{ background: {c['raised']}; color: {c['text']}; border: 1px solid {c['border']}; border-radius: 10px; min-height: 30px; padding: 4px 13px; font-weight: 500; }} QPushButton:hover {{ background: {c['active']}; border-color: {c['strong']}; }} QPushButton:pressed {{ background: {c['surface']}; }} QPushButton:focus, QCheckBox:focus {{ border: 1px solid {focus}; }} QPushButton:disabled {{ background: {c['surface']}; color: {c['faint']}; border-color: {c['border']}; }}
QPushButton#btn_primary, QPushButton#btn_open, QPushButton#btn_analyze, QPushButton#btn_step {{ background: {c['primary']}; color: {c['primary_text']}; border-color: {c['primary']}; font-weight: 600; }} QPushButton#btn_primary:hover, QPushButton#btn_open:hover, QPushButton#btn_analyze:hover, QPushButton#btn_step:hover {{ background: {hover_primary}; border-color: {hover_primary}; }}
QPushButton#btn_ai, QPushButton#btn_sdr, QPushButton#btn_export, QPushButton#btn_compare, QPushButton#btn_recent, QPushButton#btn_history, QPushButton#btn_train, QPushButton#btn_about, QPushButton#btn_settings, QPushButton#btn_cancel, QPushButton#btn_sync_db, QPushButton#btn_language {{ background: transparent; color: {c['muted']}; border-color: transparent; }} QPushButton#btn_ai:hover, QPushButton#btn_sdr:hover, QPushButton#btn_export:hover, QPushButton#btn_compare:hover, QPushButton#btn_recent:hover, QPushButton#btn_history:hover, QPushButton#btn_train:hover, QPushButton#btn_about:hover, QPushButton#btn_settings:hover, QPushButton#btn_cancel:hover, QPushButton#btn_sync_db:hover, QPushButton#btn_language:hover {{ background: {c['active']}; color: {c['text']}; border-color: {c['border']}; }} QPushButton:checked {{ background: {c['selection']}; color: {c['text']}; border-color: {c['strong']}; }}
QPushButton#theme_light, QPushButton#theme_dark {{ min-width: 44px; padding: 3px 9px; }} QPushButton#theme_light:checked, QPushButton#theme_dark:checked {{ background: {c['primary']}; color: {c['primary_text']}; border-color: {c['primary']}; }}
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QAbstractSpinBox {{ background: {c['raised']}; color: {c['text']}; border: 1px solid {c['border']}; border-radius: 8px; min-height: 28px; padding: 3px 8px; selection-background-color: {c['selection']}; }} QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QAbstractSpinBox:focus {{ border-color: {focus}; }} QComboBox QAbstractItemView {{ background: {c['raised']}; color: {c['text']}; border: 1px solid {c['border']}; selection-background-color: {c['selection']}; outline: none; }}
QCheckBox {{ color: {c['muted']}; spacing: 8px; }} QCheckBox::indicator {{ width: 15px; height: 15px; border: 1px solid {c['strong']}; border-radius: 4px; background: {c['raised']}; }} QCheckBox::indicator:hover {{ border-color: {focus}; }} QCheckBox::indicator:checked {{ background: {c['check']}; border-color: {c['check']}; image: url("{CHECKMARK_ICON}"); }} QCheckBox::indicator:checked:disabled {{ background: {c['faint']}; border-color: {c['faint']}; }}
QSlider::groove:horizontal {{ height: 3px; background: {c['strong']}; border-radius: 1px; }} QSlider::handle:horizontal {{ width: 14px; margin: -6px 0; background: {c['primary']}; border-radius: 7px; }}
QGroupBox {{ background: transparent; color: {c['muted']}; border: 0; margin-top: 16px; padding-top: 6px; font-size: 10px; font-weight: 600; letter-spacing: 1.1px; }} QGroupBox#side_panel {{ background: {c['raised']}; border: 1px solid {c['border']}; border-radius: 14px; margin-top: 17px; padding: 10px 8px 8px; }} QWidget#pipeline_container QGroupBox#step_group {{ background: {pipeline_surface}; border: 1px solid {c['border']}; border-radius: 14px; margin-top: 24px; padding: 12px 8px 8px; }} QWidget#pipeline_container QGroupBox#step_group::title {{ subcontrol-origin: margin; color: {c['text']}; background: {pipeline_surface}; border-radius: 6px; font-size: 14px; font-weight: 700; letter-spacing: 0; left: 10px; top: 3px; padding: 1px 6px; }} QGroupBox#side_panel::title {{ left: 10px; padding: 0 3px; }}
QWidget#file_panel_root {{ background: {c['surface']}; border-right: 1px solid {c['border']}; }} QWidget#antarang_hero, QWidget#tarang_info_card {{ background: {c['raised']}; border: 1px solid {c['border']}; border-radius: 16px; }}
QWidget#pipeline_container, QScrollArea#pipeline_scroll, QScrollArea#pipeline_scroll::viewport, QWidget#pipeline_row {{ background: {c['window']}; }}
QWidget#pipeline_container QComboBox, QWidget#pipeline_container QAbstractSpinBox, QWidget#pipeline_container QCheckBox::indicator:unchecked {{ background: {pipeline_surface}; }}
QLabel#pipeline_arrow {{ color: {c['muted']}; font-size: 20px; font-weight: 300; background: transparent; padding: 0 2px; }}
QTabWidget::pane {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 14px; top: -1px; }} QTabBar::tab {{ color: {c['muted']}; border: none; border-bottom: 2px solid transparent; padding: 10px 14px 8px; margin-right: 4px; }} QTabBar::tab:hover {{ color: {c['text']}; background: {c['active']}; border-radius: 8px; }} QTabBar::tab:selected {{ color: {c['text']}; border-bottom-color: {c['primary']}; }}
QScrollArea {{ background: transparent; border: none; }} QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{ background: {c['strong']}; min-height: 28px; min-width: 28px; border-radius: 4px; }} QProgressBar {{ background: {c['raised']}; border: 0; border-radius: 3px; min-height: 5px; max-height: 5px; }} QProgressBar::chunk {{ background: {c['primary']}; border-radius: 3px; }}
QListWidget, QTreeWidget, QTableWidget {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 10px; color: {c['text']}; outline: none; alternate-background-color: {c['raised']}; }} QHeaderView::section {{ background: {c['raised']}; color: {c['muted']}; border: none; border-bottom: 1px solid {c['border']}; padding: 8px; font-size: 11px; font-weight: 600; }} QAbstractItemView::item:selected {{ background: {c['selection']}; color: {c['text']}; }}
QDockWidget::title {{ background: {c['raised']}; color: {c['text']}; padding: 10px; border-bottom: 1px solid {c['border']}; font-weight: 600; }} QDialogButtonBox QPushButton {{ min-width: 76px; }}
{pipeline_actions}
"""


DARK_THEME = _theme(dark=True)
LIGHT_THEME = _theme(dark=False)


def theme_stylesheet(mode: str) -> str:
    return LIGHT_THEME if mode == "light" else DARK_THEME


def apply_window_chrome(widget, mode: str) -> None:
    """Match Windows' native title bar to Antarang's active theme."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        value = ctypes.c_int(mode == "dark")
        hwnd = int(widget.winId())
        # 20 is current Windows 10/11; 19 keeps older Windows 10 builds aligned.
        for attribute in (20, 19):
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, attribute, ctypes.byref(value), ctypes.sizeof(value))
    except (AttributeError, OSError):
        pass


def window_theme_mode(parent) -> str:
    """Read the main window's mode for newly created Antarang dialogs."""
    window = parent.window() if parent else None
    return getattr(window, "_theme_mode", "dark")
