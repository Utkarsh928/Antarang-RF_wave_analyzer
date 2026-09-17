"""
Main Window — Signal Analyzer Pro
Full application window with file loading, tabbed plots, pipeline controls,
AI overview, and export.
"""
import os
import time
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QPushButton, QLabel, QProgressBar,
    QFileDialog, QMessageBox, QTabWidget, QDockWidget,
    QToolBar, QStatusBar, QDialog, QComboBox,
    QDoubleSpinBox, QFormLayout, QDialogButtonBox,
    QApplication, QStackedWidget, QFrame, QSizePolicy, QButtonGroup,
    QToolButton, QMenu
)
from PyQt6.QtCore import Qt, QTimer, QSize, QSettings, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QIcon, QAction, QDragEnterEvent, QDropEvent, QKeySequence, QShortcut, QFont

from gui.theme import theme_stylesheet, apply_window_chrome
from gui.branding import logo_pixmap, window_icon
from gui.icons import icon
from gui.language_manager import LM
from gui.language_selector import LanguageSelectorWidget
from gui.drop_overlay import DropOverlay
from gui.widgets.file_panel import FilePanel
from gui.widgets.spectrum_widget import SpectrumWidget
from gui.widgets.waterfall_widget import WaterfallWidget
from gui.widgets.constellation_widget import ConstellationWidget
from gui.widgets.bits_widget import BitsWidget
from gui.widgets.protocol_widget import ProtocolWidget
from gui.widgets.control_panel import ControlPanel
from gui.ai_overview import AIOverviewPanel
from gui.workers import (AnalysisWorker, DemodulateWorker,
                          DeinterleaveWorker, FECWorker,
                          CorrelateWorker, PlotDataWorker)
from gui.history_panel import HistoryPanel   # Fix #19
from core.signal_info import SignalInfo


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._signal_info: SignalInfo = SignalInfo()
        self._active_worker = None
        self._plot_freqs = None
        self._plot_power = None
        self._spec_freqs = None
        self._spec_times = None
        self._spec_power = None
        self._settings = QSettings("SignalAnalyzerPro", "SignalAnalyzerPro")
        self._theme_mode = str(self._settings.value("ui/theme", "dark"))
        self._op_start_time: float = 0.0   # for timing display

        self.setWindowTitle("Antarang — Signal Analysis")
        self.setMinimumSize(900, 600)
        self.resize(1400, 850)
        self.setAcceptDrops(True)
        stylesheet = theme_stylesheet(self._theme_mode)
        QApplication.instance().setStyleSheet(stylesheet)
        self.setStyleSheet(stylesheet)
        apply_window_chrome(self, self._theme_mode)

        self._build_ui()
        self._apply_theme(self._theme_mode, persist=False)
        self._connect_signals()
        self._setup_shortcuts()
        self._restore_settings()
        # Wire live language switching
        LM.language_changed.connect(self._retranslate_ui)
        self._update_status(LM.t("ready_msg"))

    # ─────────────────────────────────────────────────────────────────────
    # UI Construction
    # ─────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        central.setObjectName("central_root")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Two-row Toolbar ───────────────────────────────────────────────
        toolbar_container = QWidget()
        toolbar_container.setObjectName("toolbar_container")
        # No fixed/max height — let rows size themselves naturally
        toolbar_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        tc_layout = QVBoxLayout(toolbar_container)
        tc_layout.setContentsMargins(0, 0, 0, 0)
        tc_layout.setSpacing(0)
        tc_layout.addWidget(self._build_toolbar_row1())
        tc_layout.addWidget(self._build_toolbar_row2())
        root.addWidget(toolbar_container)

        # ── Thin progress bar ─────────────────────────────────────────────
        prog_row = QHBoxLayout()
        prog_row.setContentsMargins(8, 2, 8, 2)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.hide()
        prog_row.addWidget(self._progress, stretch=1)

        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setObjectName("btn_cancel")
        self._btn_cancel.setToolTip("Cancel the current operation")
        self._btn_cancel.clicked.connect(self._cancel_worker)
        self._btn_cancel.hide()
        prog_row.addWidget(self._btn_cancel)

        prog_widget = QWidget()
        prog_widget.setLayout(prog_row)
        prog_widget.setMaximumHeight(22)
        root.addWidget(prog_widget)

        # ── Thin separator ────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("separator")
        root.addWidget(sep)

        # ── Main splitter ─────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(6)

        # Left file panel — responsive width
        self._file_panel = FilePanel()
        self._file_panel.setObjectName("file_panel_root")
        self._file_panel.setMinimumWidth(160)
        self._file_panel.setMaximumWidth(260)
        self._file_panel.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        splitter.addWidget(self._file_panel)

        # Right: stacked (drop overlay + content)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # Tabs — placed directly, no overlay container needed
        self._plot_tabs = QTabWidget()
        self._plot_tabs.setTabPosition(QTabWidget.TabPosition.North)
        self._plot_tabs.currentChanged.connect(self._on_tab_changed)

        self._spectrum_widget      = SpectrumWidget()
        self._waterfall_widget     = WaterfallWidget()
        self._constellation_widget = ConstellationWidget()
        self._bits_widget          = BitsWidget()
        self._protocol_widget      = ProtocolWidget()

        self._plot_tabs.addTab(self._spectrum_widget,      LM.t("tabs.spectrum"))
        self._plot_tabs.addTab(self._waterfall_widget,     LM.t("tabs.waterfall"))
        self._plot_tabs.addTab(self._constellation_widget, LM.t("tabs.constellation"))
        self._plot_tabs.addTab(self._bits_widget,          LM.t("tabs.bits"))
        self._plot_tabs.addTab(self._protocol_widget,      LM.t("tabs.protocol"))

        # ── Empty-state hint overlay (shown until first file is loaded) ──
        from PyQt6.QtWidgets import QStackedWidget
        self._plot_stack = QStackedWidget()

        # Page 0: empty-state hint
        _empty_page = QWidget()
        _empty_layout = QVBoxLayout(_empty_page)
        _empty_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_mark = QLabel("∿")
        empty_mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_mark.setStyleSheet("color: #AEAEB2; font-size: 54px; font-weight: 300; background: transparent;")
        _empty_layout.addWidget(empty_mark)
        self._lbl_empty_hint = QLabel(
            "Ready to analyze\n\n"
            "Open a signal file or start a live SDR capture.\n"
            "You can also drag an IQ or WAV file into this workspace.")
        self._lbl_empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_empty_hint.setStyleSheet(
            "color: #AEAEB2; font-size: 15px; font-weight: 400; "
            "background: transparent; padding: 12px;")
        self._lbl_empty_hint.setWordWrap(True)
        _empty_layout.addWidget(self._lbl_empty_hint)
        empty_actions = QHBoxLayout()
        empty_actions.setSpacing(10)
        self._btn_empty_open = QPushButton("Open Signal")
        self._btn_empty_open.setObjectName("btn_primary")
        self._btn_empty_sdr = QPushButton("Live Capture")
        empty_actions.addStretch()
        empty_actions.addWidget(self._btn_empty_open)
        empty_actions.addWidget(self._btn_empty_sdr)
        empty_actions.addStretch()
        _empty_layout.addLayout(empty_actions)

        # Page 1: real tabs
        self._plot_stack.addWidget(_empty_page)          # index 0 — no file
        self._plot_stack.addWidget(self._plot_tabs)      # index 1 — file loaded
        self._plot_stack.setCurrentIndex(0)              # start on empty page

        right_layout.addWidget(self._plot_stack, stretch=1)

        workspace_separator = QFrame()
        workspace_separator.setObjectName("workspace_separator")
        workspace_separator.setFrameShape(QFrame.Shape.HLine)
        workspace_separator.setFixedHeight(1)
        right_layout.addWidget(workspace_separator)

        # Pipeline control panel
        self._control_panel = ControlPanel()
        right_layout.addWidget(self._control_panel)

        splitter.addWidget(right_widget)
        # Responsive initial sizes: file panel ~200px, rest fills
        splitter.setSizes([200, 9999])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, stretch=1)

        # ── 3-zone Status Bar ─────────────────────────────────────────────
        self._status_bar = QStatusBar()
        self._status_bar.setFixedHeight(24)
        self.setStatusBar(self._status_bar)
        self._build_status_bar()

        # ── AI Overview Dock ──────────────────────────────────────────────
        self._ai_panel = AIOverviewPanel()
        self._ai_dock  = QDockWidget(LM.t("ai_panel.title"), self)
        self._ai_dock.setWidget(self._ai_panel)
        self._ai_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea |
            Qt.DockWidgetArea.BottomDockWidgetArea)
        self._ai_dock.setMinimumWidth(380)
        self._ai_dock.hide()
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._ai_dock)

        # ── History Dock ──────────────────────────────────────────────────
        self._history_panel = HistoryPanel()
        self._history_panel.sig_reopen.connect(self._load_file)
        self._history_dock = QDockWidget("Session History", self)
        self._history_dock.setWidget(self._history_panel)
        self._history_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea |
            Qt.DockWidgetArea.RightDockWidgetArea |
            Qt.DockWidgetArea.BottomDockWidgetArea)
        self._history_dock.setMinimumWidth(300)
        self._history_dock.hide()
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._history_dock)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_toolbar_overflow()

    # ─────────────────────────────────────────────────────────────────────
    # Toolbar Row 1 — File ops + title + feature toggles
    # ─────────────────────────────────────────────────────────────────────
    def _build_toolbar_row1(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("toolbar_row1")
        bar.setMinimumHeight(44)
        bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(8, 3, 8, 3)
        lay.setSpacing(5)

        # ── LEFT group: file operations ───────────────────────────────────
        self._btn_open = QPushButton(LM.t("toolbar.open_file"))
        self._btn_open.setObjectName("btn_open")
        self._btn_open.setToolTip("Open .wav or .iq file  (Ctrl+O)")
        self._btn_open.setMinimumWidth(100)
        self._btn_open.setMaximumHeight(32)
        lay.addWidget(self._btn_open)

        self._btn_recent = QPushButton("Recent")
        self._btn_recent.setObjectName("btn_recent")
        self._btn_recent.setToolTip("Recently opened files")
        self._btn_recent.setMaximumHeight(32)
        self._btn_recent.clicked.connect(self._show_recent_files_menu)
        lay.addWidget(self._btn_recent)

        self._btn_sdr = QPushButton(LM.t("toolbar.live_sdr"))
        self._btn_sdr.setObjectName("btn_sdr")
        self._btn_sdr.setToolTip("Capture from RTL-SDR / HackRF / Simulated")
        self._btn_sdr.setMaximumHeight(32)
        lay.addWidget(self._btn_sdr)

        self._btn_analyze = QPushButton(LM.t("toolbar.analyze"))
        self._btn_analyze.setObjectName("btn_analyze")
        self._btn_analyze.setToolTip("Run full signal analysis  (Ctrl+R)\n"
                                     "Load a file first to enable this button.")
        self._btn_analyze.setMaximumHeight(32)
        self._btn_analyze.setMinimumWidth(80)
        self._btn_analyze.setEnabled(False)
        lay.addWidget(self._btn_analyze)

        # ── CENTER: app title, equal stretch on both sides ────────────────
        lay.addStretch(1)
        title_group = QWidget()
        title_layout = QHBoxLayout(title_group)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(0)
        self._title_logo = QLabel()
        self._title_logo.setFixedSize(112, 32)
        self._title_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_layout.addWidget(self._title_logo)
        lay.addWidget(title_group)
        lay.addStretch(1)

        # ── RIGHT group: features, consistent spacing ─────────────────────
        right_group = QWidget()
        right_layout = QHBoxLayout(right_group)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(5)
        self._toolbar_right_group = right_group

        self._lang_selector = LanguageSelectorWidget()
        self._lang_selector.setMaximumHeight(32)
        right_layout.addWidget(self._lang_selector)

        self._btn_export = QPushButton(LM.t("toolbar.export"))
        self._btn_export.setObjectName("btn_export")
        self._btn_export.setToolTip("Export results and plots  (Ctrl+E)\n"
                                    "Analyze a file first to enable this button.")
        self._btn_export.setMaximumHeight(32)
        self._btn_export.setEnabled(False)
        right_layout.addWidget(self._btn_export)

        self._btn_compare = QPushButton("Compare")
        self._btn_compare.setObjectName("btn_compare")
        self._btn_compare.setToolTip("Side-by-side comparison of two signal files\n"
                                     "Analyze a file first to enable this button.")
        self._btn_compare.setMaximumHeight(32)
        self._btn_compare.setEnabled(False)
        right_layout.addWidget(self._btn_compare)

        self._btn_ai = QPushButton(LM.t("toolbar.ai_overview"))
        self._btn_ai.setObjectName("btn_ai")
        self._btn_ai.setToolTip("AI assistant panel  (Ctrl+I)")
        self._btn_ai.setMaximumHeight(32)
        self._btn_ai.setCheckable(True)
        right_layout.addWidget(self._btn_ai)

        self._btn_history = QPushButton("History")
        self._btn_history.setObjectName("btn_history")
        self._btn_history.setToolTip("Session signal history  (Ctrl+H)")
        self._btn_history.setMaximumHeight(32)
        self._btn_history.setCheckable(True)
        right_layout.addWidget(self._btn_history)

        theme_group = QWidget()
        theme_lay = QHBoxLayout(theme_group)
        theme_lay.setContentsMargins(4, 0, 0, 0)
        theme_lay.setSpacing(0)
        self._btn_theme_light = QPushButton("Light")
        self._btn_theme_light.setObjectName("theme_light")
        self._btn_theme_dark = QPushButton("Dark")
        self._btn_theme_dark.setObjectName("theme_dark")
        for button in (self._btn_theme_light, self._btn_theme_dark):
            button.setCheckable(True)
            button.setMaximumHeight(28)
            theme_lay.addWidget(button)
        self._theme_buttons = QButtonGroup(self)
        self._theme_buttons.setExclusive(True)
        self._theme_buttons.addButton(self._btn_theme_light)
        self._theme_buttons.addButton(self._btn_theme_dark)
        (self._btn_theme_light if self._theme_mode == "light" else self._btn_theme_dark).setChecked(True)
        self._btn_theme_light.clicked.connect(lambda: self._set_theme("light"))
        self._btn_theme_dark.clicked.connect(lambda: self._set_theme("dark"))
        right_layout.addWidget(theme_group)
        lay.addWidget(right_group)

        self._toolbar_overflow = QToolButton()
        self._toolbar_overflow.setObjectName("toolbar_overflow")
        self._toolbar_overflow.setText("⋮")
        self._toolbar_overflow.setToolTip("More actions")
        self._toolbar_overflow.setPopupMode(
            QToolButton.ToolButtonPopupMode.InstantPopup)
        self._toolbar_overflow.setMenu(QMenu(self._toolbar_overflow))
        self._build_toolbar_overflow_menu()
        self._toolbar_overflow.hide()
        lay.addWidget(self._toolbar_overflow)
        self._apply_toolbar_icons()

        return bar

    def _build_toolbar_overflow_menu(self):
        menu = self._toolbar_overflow.menu()
        language = menu.addAction("Language")
        language.triggered.connect(self._lang_selector._open_dialog)
        menu.addSeparator()
        for button in (
            self._btn_export, self._btn_compare, self._btn_ai,
            self._btn_history,
        ):
            action = menu.addAction(button.text())
            action.triggered.connect(button.click)
        menu.addSeparator()
        light = menu.addAction("Light")
        light.triggered.connect(self._btn_theme_light.click)
        dark = menu.addAction("Dark")
        dark.triggered.connect(self._btn_theme_dark.click)

    def _update_toolbar_overflow(self):
        if not hasattr(self, "_toolbar_overflow"):
            return
        compact = self.width() < 1180
        self._toolbar_right_group.setVisible(not compact)
        self._toolbar_overflow.setVisible(compact)

    # ─────────────────────────────────────────────────────────────────────
    # Toolbar Row 2 — ML status + settings + about
    # ─────────────────────────────────────────────────────────────────────
    def _build_toolbar_row2(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("toolbar_row2")
        bar.setMinimumHeight(30)
        bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(8, 2, 8, 2)
        lay.setSpacing(6)

        # ML status
        self._lbl_model = QLabel(LM.t("toolbar.ml_not_trained"))
        self._lbl_model.setStyleSheet(
            "color: #8E8E93; font-size: 11px; padding: 0 4px; background: transparent;")
        self._lbl_model.setToolTip("ML modulation classifier status")
        lay.addWidget(self._lbl_model)

        self._btn_train_model = QPushButton(LM.t("toolbar.train_ml"))
        self._btn_train_model.setObjectName("btn_train")
        self._btn_train_model.setToolTip(
            "Train ML classifier — improves modulation detection accuracy to ~88%")
        self._btn_train_model.setMaximumHeight(28)
        lay.addWidget(self._btn_train_model)

        lay.addStretch()

        # About + Settings
        self._btn_about = QPushButton("About")
        self._btn_about.setObjectName("btn_about")
        self._btn_about.setToolTip("About Signal Analyzer Pro")
        self._btn_about.setMaximumHeight(28)
        lay.addWidget(self._btn_about)

        self._btn_settings = QPushButton("Settings")
        self._btn_settings.setObjectName("btn_settings")
        self._btn_settings.setToolTip("Settings & Preferences")
        self._btn_settings.setMaximumHeight(28)
        self._btn_settings.clicked.connect(self._open_settings)
        lay.addWidget(self._btn_settings)
        self._apply_toolbar_icons()

        return bar
    # ─────────────────────────────────────────────────────────────────────
    def _build_status_bar(self):
        # Zone 1 — current operation message
        self._lbl_status_op = QLabel(LM.t("ready_msg"))
        self._lbl_status_op.setObjectName("status_op")
        self._status_bar.addWidget(self._lbl_status_op, 3)

        # Separator
        sep1 = QLabel("|")
        sep1.setObjectName("status_sep")
        self._status_bar.addWidget(sep1)

        # Zone 2 — live signal parameters after load
        self._lbl_status_sig = QLabel("")
        self._lbl_status_sig.setObjectName("status_sig")
        self._status_bar.addWidget(self._lbl_status_sig, 2)

        # Separator
        sep2 = QLabel("|")
        sep2.setObjectName("status_sep")
        self._status_bar.addWidget(sep2)

        # Zone 3 — operation timing (right-aligned)
        self._lbl_status_time = QLabel("No operation run yet")
        self._lbl_status_time.setObjectName("status_time")
        self._status_bar.addPermanentWidget(self._lbl_status_time)

    def _connect_signals(self):
        self._btn_open.clicked.connect(self._open_file_dialog)
        self._btn_empty_open.clicked.connect(self._open_file_dialog)
        self._btn_empty_sdr.clicked.connect(self._open_sdr_dialog)
        self._btn_sdr.clicked.connect(self._open_sdr_dialog)
        self._btn_analyze.clicked.connect(self._run_analysis)
        self._btn_export.clicked.connect(self._export_results)
        self._btn_compare.clicked.connect(self._open_compare_dialog)
        self._btn_ai.toggled.connect(self._toggle_ai_panel)
        self._btn_history.toggled.connect(self._toggle_history_panel)
        self._btn_about.clicked.connect(self._show_about)
        self._btn_train_model.clicked.connect(self._open_train_dialog)
        # Drop zone in file panel — clicking it opens the file dialog
        self._file_panel.drop_zone.sig_open_file.connect(self._open_file_dialog)

        self._control_panel.sig_demodulate.connect(self._run_demodulate)
        self._control_panel.sig_deinterleave.connect(self._run_deinterleave)
        self._control_panel.sig_fec_decode.connect(self._run_fec)
        self._control_panel.sig_correlate.connect(self._run_correlate)
        self._control_panel.sig_protocol_analysis.connect(self._run_protocol_analysis)

    def _setup_shortcuts(self):
        """Keyboard shortcuts for power users."""
        QShortcut(QKeySequence("Ctrl+O"), self).activated.connect(
            self._open_file_dialog)
        QShortcut(QKeySequence("Ctrl+R"), self).activated.connect(
            self._run_analysis)
        QShortcut(QKeySequence("Ctrl+E"), self).activated.connect(
            self._export_results)
        QShortcut(QKeySequence("Ctrl+I"), self).activated.connect(
            lambda: self._btn_ai.setChecked(not self._btn_ai.isChecked()))
        QShortcut(QKeySequence("Ctrl+H"), self).activated.connect(  # Fix #19
            lambda: self._btn_history.setChecked(not self._btn_history.isChecked()))
        QShortcut(QKeySequence("1"), self).activated.connect(
            lambda: self._plot_tabs.setCurrentIndex(0))
        QShortcut(QKeySequence("2"), self).activated.connect(
            lambda: self._plot_tabs.setCurrentIndex(1))
        QShortcut(QKeySequence("3"), self).activated.connect(
            lambda: self._plot_tabs.setCurrentIndex(2))
        QShortcut(QKeySequence("4"), self).activated.connect(
            lambda: self._plot_tabs.setCurrentIndex(3))
        QShortcut(QKeySequence("5"), self).activated.connect(
            lambda: self._plot_tabs.setCurrentIndex(4))
        QShortcut(QKeySequence("F11"), self).activated.connect(
            self._toggle_fullscreen)

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _restore_settings(self):
        """Restore window geometry and last folder from previous session."""
        geom = self._settings.value("geometry")
        if geom:
            self.restoreGeometry(geom)
        # Update ML model status after UI is built
        QTimer.singleShot(100, self._update_model_status)

    def _set_theme(self, mode: str):
        """Switch the shared stylesheet without rebuilding the analysis workspace."""
        self._apply_theme(mode)

    def _apply_theme(self, mode: str, *, persist: bool = True):
        """Apply the stored mode to application chrome and custom-painted widgets."""
        self._theme_mode = mode
        (self._btn_theme_light if mode == "light" else self._btn_theme_dark).setChecked(True)
        stylesheet = theme_stylesheet(mode)
        QApplication.instance().setStyleSheet(stylesheet)
        self.setStyleSheet(stylesheet)
        apply_window_chrome(self, mode)
        if persist:
            self._settings.setValue("ui/theme", mode)
        self._apply_toolbar_icons()
        self._ai_panel.apply_theme(mode)
        self._file_panel.apply_theme(mode)
        self._control_panel.apply_theme()
        for plot in (self._spectrum_widget, self._waterfall_widget, self._constellation_widget,
                     self._bits_widget):
            plot.apply_theme(mode)
        self._protocol_widget.apply_theme(mode)

    def _apply_toolbar_icons(self):
        color = "#111318" if self._theme_mode == "light" else "#F2F2F7"
        primary = "#FFFFFF" if self._theme_mode == "light" else "#000000"
        self.setWindowIcon(window_icon(self._theme_mode))
        if hasattr(self, "_title_logo"):
            self._title_logo.setPixmap(logo_pixmap(self._theme_mode, 112, 32))
        for button, name, tone in (
            (getattr(self, "_btn_open", None), "open", primary),
            (getattr(self, "_btn_recent", None), "recent", color),
            (getattr(self, "_btn_sdr", None), "radio", color),
            (getattr(self, "_btn_export", None), "export", color),
            (getattr(self, "_btn_compare", None), "compare", color),
            (getattr(self, "_btn_ai", None), "brain", color),
            (getattr(self, "_btn_history", None), "history", color),
            (getattr(self, "_btn_train_model", None), "train", color),
            (getattr(self, "_btn_about", None), "info", color),
            (getattr(self, "_btn_settings", None), "settings", color),
        ):
            if button:
                button.setIcon(icon(name, tone))
                button.setIconSize(QSize(16, 16))

    def closeEvent(self, event):
        """Save settings on close."""
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("last_folder",
                                 self._settings.value("last_folder", ""))
        # Stop any running worker
        if self._active_worker and self._active_worker.isRunning():
            self._active_worker.quit()
            self._active_worker.wait(2000)
        super().closeEvent(event)

    # ─────────────────────────────────────────────────────────────────────
    # Drag & Drop
    # ─────────────────────────────────────────────────────────────────────
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            path = url.toLocalFile()
            if path.lower().endswith(('.wav', '.iq', '.bin', '.cf32',
                                       '.cs16', '.cs8', '.cfile')):
                event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        url = event.mimeData().urls()[0]
        path = url.toLocalFile()
        self._add_to_recent(path)   # Fix #15
        self._load_file(path)

    def _open_train_dialog(self):
        """Open the ML model training dialog."""
        from gui.train_dialog import TrainingDialog
        dlg = TrainingDialog(self)
        dlg.model_trained.connect(self._on_model_trained)
        dlg.exec()

    def _on_model_trained(self):
        """Called after successful model training — update UI."""
        self._update_model_status()
        # Reset classifier cache so new model is loaded
        try:
            import core.modulation.classifier as clf_mod
            clf_mod._ml_model_data = None
            clf_mod._ml_model_checked = False
        except Exception:
            pass
        self._update_status("ML model trained — classifier accuracy improved")

    def _update_model_status(self):
        """Update the ML model status indicator in toolbar."""
        try:
            from core.modulation.ml_classifier import get_model_info
            info = get_model_info()
            if info.get('available'):
                n_cls = info.get('n_classes', 0)
                mb = info.get('size_mb', 0)
                self._lbl_model.setText(f"ML model · {n_cls} classes")
                self._lbl_model.setStyleSheet(
                    "color: #AEAEB2; font-size: 11px; padding: 0 6px;")
                self._lbl_model.setToolTip(
                    f"ML model active — {n_cls} modulation classes, "
                    f"{mb:.1f} MB\nClick Train ML to retrain")
                self._btn_train_model.setText(LM.t("toolbar.retrain_ml"))
            else:
                self._lbl_model.setText(LM.t("toolbar.ml_not_trained"))
                self._lbl_model.setStyleSheet(
                    "color: #8E8E93; font-size: 11px; padding: 0 6px;")
                self._btn_train_model.setText(LM.t("toolbar.train_ml"))
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────
    # Live retranslation — called when user switches language
    # ─────────────────────────────────────────────────────────────────────
    def _retranslate_ui(self):
        """Update every hardcoded string in MainWindow without restart."""
        self.setWindowTitle("Antarang — Signal Analysis")
        self._btn_open.setText(LM.t("toolbar.open_file"))
        self._btn_sdr.setText(LM.t("toolbar.live_sdr"))
        self._btn_analyze.setText(LM.t("toolbar.analyze"))
        self._btn_export.setText(LM.t("toolbar.export"))
        self._btn_ai.setText(LM.t("toolbar.ai_overview"))
        self._btn_about.setText("About")
        self._apply_toolbar_icons()
        self._plot_tabs.setTabText(0, LM.t("tabs.spectrum"))
        self._plot_tabs.setTabText(1, LM.t("tabs.waterfall"))
        self._plot_tabs.setTabText(2, LM.t("tabs.constellation"))
        self._plot_tabs.setTabText(3, LM.t("tabs.bits"))
        self._plot_tabs.setTabText(4, LM.t("tabs.protocol"))
        self._ai_dock.setWindowTitle("AI Overview")
        self._update_model_status()
        self._update_status(LM.t("ready_msg"))
        self._file_panel.retranslate()
        self._control_panel.retranslate()

    def _show_about(self):
        dialog = QMessageBox(self)
        dialog.setWindowTitle("About Antarang")
        dialog.setIconPixmap(logo_pixmap(self._theme_mode, 120, 34))
        dialog.setText(
            ""
            "<p>Automated .IQ and .WAV signal analysis platform.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Auto-detect WAV and IQ file formats</li>"
            "<li>FFT Spectrum, Waterfall, Constellation plots</li>"
            "<li>ML classifier — 95% accuracy at SNR ≥ 10dB</li>"
            "<li>FSK / PSK / QAM demodulation</li>"
            "<li>Block / Conv / Diagonal / Pseudo-random de-interleaving</li>"
            "<li>Viterbi / Reed-Solomon / LDPC / Concatenated FEC</li>"
            "<li>Bit stream correlation (header / payload)</li>"
            "<li>Protocol & Security analysis</li>"
            "<li>AI Overview with multi-provider intelligence & offline LLaVA 3B</li>"
            "<li>13 Indian languages — live switching</li>"
            "<li>Recent files, comparison view, settings</li>"
            "</ul>"
            "<p><b>Keyboard Shortcuts:</b><br>"
            "Ctrl+O: Open file &nbsp;&nbsp; Ctrl+R: Analyze<br>"
            "Ctrl+E: Export &nbsp;&nbsp;&nbsp; Ctrl+I: AI Panel<br>"
            "1-5: Switch tabs &nbsp; F11: Fullscreen</p>"
            "<p><small>Built with Python 3.11 · PyQt6 · pyqtgraph · "
            "numpy · scipy · scikit-learn · Tarang AI Engine</small></p>"
        )
        dialog.exec()

    def _open_settings(self):
        """Open settings/preferences dialog."""
        from gui.settings_dialog import SettingsDialog
        dlg = SettingsDialog(self)
        if dlg.exec():
            if hasattr(self, '_ai_panel'):
                self._ai_panel.refresh_status()

    # ─────────────────────────────────────────────────────────────────────
    # File Loading
    # ─────────────────────────────────────────────────────────────────────
    def _open_sdr_dialog(self):
        """Open live SDR capture dialog."""
        from gui.sdr_dialog import SDRDialog
        dlg = SDRDialog(self)
        dlg.capture_done.connect(self._load_sdr_samples)
        dlg.exec()

    def _load_sdr_samples(self, samples: np.ndarray,
                           sample_rate: float, center_freq: float):
        """
        Handle live-captured IQ samples from SDR dialog.
        Creates a SignalInfo and runs the full analysis pipeline —
        identical to loading a file, just the source is live hardware.
        """
        from core.signal_info import SignalInfo
        from core.parameter_estimator import estimate_parameters
        from core.modulation.classifier import classify_modulation
        from core.signal_filter import remove_dc, normalize_power

        self._reset_ui()
        self._update_status(
            f"SDR capture: {len(samples):,} samples  "
            f"SR={sample_rate/1e6:.2f} MHz  CF={center_freq/1e6:.3f} MHz")
        self._show_progress(True)
        self._on_progress(20, "Processing captured IQ samples...")

        # Build SignalInfo from captured samples
        info = SignalInfo()
        info.file_name    = f"live_sdr_{center_freq/1e6:.3f}MHz.iq"
        info.file_type    = "IQ"
        info.sample_rate  = float(sample_rate)
        info.center_freq_hz = float(center_freq)
        info.num_samples  = len(samples)
        info.duration_sec = len(samples) / sample_rate
        info.dtype        = "complex64"
        info.channels     = 2
        info.file_size_bytes = len(samples) * 8

        # Preprocess
        samples = remove_dc(samples)
        samples = normalize_power(samples)
        info.samples = samples

        # Estimate parameters
        self._on_progress(40, "Estimating parameters...")
        known_cf = info.center_freq_hz   # preserve hardware-set center freq
        info = estimate_parameters(info)
        if known_cf > 0:
            info.center_freq_hz = known_cf   # don't overwrite hardware-set CF

        # Classify modulation
        self._on_progress(70, "Classifying modulation...")
        results = classify_modulation(samples)
        if results:
            info.modulation    = results[0][0]
            info.mod_confidence = results[0][1]
            info.mod_top3      = results[:3]

        # Hand off to the standard analysis callback
        self._on_analysis_done(info)

    def _open_file_dialog(self):
        last_folder = self._settings.value("last_folder",
                                            os.path.expanduser("~"))
        path, _ = QFileDialog.getOpenFileName(
            self,
            LM.t("dialogs.open_signal_file"),
            last_folder,
            "Signal Files (*.wav *.iq *.bin *.cf32 *.cs16 *.cs8 *.cfile);;"
            "WAV Files (*.wav);;"
            "IQ Files (*.iq *.bin *.cf32 *.cs16 *.cs8 *.cfile);;"
            "All Files (*.*)"
        )
        if path:
            self._settings.setValue("last_folder",
                                     os.path.dirname(path))
            self._add_to_recent(path)   # Fix #15
            self._load_file(path)

    def _load_file(self, path: str):
        from core.file_loader import detect_file_type
        from core.chunked_loader import get_file_info
        file_type = detect_file_type(path)
        file_size = os.path.getsize(path)

        if file_type == 'IQ':
            params = self._ask_iq_params()
            if params is None:
                return
            iq_dtype, iq_sr = params
        else:
            iq_dtype = 'float32'
            iq_sr = 0.0

        # Large file warning
        if file_size > 500 * 1024 * 1024:
            mb = file_size / (1024 * 1024)
            reply = QMessageBox.question(
                self,
                LM.t("dialogs.large_file_title"),
                LM.t("dialogs.large_file_msg", mb=mb),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._reset_ui()
        self._update_status(LM.t("status.loading", filename=os.path.basename(path)))
        self._show_progress(True)

        worker = AnalysisWorker(path, iq_dtype, iq_sr, self)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_analysis_done)
        worker.error.connect(self._on_worker_error)
        self._active_worker = worker
        worker.start()

    def _ask_iq_params(self):
        """Dialog to ask user for IQ file parameters."""
        dlg = IQParamsDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            return dlg.get_params()
        return None

    # ── Fix #15: Recent files ─────────────────────────────────────────────
    _MAX_RECENT = 10

    def _add_to_recent(self, path: str):
        """Store path at the front of the recent-files list."""
        recent = self._settings.value("recent_files", []) or []
        if isinstance(recent, str):
            recent = [recent]
        # Remove duplicates, put at front, cap at max
        recent = [p for p in recent if p != path]
        recent.insert(0, path)
        recent = recent[:self._MAX_RECENT]
        self._settings.setValue("recent_files", recent)

    def _show_recent_files_menu(self):
        """Show a pop-up menu of recently opened files under the button."""
        from PyQt6.QtWidgets import QMenu
        recent = self._settings.value("recent_files", []) or []
        if isinstance(recent, str):
            recent = [recent]
        menu = QMenu(self)
        menu.setStyleSheet(self.styleSheet())
        if not recent:
            menu.addAction("(No recent files)").setEnabled(False)
        else:
            for path in recent:
                display = os.path.basename(path)
                action = menu.addAction(f"  {display}")
                action.setToolTip(path)
                action.setData(path)
            menu.addSeparator()
            menu.addAction("Clear Recent Files").triggered.connect(
                lambda: self._settings.remove("recent_files"))
        chosen = menu.exec(
            self._btn_recent.mapToGlobal(
                self._btn_recent.rect().bottomLeft()))
        if chosen and chosen.data():
            self._load_file(chosen.data())

    # ─────────────────────────────────────────────────────────────────────
    # Analysis
    # ─────────────────────────────────────────────────────────────────────
    def _run_analysis(self):
        if not self._signal_info.is_loaded():
            return
        self._update_status(LM.t("status.computing_plots"))
        self._show_progress(True)

        worker = PlotDataWorker(self._signal_info, self)
        worker.finished.connect(self._on_plot_data_ready)
        worker.error.connect(self._on_worker_error)
        self._active_worker = worker
        worker.start()

    def _on_analysis_done(self, info: SignalInfo):
        self._signal_info = info
        self._file_panel.update_file_info(info)
        self._file_panel.update_parameters(info)
        self._control_panel.enable_demodulate(True)
        if info.modulation:
            self._control_panel.set_detected_modulation(info.modulation)

        # Switch plot area from empty-state page to real tabs
        self._plot_stack.setCurrentIndex(1)

        # Update AI with analysis data
        self._ai_panel.set_analysis_data(info.summary())

        # Fix #19: record in session history
        self._history_panel.add_entry(info)

        self._show_progress(False)
        self._btn_analyze.setEnabled(True)
        self._btn_export.setEnabled(True)
        self._btn_compare.setEnabled(True)   # Fix #16

        # Hide drop zone in file panel now that a file is loaded
        self._file_panel.drop_zone.hide_overlay()

        # Populate zone 2 status bar with key signal params
        self._update_status_sig(
            f"SR {info.sample_rate/1000:.0f} kHz  "
            f"| {info.modulation or '—'}  "
            f"| SNR {info.snr_db:.1f} dB  "
            f"| {info.duration_sec:.2f}s"
        )
        self._update_status(
            LM.t("status.loaded",
                 filename=info.file_name,
                 sr=f"{info.sample_rate/1000:.1f}",
                 dur=f"{info.duration_sec:.3f}",
                 mod=info.modulation,
                 conf=f"{info.mod_confidence*100:.0f}")
        )

        # Auto-run plots
        self._run_analysis()

    def _on_plot_data_ready(self, freqs, power_db,
                             spec_freqs, spec_times, spec_power):
        self._plot_freqs = freqs
        self._plot_power = power_db
        self._spec_freqs = spec_freqs
        self._spec_times = spec_times
        self._spec_power = spec_power

        # Update spectrum
        self._spectrum_widget.plot(
            freqs, power_db,
            self._signal_info.center_freq_hz,
            self._signal_info.bandwidth_hz,
            self._signal_info.snr_db
        )

        # Update waterfall
        self._waterfall_widget.plot(spec_freqs, spec_times, spec_power)

        # Update constellation if complex
        if np.iscomplexobj(self._signal_info.samples):
            self._constellation_widget.plot(
                self._signal_info.samples,
                self._signal_info.modulation
            )

        self._show_progress(False)
        self._update_status(LM.t("status.analysis_complete"))

    # ─────────────────────────────────────────────────────────────────────
    # Pipeline steps
    # ─────────────────────────────────────────────────────────────────────
    def _run_demodulate(self, modulation: str):
        if not self._signal_info.is_loaded():
            return
        self._file_panel.set_pipeline_status('demod', 'running')
        self._show_progress(True)
        self._update_status(LM.t("status.demodulating"))

        worker = DemodulateWorker(self._signal_info, modulation, self)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_demod_done)
        worker.error.connect(self._on_worker_error)
        self._active_worker = worker
        worker.start()

    def _on_demod_done(self, info: SignalInfo):
        self._signal_info = info
        self._file_panel.update_parameters(info)
        self._file_panel.set_pipeline_status('demod', 'done')
        self._control_panel.enable_deinterleave(True)
        self._control_panel.enable_fec(True)
        self._control_panel.enable_correlate(True)
        self._show_progress(False)

        if info.raw_bits is not None:
            self._bits_widget.set_raw_bits(info.raw_bits, info.bit_rate_bps)
            self._plot_tabs.setCurrentIndex(3)  # Switch to Bits tab

        self._ai_panel.set_analysis_data(info.summary())
        self._update_status(
            LM.t("status.demodulated",
                 bits=f"{len(info.raw_bits):,}",
                 bps=f"{info.bit_rate_bps:.0f}")
            + "  →  Next: De-interleave or FEC Decode"   # Fix #10
        )

    def _run_deinterleave(self, method: str, rows: int, cols: int):
        if self._signal_info.raw_bits is None:
            QMessageBox.warning(self, LM.t("dialogs.no_bits_title"),
                                LM.t("dialogs.no_bits_msg"))
            return
        self._file_panel.set_pipeline_status('deinter', 'running')
        self._show_progress(True)
        self._update_status(LM.t("status.deinterleaving"))

        worker = DeinterleaveWorker(self._signal_info, method, rows,
                                     cols, self)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_deinter_done)
        worker.error.connect(self._on_worker_error)
        self._active_worker = worker
        worker.start()

    def _on_deinter_done(self, info: SignalInfo):
        self._signal_info = info
        self._file_panel.set_pipeline_status('deinter', 'done')
        self._show_progress(False)
        self._ai_panel.set_analysis_data(info.summary())
        self._update_status(
            LM.t("status.deinterleaved",
                 method=info.interleave_type,
                 bits=f"{len(info.deinterleaved_bits):,}")
            + "  →  Next: FEC Decode or Correlate"   # Fix #10
        )

    def _run_fec(self, fec_type: str):
        bits = (self._signal_info.deinterleaved_bits
                if self._signal_info.deinterleaved_bits is not None
                else self._signal_info.raw_bits)
        if bits is None:
            QMessageBox.warning(self, LM.t("dialogs.no_bits_title"),
                                LM.t("dialogs.no_bits_msg"))
            return
        self._file_panel.set_pipeline_status('fec', 'running')
        self._show_progress(True)
        self._update_status(LM.t("status.fec_decoding"))

        worker = FECWorker(self._signal_info, fec_type, self)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_fec_done)
        worker.error.connect(self._on_worker_error)
        self._active_worker = worker
        worker.start()

    def _on_fec_done(self, info: SignalInfo):
        self._signal_info = info
        self._file_panel.update_parameters(info)
        self._file_panel.set_pipeline_status('fec', 'done')
        self._show_progress(False)

        if info.corrected_bits is not None:
            self._bits_widget.set_corrected_bits(
                info.corrected_bits,
                info.fec_errors_detected,
                info.fec_errors_corrected
            )

        self._ai_panel.set_analysis_data(info.summary())
        self._update_status(
            LM.t("status.fec_done",
                 type=info.fec_type,
                 det=info.fec_errors_detected,
                 corr=info.fec_errors_corrected)
            + "  →  Next: Correlate"   # Fix #10
        )

    def _run_correlate(self, sync_name: str):
        bits = (self._signal_info.corrected_bits
                if self._signal_info.corrected_bits is not None
                else self._signal_info.deinterleaved_bits
                if self._signal_info.deinterleaved_bits is not None
                else self._signal_info.raw_bits)
        if bits is None:
            QMessageBox.warning(self, LM.t("dialogs.no_bits_title"),
                                LM.t("dialogs.no_bits_msg"))
            return
        self._file_panel.set_pipeline_status('correlate', 'running')
        self._show_progress(True)
        self._update_status(LM.t("status.correlating"))

        worker = CorrelateWorker(self._signal_info, sync_name, self)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_correlate_done)
        worker.error.connect(self._on_worker_error)
        self._active_worker = worker
        worker.start()

    def _on_correlate_done(self, info: SignalInfo):
        self._signal_info = info
        self._file_panel.set_pipeline_status('correlate', 'done')
        self._show_progress(False)

        self._bits_widget.set_correlation_result(
            info.header_offset,
            info.payload_offset,
            info.sync_word_name
        )

        self._ai_panel.set_analysis_data(info.summary())
        self._update_status(
            LM.t("status.correlated",
                 header=info.header_offset,
                 payload=info.payload_offset)
            + (f"  [{info.sync_word_name}]" if info.sync_word_name else "")
            + "  →  Next: Analyze Protocol"   # Fix #10
        )
        
        # Enable protocol analysis after correlation
        self._control_panel.enable_protocol_analysis(True)

    # ─────────────────────────────────────────────────────────────────────
    # Protocol Analysis (Category 3)
    # ─────────────────────────────────────────────────────────────────────
    def _run_protocol_analysis(self, auto_detect: bool, protocol_hint: str):
        """Run protocol analysis and security detection."""
        if not self._signal_info.is_loaded():
            return
        
        # Check if we have bits
        bits = (self._signal_info.corrected_bits 
                if self._signal_info.corrected_bits is not None
                else self._signal_info.deinterleaved_bits
                if self._signal_info.deinterleaved_bits is not None
                else self._signal_info.raw_bits)
        
        if bits is None:
            QMessageBox.warning(self, LM.t("dialogs.no_bits_title"),
                                LM.t("dialogs.no_bits_msg"))
            return
        
        self._file_panel.set_pipeline_status('protocol', 'running')
        self._show_progress(True)
        self._update_status(LM.t("status.protocol_analyzing"))

        from gui.workers import ProtocolAnalysisWorker
        worker = ProtocolAnalysisWorker(self._signal_info, auto_detect, 
                                        protocol_hint, self)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_protocol_done)
        worker.error.connect(self._on_worker_error)
        self._active_worker = worker
        worker.start()

    def _on_protocol_done(self, info: SignalInfo):
        """Handle protocol analysis completion."""
        self._signal_info = info
        self._file_panel.set_pipeline_status('protocol', 'done')
        self._show_progress(False)

        # Update protocol widget
        self._protocol_widget.set_protocol_results(info)
        
        # Switch to Protocol tab to show results
        self._plot_tabs.setCurrentIndex(4)  # Protocol is 5th tab (index 4)

        self._ai_panel.set_analysis_data(info.summary())
        
        # Build status message
        protocol = getattr(info, 'protocol_type', 'Unknown')
        confidence = getattr(info, 'protocol_confidence', 0.0)
        num_frames = getattr(info, 'num_frames', 0)
        crypto = getattr(info, 'crypto_detected', False)
        
        status_msg = f"Protocol: {protocol} ({confidence*100:.1f}%)  |  " \
                     f"Frames: {num_frames}"
        if crypto:
            status_msg += "  |  Encryption Detected"
        
        self._update_status(status_msg)

    # ─────────────────────────────────────────────────────────────────────
    # Export
    # ─────────────────────────────────────────────────────────────────────
    def _export_results(self):
        if not self._signal_info.is_loaded():
            return
        from gui.exporter import ExportDialog
        dlg = ExportDialog(self._signal_info,
                           self._spectrum_widget,
                           self._waterfall_widget,
                           self._constellation_widget,
                           protocol_widget=self._protocol_widget,  # Fix #9
                           parent=self)
        dlg.exec()

    def _open_compare_dialog(self):
        """Fix #16: Open side-by-side comparison dialog."""
        if not self._signal_info.is_loaded():
            return
        from gui.compare_dialog import CompareDialog
        dlg = CompareDialog(
            self._signal_info,
            primary_freqs=self._plot_freqs,
            primary_power=self._plot_power,
            parent=self
        )
        dlg.exec()

    # ─────────────────────────────────────────────────────────────────────
    # AI Panel
    # ─────────────────────────────────────────────────────────────────────
    def _toggle_ai_panel(self, checked: bool):
        if checked:
            self._ai_dock.show()
        else:
            self._ai_dock.hide()

    def _toggle_history_panel(self, checked: bool):
        """Fix #19: Show/hide history dock."""
        if checked:
            self._history_dock.show()
        else:
            self._history_dock.hide()

    def _on_tab_changed(self, index: int):
        # Guard — AI panel may not be built yet during init
        if not hasattr(self, '_ai_panel'):
            return
        tab_names = ['Spectrum', 'Waterfall', 'Constellation', 'Bits', 'Protocol']
        name = tab_names[index] if index < len(tab_names) else ''
        self._ai_panel._current_tab = name

        # Capture current plot as image bytes for llava vision
        image_bytes = self._capture_tab_image(index)
        self._ai_panel.set_current_plot_image(name, image_bytes)

    def _capture_tab_image(self, index: int) -> bytes:
        """Capture current tab widget as PNG bytes for AI vision.
        Supports all 5 tabs including Protocol (index 4).
        """
        import tempfile, os
        try:
            widgets = [
                self._spectrum_widget,      # 0
                self._waterfall_widget,     # 1
                self._constellation_widget, # 2
                self._bits_widget,          # 3
                self._protocol_widget,      # 4  ← Fix #2: Protocol now captured
            ]
            if index >= len(widgets):
                return None
            widget = widgets[index]

            # For pyqtgraph plot widgets use the exporter (higher quality)
            if index in (0, 1, 2) and hasattr(widget, '_plot_widget'):
                try:
                    import pyqtgraph as pg
                    import pyqtgraph.exporters
                    exporter = pg.exporters.ImageExporter(
                        widget._plot_widget.plotItem)
                    exporter.parameters()['width'] = 800
                    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    tmp.close()
                    exporter.export(tmp.name)
                    with open(tmp.name, 'rb') as f:
                        data = f.read()
                    os.unlink(tmp.name)
                    return data
                except Exception:
                    pass

            # Fallback: Qt screenshot of the widget (works for Bits + Protocol)
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            tmp.close()
            pixmap = widget.grab()
            pixmap.save(tmp.name, 'PNG')
            with open(tmp.name, 'rb') as f:
                data = f.read()
            os.unlink(tmp.name)
            return data if data else None
        except Exception:
            return None

    # ─────────────────────────────────────────────────────────────────────
    # Workers / Progress
    # ─────────────────────────────────────────────────────────────────────
    def _on_progress(self, value: int, message: str):
        self._progress.setValue(value)
        if message:
            self._update_status(message)

    def _on_worker_error(self, error_msg: str):
        self._show_progress(False)
        # Mark any running pipeline step as error
        for step in ['demod', 'deinter', 'fec', 'correlate']:
            lbl = self._file_panel._status_labels.get(step)
            if lbl and "Running" in lbl.text():
                self._file_panel.set_pipeline_status(step, 'error')
        QMessageBox.critical(self, "Error", error_msg)
        self._update_status(f"Error: {error_msg}")

    def _show_progress(self, visible: bool):
        self._progress.setVisible(visible)
        self._btn_cancel.setVisible(visible)
        if visible:
            self._progress.setValue(0)
            self._op_start_time = time.time()
        else:
            # Show elapsed time in zone 3
            if self._op_start_time > 0:
                elapsed = time.time() - self._op_start_time
                self._lbl_status_time.setText(f"⏱ {elapsed:.1f}s")
            self._op_start_time = 0.0

    def _update_status(self, message: str):
        """Zone 1 — operation message."""
        self._lbl_status_op.setText(message)

    def _cancel_worker(self):
        """Cancel the currently running background worker."""
        if self._active_worker and self._active_worker.isRunning():
            self._active_worker.quit()
            self._active_worker.wait(1000)
            for step in ['demod', 'deinter', 'fec', 'correlate', 'protocol']:
                lbl = self._file_panel._status_labels.get(step)
                if lbl:
                    text = lbl.text()
                    if "Running" in text:
                        self._file_panel.set_pipeline_status(step, 'error')
        self._show_progress(False)
        self._update_status("Operation cancelled")

    def _update_status_sig(self, text: str):
        """Zone 2 — signal parameter summary."""
        self._lbl_status_sig.setText(text)

    # ─────────────────────────────────────────────────────────────────────
    # Reset
    # ─────────────────────────────────────────────────────────────────────
    def _reset_ui(self):
        self._signal_info = SignalInfo()
        self._file_panel.reset()
        self._control_panel.reset()
        self._spectrum_widget.clear()
        self._waterfall_widget.clear()
        self._constellation_widget.clear()
        self._bits_widget.clear()
        self._protocol_widget.clear()
        self._btn_analyze.setEnabled(False)
        self._btn_export.setEnabled(False)
        self._btn_compare.setEnabled(False)
        self._file_panel.drop_zone.show_overlay()
        # Return to empty-state page when no file is loaded
        self._plot_stack.setCurrentIndex(0)
        self._update_status_sig("")
        self._lbl_status_time.setText("")


# ─────────────────────────────────────────────────────────────────────────────
# IQ Parameters Dialog
# ─────────────────────────────────────────────────────────────────────────────
class IQParamsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(LM.t("dialogs.iq_params_title"))
        self.setMinimumWidth(320)
        self.setStyleSheet(parent.styleSheet() if parent else "")
        apply_window_chrome(self, getattr(parent, "_theme_mode", "dark"))
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        info_lbl = QLabel(
            "IQ files have no standard header.\n"
            "Please specify the data format:")
        info_lbl.setWordWrap(True)
        layout.addWidget(info_lbl)

        form = QFormLayout()

        self._dtype_combo = QComboBox()
        self._dtype_combo.addItems(['float32', 'int16', 'int8', 'complex64'])
        self._dtype_combo.setToolTip(
            "float32: GNU Radio default\n"
            "int16: RTL-SDR, HackRF\n"
            "int8: some SDR formats\n"
            "complex64: numpy complex")
        form.addRow("Data Type:", self._dtype_combo)

        self._sr_spin = QDoubleSpinBox()
        self._sr_spin.setRange(0, 100_000_000)
        self._sr_spin.setValue(2_400_000)
        self._sr_spin.setSuffix(" Hz")
        self._sr_spin.setSingleStep(100_000)
        self._sr_spin.setToolTip("Sample rate in Hz (e.g. 2400000 for 2.4MHz)")
        form.addRow("Sample Rate:", self._sr_spin)

        layout.addLayout(form)

        # Hint
        hint = QLabel(
            "Common values:\n"
            "• RTL-SDR: int8 or int16, 2.4 MHz\n"
            "• GNU Radio: float32, varies\n"
            "• HackRF: int8, up to 20 MHz")
        hint.setStyleSheet("color: #AEAEB2; font-size: 11px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_params(self):
        return self._dtype_combo.currentText(), self._sr_spin.value()
