"""
Settings / Preferences Dialog for Tarang Signal Analyzer.
Exposes tuneable application settings to the user: Analysis, Interface, Export, SDR.
All settings are persisted via QSettings.
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTabWidget, QWidget, QFormLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QComboBox, QGroupBox, QFrame, QLineEdit,
    QDialogButtonBox
)
from PyQt6.QtCore import Qt, QSettings
from gui.theme import apply_window_chrome, window_theme_mode


# ─── Defaults ────────────────────────────────────────────────────────────────
DEFAULTS = {
    # Analysis
    "fft_size":            4096,
    "fft_window":          "Hann",
    "waterfall_history":   100,
    "max_display_samples": 512,      # MB to load
    # Export
    "bits_export_limit":   10_000,
    "export_dpi":          150,
    # UI
    "tab_after_demod":     True,
    "auto_analyze_on_load": True,
    "show_pipeline_hints":  True,
    "recent_files_count":  10,
    # SDR
    "default_sample_rate": 2_400_000,
    "default_center_freq": 100_000_000,
}


def load_setting(key: str, qsettings: QSettings = None):
    s = qsettings or QSettings("SignalAnalyzerPro", "SignalAnalyzerPro")
    return s.value(f"pref/{key}", DEFAULTS.get(key))


def save_setting(key: str, value, qsettings: QSettings = None):
    s = qsettings or QSettings("SignalAnalyzerPro", "SignalAnalyzerPro")
    s.setValue(f"pref/{key}", value)


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings = QSettings("SignalAnalyzerPro", "SignalAnalyzerPro")

        self.setWindowTitle("Antarang — Preferences")
        self.setMinimumWidth(480)
        self.setMinimumHeight(420)
        self.setStyleSheet(parent.styleSheet() if parent else "")
        apply_window_chrome(self, window_theme_mode(parent))
        self._build_ui()
        self._load_values()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)

        tabs = QTabWidget()

        # Tabs
        tabs.addTab(self._build_analysis_tab(), "Analysis")
        tabs.addTab(self._build_ui_tab(),       "Interface")
        tabs.addTab(self._build_export_tab(),   "Export")
        tabs.addTab(self._build_sdr_tab(),      "SDR Defaults")

        root.addWidget(tabs, stretch=1)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("separator")
        root.addWidget(sep)

        # Buttons
        btn_row = QHBoxLayout()
        btn_reset = QPushButton("Reset to Defaults")
        btn_reset.clicked.connect(self._reset_defaults)
        btn_row.addWidget(btn_reset)
        btn_row.addStretch()
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(self._save_and_close)
        btn_box.rejected.connect(self.reject)
        btn_row.addWidget(btn_box)
        root.addLayout(btn_row)

    def _build_analysis_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setSpacing(12)

        self._spin_fft = QSpinBox()
        self._spin_fft.setRange(256, 65536)
        self._spin_fft.setSingleStep(256)
        self._spin_fft.setToolTip("Larger = higher frequency resolution, slower")
        form.addRow("FFT Size:", self._spin_fft)

        self._combo_window = QComboBox()
        self._combo_window.addItems(["Hann", "Hamming", "Blackman",
                                      "Bartlett", "FlatTop", "Rectangular"])
        self._combo_window.setToolTip("Windowing function applied before FFT")
        form.addRow("FFT Window:", self._combo_window)

        self._spin_waterfall = QSpinBox()
        self._spin_waterfall.setRange(10, 1000)
        self._spin_waterfall.setSuffix(" rows")
        self._spin_waterfall.setToolTip("Number of time rows in the waterfall plot")
        form.addRow("Waterfall History:", self._spin_waterfall)

        self._spin_max_load = QSpinBox()
        self._spin_max_load.setRange(64, 2048)
        self._spin_max_load.setSuffix(" MB")
        self._spin_max_load.setToolTip("Maximum file data loaded into RAM for analysis")
        form.addRow("Max Load Size:", self._spin_max_load)

        return w

    def _build_ui_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setSpacing(12)

        self._chk_tab_demod = QCheckBox("Switch to Bits tab after demodulation")
        form.addRow("", self._chk_tab_demod)

        self._chk_auto_analyze = QCheckBox("Auto-analyze when file is loaded")
        form.addRow("", self._chk_auto_analyze)

        self._chk_pipeline_hints = QCheckBox("Show next-step hints in status bar")
        form.addRow("", self._chk_pipeline_hints)

        self._spin_recent = QSpinBox()
        self._spin_recent.setRange(1, 30)
        self._spin_recent.setToolTip("Max number of recent files to remember")
        form.addRow("Recent Files Count:", self._spin_recent)

        return w

    def _build_export_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setSpacing(12)

        self._spin_bits_limit = QSpinBox()
        self._spin_bits_limit.setRange(1000, 1_000_000)
        self._spin_bits_limit.setSingleStep(1000)
        self._spin_bits_limit.setToolTip("Max bits written per export file")
        form.addRow("Bits Export Limit:", self._spin_bits_limit)

        self._spin_export_dpi = QSpinBox()
        self._spin_export_dpi.setRange(72, 600)
        self._spin_export_dpi.setSuffix(" DPI")
        self._spin_export_dpi.setToolTip("PNG export resolution")
        form.addRow("PNG Export DPI:", self._spin_export_dpi)

        return w

    def _build_sdr_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setSpacing(12)

        self._spin_sdr_sr = QSpinBox()
        self._spin_sdr_sr.setRange(100_000, 100_000_000)
        self._spin_sdr_sr.setSingleStep(100_000)
        self._spin_sdr_sr.setSuffix(" Hz")
        form.addRow("Default Sample Rate:", self._spin_sdr_sr)

        self._spin_sdr_cf = QSpinBox()
        self._spin_sdr_cf.setRange(100_000, 2_147_000_000)
        self._spin_sdr_cf.setSingleStep(100_000)
        self._spin_sdr_cf.setSuffix(" Hz")
        form.addRow("Default Center Freq:", self._spin_sdr_cf)

        return w

    def _load_values(self):
        def _int(key):
            v = load_setting(key, self._settings)
            try: return int(v)
            except: return DEFAULTS[key]
        def _bool(key):
            v = load_setting(key, self._settings)
            return str(v).lower() in ('true', '1', 'yes')

        self._spin_fft.setValue(_int("fft_size"))
        self._combo_window.setCurrentText(str(load_setting("fft_window", self._settings)))
        self._spin_waterfall.setValue(_int("waterfall_history"))
        self._spin_max_load.setValue(_int("max_display_samples"))

        self._chk_tab_demod.setChecked(_bool("tab_after_demod"))
        self._chk_auto_analyze.setChecked(_bool("auto_analyze_on_load"))
        self._chk_pipeline_hints.setChecked(_bool("show_pipeline_hints"))
        self._spin_recent.setValue(_int("recent_files_count"))

        self._spin_bits_limit.setValue(_int("bits_export_limit"))
        self._spin_export_dpi.setValue(_int("export_dpi"))

        self._spin_sdr_sr.setValue(_int("default_sample_rate"))
        self._spin_sdr_cf.setValue(_int("default_center_freq"))

    def _save_and_close(self):
        save_setting("fft_size",             self._spin_fft.value(),         self._settings)
        save_setting("fft_window",           self._combo_window.currentText(), self._settings)
        save_setting("waterfall_history",    self._spin_waterfall.value(),   self._settings)
        save_setting("max_display_samples",  self._spin_max_load.value(),    self._settings)

        save_setting("tab_after_demod",      self._chk_tab_demod.isChecked(),     self._settings)
        save_setting("auto_analyze_on_load", self._chk_auto_analyze.isChecked(),  self._settings)
        save_setting("show_pipeline_hints",  self._chk_pipeline_hints.isChecked(), self._settings)
        save_setting("recent_files_count",   self._spin_recent.value(),           self._settings)

        save_setting("bits_export_limit",    self._spin_bits_limit.value(),  self._settings)
        save_setting("export_dpi",           self._spin_export_dpi.value(),  self._settings)

        save_setting("default_sample_rate",  self._spin_sdr_sr.value(),      self._settings)
        save_setting("default_center_freq",  self._spin_sdr_cf.value(),      self._settings)

        self.accept()

    def _reset_defaults(self):
        for key, val in DEFAULTS.items():
            save_setting(key, val, self._settings)
        self._load_values()
