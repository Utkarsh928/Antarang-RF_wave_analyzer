"""
Comparison Dialog — side-by-side view of two loaded signals.
Fix #16: Allows users to compare spectrum, waterfall, or constellation
of the current file against a second file chosen from disk.
"""
import os
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QTabWidget, QWidget, QSplitter, QGroupBox,
    QGridLayout, QComboBox, QFrame
)
from PyQt6.QtCore import Qt

from core.signal_info import SignalInfo
from gui.theme import apply_window_chrome


class CompareDialog(QDialog):
    """Side-by-side comparison of two signal files."""

    def __init__(self, primary_info: SignalInfo,
                 primary_freqs=None, primary_power=None,
                 parent=None):
        super().__init__(parent)
        self._primary = primary_info
        self._primary_freqs = primary_freqs
        self._primary_power = primary_power
        self._secondary: SignalInfo = None
        self._sec_freqs = None
        self._sec_power = None

        self.setWindowTitle("Antarang — Compare Signals")
        self.setMinimumSize(1100, 650)
        self._theme_mode = getattr(parent, "_theme_mode", "dark")
        self._build_ui()
        self.apply_theme(self._theme_mode)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # ── Header ──────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        hdr.addWidget(QLabel(f"<b>Primary:</b>  {self._primary.file_name or '—'}"))
        hdr.addStretch()
        self._lbl_secondary = QLabel("<b>Secondary:</b>  (none loaded)")
        hdr.addWidget(self._lbl_secondary)
        self._btn_load_second = QPushButton("Load Second File…")
        self._btn_load_second.setObjectName("btn_primary")
        self._btn_load_second.clicked.connect(self._load_second)
        hdr.addWidget(self._btn_load_second)
        root.addLayout(hdr)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("separator")
        root.addWidget(sep)

        # ── Plot tabs ────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_spectrum_tab(), "Spectrum Comparison")
        self._tabs.addTab(self._build_params_tab(),   "Parameter Comparison")
        root.addWidget(self._tabs, stretch=1)

        # ── Close ────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        btn_row.addWidget(btn_close)
        root.addLayout(btn_row)

    def _build_spectrum_tab(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setSpacing(4)

        # Left plot — primary
        left_box = QGroupBox("Primary")
        ll = QVBoxLayout(left_box)
        self._plot_left = pg.PlotWidget()
        self._plot_left.setLabel('left', 'Power', units='dB')
        self._plot_left.setLabel('bottom', 'Frequency', units='Hz')
        self._plot_left.showGrid(x=True, y=True, alpha=0.3)
        self._curve_left = self._plot_left.plot(pen=pg.mkPen('#1677E8', width=1.5))
        ll.addWidget(self._plot_left)
        self._lbl_left_info = QLabel("—")
        self._lbl_left_info.setObjectName("label_key")
        ll.addWidget(self._lbl_left_info)
        layout.addWidget(left_box)

        # Right plot — secondary
        right_box = QGroupBox("Secondary")
        rl = QVBoxLayout(right_box)
        self._plot_right = pg.PlotWidget()
        self._plot_right.setLabel('left', 'Power', units='dB')
        self._plot_right.setLabel('bottom', 'Frequency', units='Hz')
        self._plot_right.showGrid(x=True, y=True, alpha=0.3)
        self._curve_right = self._plot_right.plot(pen=pg.mkPen('#38BDF8', width=1.5))
        rl.addWidget(self._plot_right)
        self._lbl_right_info = QLabel("Load a second file to compare")
        self._lbl_right_info.setObjectName("label_key")
        rl.addWidget(self._lbl_right_info)
        layout.addWidget(right_box)

        # Populate primary if we have FFT data, or compute from samples
        if (self._primary_freqs is None or self._primary_power is None) and self._primary is not None and getattr(self._primary, 'samples', None) is not None:
            try:
                from core.parameter_estimator import compute_fft
                self._primary_freqs, self._primary_power = compute_fft(self._primary.samples, self._primary.sample_rate)
            except Exception:
                pass

        if self._primary_freqs is not None and self._primary_power is not None:
            self._curve_left.setData(x=self._primary_freqs, y=self._primary_power)
            self._plot_left.autoRange()
            mod_str = self._primary.modulation if self._primary.modulation else "—"
            snr_str = f"{self._primary.snr_db:.1f} dB" if self._primary.snr_db is not None else "—"
            sr_str = f"{self._primary.sample_rate/1000:.1f} kHz" if self._primary.sample_rate else "—"
            self._lbl_left_info.setText(f"SR: {sr_str}  |  Mod: {mod_str}  |  SNR: {snr_str}")

        return w

    def apply_theme(self, mode: str):
        """Use the application's chrome while retaining blue comparison traces."""
        self._theme_mode = mode
        apply_window_chrome(self, mode)
        background, text = (('#050505', '#8E8E93') if mode == 'dark'
                            else ('#FFFFFF', '#4B5563'))
        for plot in (self._plot_left, self._plot_right):
            plot.setBackground(background)
            for axis in ('left', 'bottom'):
                plot.getAxis(axis).setPen(pg.mkPen(text))
                plot.getAxis(axis).setTextPen(pg.mkPen(text))

    def _build_params_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        self._params_grid = QGridLayout()
        layout.addLayout(self._params_grid)
        layout.addStretch()
        self._refresh_params_table()
        return w

    def _refresh_params_table(self):
        # Clear
        while self._params_grid.count():
            item = self._params_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        fields = [
            ('File',         'file_name'),
            ('Type',         'file_type'),
            ('Sample Rate',  'sample_rate_hz'),
            ('Duration',     'duration_sec'),
            ('Modulation',   'modulation'),
            ('Confidence',   'mod_confidence_pct'),
            ('SNR',          'snr_db'),
            ('Bandwidth',    'bandwidth_hz'),
            ('Bit Rate',     'bit_rate_bps'),
            ('Symbol Rate',  'symbol_rate_hz'),
            ('Band',         'band'),
        ]

        def _hdr(text):
            lbl = QLabel(f"<b>{text}</b>")
            lbl.setStyleSheet("padding: 4px;")
            return lbl

        self._params_grid.addWidget(_hdr("Parameter"), 0, 0)
        self._params_grid.addWidget(_hdr("Primary"), 0, 1)
        self._params_grid.addWidget(_hdr("Secondary"), 0, 2)
        self._params_grid.addWidget(_hdr("Difference"), 0, 3)

        p_sum = self._primary.summary()
        s_sum = self._secondary.summary() if self._secondary else {}

        for row, (label, key) in enumerate(fields, 1):
            lbl = QLabel(label + ":")
            lbl.setObjectName("label_key")
            self._params_grid.addWidget(lbl, row, 0)

            p_val = p_sum.get(key, '—')
            s_val = s_sum.get(key, '—') if s_sum else '—'

            p_lbl = QLabel(str(p_val))
            p_lbl.setStyleSheet("")
            self._params_grid.addWidget(p_lbl, row, 1)

            s_lbl = QLabel(str(s_val))
            s_lbl.setStyleSheet("")
            self._params_grid.addWidget(s_lbl, row, 2)

            # Show numeric difference where applicable
            diff_text = "—"
            diff_style = ""
            try:
                p_n, s_n = float(p_val), float(s_val)
                diff = s_n - p_n
                diff_text = f"{diff:+.3g}"
                diff_style = ""
            except (ValueError, TypeError):
                if str(p_val) == str(s_val):
                    diff_text = "same"
                    diff_style = ""

            d_lbl = QLabel(diff_text)
            d_lbl.setStyleSheet(diff_style)
            self._params_grid.addWidget(d_lbl, row, 3)

    def _load_second(self):
        """Load a second file for comparison."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Second Signal File", os.path.expanduser("~"),
            "Signal Files (*.wav *.iq *.bin *.cf32 *.cs16 *.cs8 *.cfile);;"
            "All Files (*.*)"
        )
        if not path:
            return

        try:
            from core.file_loader import load_file
            from core.parameter_estimator import estimate_parameters, compute_fft
            from core.modulation.classifier import classify_modulation
            from core.signal_filter import remove_dc, normalize_power

            info = load_file(path)
            if info.samples is not None and len(info.samples) > 0:
                info.samples = remove_dc(info.samples)
                info.samples = normalize_power(info.samples)
                info = estimate_parameters(info)
                results = classify_modulation(info.samples, sample_rate=info.sample_rate)
                if results:
                    info.modulation = results[0][0]
                    info.mod_confidence = results[0][1]

                # Compute FFT for spectrum comparison
                try:
                    freqs, power = compute_fft(info.samples, info.sample_rate)
                    self._sec_freqs = freqs
                    self._sec_power = power
                    self._curve_right.setData(x=freqs, y=power)
                    self._plot_right.autoRange()
                except Exception:
                    pass

            self._secondary = info
            self._lbl_secondary.setText(f"<b>Secondary:</b>  {info.file_name}")
            mod_text = info.modulation if info.modulation else "—"
            snr_text = f"{info.snr_db:.1f} dB" if info.snr_db is not None else "—"
            sr_text = f"{info.sample_rate/1000:.1f} kHz" if info.sample_rate else "—"
            self._lbl_right_info.setText(
                f"SR: {sr_text}  |  Mod: {mod_text}  |  SNR: {snr_text}")
            self._refresh_params_table()

        except Exception as e:
            # Graceful fallback — try load_wav directly
            try:
                from core.file_loader import load_wav
                from core.parameter_estimator import estimate_parameters, compute_fft
                from core.modulation.classifier import classify_modulation
                info = load_wav(path)
                if info.samples is not None and len(info.samples) > 0:
                    try:
                        info = estimate_parameters(info)
                        results = classify_modulation(info.samples, sample_rate=info.sample_rate)
                        if results:
                            info.modulation = results[0][0]
                            info.mod_confidence = results[0][1]
                    except Exception:
                        pass
                    try:
                        freqs, power = compute_fft(info.samples, info.sample_rate)
                        self._sec_freqs = freqs
                        self._sec_power = power
                        self._curve_right.setData(x=freqs, y=power)
                        self._plot_right.autoRange()
                    except Exception:
                        pass

                self._secondary = info
                self._lbl_secondary.setText(f"<b>Secondary:</b>  {info.file_name}")
                mod_text = info.modulation if info.modulation else "—"
                snr_text = f"{info.snr_db:.1f} dB" if info.snr_db is not None else "—"
                sr_text = f"{info.sample_rate/1000:.1f} kHz" if info.sample_rate else "—"
                self._lbl_right_info.setText(
                    f"SR: {sr_text}  |  Mod: {mod_text}  |  SNR: {snr_text}")
                self._refresh_params_table()
            except Exception as e2:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Load Error", f"Could not load file:\n{e2}")
