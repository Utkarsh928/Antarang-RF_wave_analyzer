"""
Spectrum Widget — FFT power vs frequency plot using pyqtgraph.
"""
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from gui.theme import apply_window_chrome


class SpectrumWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        pg.setConfigOptions(antialias=True)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Info bar
        info_layout = QHBoxLayout()
        self._lbl_peak = QLabel("Peak: —")
        self._lbl_peak.setObjectName("label_key")
        self._lbl_peak.setToolTip("Frequency of the strongest spectral component")
        self._lbl_snr = QLabel("SNR: —")
        self._lbl_snr.setObjectName("label_key")
        self._lbl_snr.setToolTip("Signal-to-noise ratio in dB")
        self._lbl_bw = QLabel("BW: —")
        self._lbl_bw.setObjectName("label_key")
        self._lbl_bw.setToolTip("Estimated signal bandwidth")
        for lbl in [self._lbl_peak, self._lbl_snr, self._lbl_bw]:
            info_layout.addWidget(lbl)
        info_layout.addStretch()

        self._btn_reset_zoom = QPushButton("Reset Zoom")
        self._btn_reset_zoom.setObjectName("btn_reset_zoom")
        self._btn_reset_zoom.setMaximumHeight(22)
        self._btn_reset_zoom.setToolTip("Reset zoom to fit all data (right-click plot for more options)")
        self._btn_reset_zoom.clicked.connect(self._reset_zoom)
        info_layout.addWidget(self._btn_reset_zoom)
        layout.addLayout(info_layout)

        # Plot widget
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setLabel('left', 'Power', units='dB')
        self._plot_widget.setLabel('bottom', 'Frequency', units='Hz')
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.setMenuEnabled(True)
        self._average_action = self._plot_widget.getViewBox().menu.addAction("Average")
        self._average_action.triggered.connect(self._average_spectrum)
        self.apply_theme('dark')
        self._install_export_dialog_theme_hook()
        self._plot_widget.setTitle(
            "<span style='color:#636366;font-size:13px'>"
            "Load a signal file to view the spectrum</span>")
        self._has_data = False
        self._freqs = None
        self._power_db = None

        # Main spectrum curve
        self._curve = self._plot_widget.plot(
            pen=pg.mkPen(color='#38BDF8', width=1.7))

        # Peak marker
        self._peak_line = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(color='#F59E0B', width=1, style=Qt.PenStyle.DashLine))
        self._plot_widget.addItem(self._peak_line)

        # Bandwidth region
        self._bw_region = pg.LinearRegionItem(
            values=[0, 0],
            brush=pg.mkBrush(color=(56, 189, 248, 24)),
            pen=pg.mkPen(color='#38BDF8', width=1,
                         style=Qt.PenStyle.DashLine),
            movable=False)
        self._plot_widget.addItem(self._bw_region)

        layout.addWidget(self._plot_widget)

    def _install_export_dialog_theme_hook(self):
        """Apply the application theme to pyqtgraph's built-in export dialog."""
        scene = self._plot_widget.scene()
        original_show_export_dialog = scene.showExportDialog

        def show_export_dialog():
            original_show_export_dialog()
            self._style_export_dialog(scene.exportDialog)

        export_action = scene.contextMenu[0]
        export_action.triggered.disconnect(original_show_export_dialog)
        export_action.triggered.connect(show_export_dialog)
        scene.showExportDialog = show_export_dialog

    def _style_export_dialog(self, dialog):
        if dialog is None:
            return
        mode = getattr(self, "_theme_mode", "dark")
        if mode == "light":
            window, surface, active = "#F5F6F8", "#FFFFFF", "#F1F3F6"
            text, muted, border = "#111318", "#4B5563", "#D9DDE3"
        else:
            window, surface, active = "#080809", "#101011", "#242426"
            text, muted, border = "#F2F2F7", "#8E8E93", "#2C2C2E"

        dialog.setStyleSheet(f"""
            QWidget {{ background: {window}; color: {text}; }}
            QLabel {{ background: transparent; color: {text}; }}
            QTreeWidget, QListWidget, QTreeView, QAbstractItemView {{
                background: {surface};
                color: {text};
                border: 1px solid {border};
                selection-background-color: {active};
                selection-color: {text};
            }}
            QHeaderView::section {{
                background: {surface};
                color: {muted};
                border: none;
            }}
            QPushButton {{
                background: {surface};
                color: {text};
                border: 1px solid {border};
                border-radius: 8px;
                padding: 4px 10px;
            }}
            QPushButton:hover {{
                background: {active};
                border-color: {border};
            }}
        """)
        palette = dialog.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(window))
        palette.setColor(QPalette.ColorRole.Base, QColor(surface))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(active))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(text))
        palette.setColor(QPalette.ColorRole.Text, QColor(text))
        dialog.setPalette(palette)
        dialog.setAutoFillBackground(True)
        apply_window_chrome(dialog, mode)

    def apply_theme(self, mode: str):
        """Adapt plot chrome only; RF trace colours intentionally stay fixed."""
        self._theme_mode = mode
        bg, text = (('#050505', '#8E8E93') if mode == 'dark' else ('#FFFFFF', '#4B5563'))
        self._plot_widget.setBackground(bg)
        for axis in ('left', 'bottom'):
            self._plot_widget.getAxis(axis).setPen(pg.mkPen(text))
            self._plot_widget.getAxis(axis).setTextPen(pg.mkPen(text))
        scene = self._plot_widget.scene()
        if getattr(scene, "exportDialog", None) is not None:
            self._style_export_dialog(scene.exportDialog)

    def plot(self, freqs: np.ndarray, power_db: np.ndarray,
             center_freq: float = 0.0, bandwidth: float = 0.0,
             snr_db: float = 0.0):
        """Update the spectrum plot with new data."""
        if not self._has_data:
            self._plot_widget.setTitle("")
            self._has_data = True
        self._freqs = np.asarray(freqs)
        self._power_db = np.asarray(power_db)
        self._curve.setData(x=self._freqs, y=self._power_db)

        # Update peak marker
        peak_idx = np.argmax(self._power_db)
        peak_freq = float(self._freqs[peak_idx])
        peak_pow = float(self._power_db[peak_idx])
        self._peak_line.setValue(center_freq if center_freq != 0 else peak_freq)

        # Update bandwidth region
        if bandwidth > 0:
            cf = center_freq if center_freq != 0 else peak_freq
            self._bw_region.setRegion([cf - bandwidth / 2,
                                        cf + bandwidth / 2])

        # Auto-range
        self._plot_widget.autoRange()

        self._lbl_peak.setText(f"Peak: {_format_freq(peak_freq)}  ({peak_pow:.1f} dB)")
        self._lbl_snr.setText(f"SNR: {snr_db:.1f} dB")
        self._lbl_bw.setText(f"BW: {_format_freq(bandwidth)}")

    def _average_spectrum(self):
        """Smooth the displayed trace without losing its source data or zoom."""
        if self._freqs is None or self._power_db is None:
            return
        averaged = self._average_power(self._power_db)
        self._curve.setData(x=self._freqs, y=averaged)
        peak_idx = np.argmax(averaged)
        self._lbl_peak.setText(
            f"Peak: {_format_freq(float(self._freqs[peak_idx]))}  "
            f"({float(averaged[peak_idx]):.1f} dB)")

    @staticmethod
    def _average_power(power_db: np.ndarray) -> np.ndarray:
        """Centered moving average with stable spectrum edges."""
        if len(power_db) < 3:
            return power_db.copy()
        width = min(9, len(power_db) if len(power_db) % 2 else len(power_db) - 1)
        edge = width // 2
        return np.convolve(np.pad(power_db, edge, mode='edge'),
                           np.full(width, 1 / width), mode='valid')

    def clear(self):
        self._curve.setData(x=[], y=[])
        self._freqs = None
        self._power_db = None
        self._peak_line.setValue(0)
        self._bw_region.setRegion([0, 0])
        self._lbl_peak.setText("Peak: —")
        self._lbl_snr.setText("SNR: —")
        self._lbl_bw.setText("BW: —")
        self._has_data = False
        self._plot_widget.setTitle(
            "<span style='color:#636366;font-size:13px'>"
            "Load a signal file to view the spectrum</span>")

    def _reset_zoom(self):
        """Fix #17: Reset zoom to fit all data."""
        self._plot_widget.autoRange()

    def get_plot_image(self) -> bytes:
        """Export current plot as PNG bytes via temp file."""
        import tempfile
        import os
        try:
            import pyqtgraph.exporters
            exporter = pg.exporters.ImageExporter(self._plot_widget.plotItem)
            exporter.parameters()['width'] = 1200
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            tmp.close()
            exporter.export(tmp.name)
            with open(tmp.name, 'rb') as f:
                data = f.read()
            os.unlink(tmp.name)
            return data
        except Exception:
            # Qt screenshot fallback
            try:
                tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                tmp.close()
                pixmap = self._plot_widget.grab()
                pixmap.save(tmp.name, 'PNG')
                with open(tmp.name, 'rb') as f:
                    data = f.read()
                os.unlink(tmp.name)
                return data
            except Exception:
                return b''


def _format_freq(hz: float) -> str:
    if abs(hz) >= 1_000_000:
        return f"{hz/1_000_000:.3f} MHz"
    elif abs(hz) >= 1_000:
        return f"{hz/1_000:.3f} kHz"
    return f"{hz:.1f} Hz"
