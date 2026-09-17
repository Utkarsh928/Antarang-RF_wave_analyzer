"""
Constellation Widget — IQ scatter plot using pyqtgraph.
"""
import colorsys
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtGui import QColor, QPalette
from gui.theme import apply_window_chrome


class ConstellationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._has_data = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Info bar
        info_layout = QHBoxLayout()
        self._lbl_mod = QLabel("Modulation: —")
        self._lbl_mod.setObjectName("label_key")
        self._lbl_points = QLabel("Points: —")
        self._lbl_points.setObjectName("label_key")
        self._lbl_evm = QLabel("EVM: —")
        self._lbl_evm.setObjectName("label_key")
        self._lbl_evm.setToolTip(
            "Error Vector Magnitude — estimated by nearest-neighbor matching.\n"
            "Approximate only: no phase correction applied.\n"
            "High-order QAM with rotation may show inflated values."
        )
        for lbl in [self._lbl_mod, self._lbl_points, self._lbl_evm]:
            info_layout.addWidget(lbl)
        info_layout.addStretch()

        # Reset Zoom button — matches spectrum/waterfall
        self._btn_reset_zoom = QPushButton("Reset Zoom")
        self._btn_reset_zoom.setObjectName("btn_reset_zoom")
        self._btn_reset_zoom.setMaximumHeight(22)
        self._btn_reset_zoom.setToolTip("Reset zoom to fit all IQ points")
        self._btn_reset_zoom.clicked.connect(self._reset_zoom)
        info_layout.addWidget(self._btn_reset_zoom)
        layout.addLayout(info_layout)

        # Plot
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setLabel('left', 'Q (Quadrature)')
        self._plot_widget.setLabel('bottom', 'I (In-phase)')
        self._plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self._plot_widget.setAspectLocked(True)
        self.apply_theme('dark')
        self._install_export_dialog_theme_hook()
        self._plot_widget.setTitle(
            "<span style='color:#636366;font-size:13px'>"
            "Demodulate a signal to view the constellation</span>")

        # Draw axes lines through origin
        h_line = pg.InfiniteLine(
            angle=0, pos=0,
            pen=pg.mkPen('#38383A', width=1))
        v_line = pg.InfiniteLine(
            angle=90, pos=0,
            pen=pg.mkPen('#38383A', width=1))
        self._plot_widget.addItem(h_line)
        self._plot_widget.addItem(v_line)

        # Unit circle
        theta = np.linspace(0, 2 * np.pi, 200)
        self._unit_circle = self._plot_widget.plot(
            np.cos(theta), np.sin(theta),
            pen=pg.mkPen('#2C2C2E', width=1))

        # Scatter plot for IQ points
        self._scatter = pg.ScatterPlotItem(
            size=3,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(56, 189, 248, 125)
        )
        self._plot_widget.addItem(self._scatter)

        # Ideal constellation points overlay
        self._ideal_scatter = pg.ScatterPlotItem(
            size=10,
            pen=pg.mkPen('#F59E0B', width=2),
            brush=pg.mkBrush(None),
            symbol='x'
        )
        self._plot_widget.addItem(self._ideal_scatter)

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
        self._theme_mode = mode
        bg, text, reference = (('#050505', '#8E8E93', '#38383A') if mode == 'dark'
                               else ('#FFFFFF', '#4B5563', '#D9DDE3'))
        self._plot_widget.setBackground(bg)
        for axis in ('left', 'bottom'):
            self._plot_widget.getAxis(axis).setPen(pg.mkPen(text))
            self._plot_widget.getAxis(axis).setTextPen(pg.mkPen(text))
        if hasattr(self, '_unit_circle'):
            self._unit_circle.setPen(pg.mkPen(reference, width=1))
        scene = self._plot_widget.scene()
        if getattr(scene, "exportDialog", None) is not None:
            self._style_export_dialog(scene.exportDialog)

    def plot(self, samples: np.ndarray, modulation: str = "",
             max_points: int = 5000):
        """Plot IQ constellation from complex samples."""
        if not np.iscomplexobj(samples):
            return

        total = len(samples)
        # Subsample for performance
        if len(samples) > max_points:
            idx = np.random.choice(len(samples), max_points, replace=False)
            samples = samples[idx]
            subsampled = True
        else:
            subsampled = False

        # Normalize power
        pwr = np.sqrt(np.mean(np.abs(samples) ** 2))
        if pwr > 0:
            samples = samples / pwr

        i_vals = samples.real.astype(np.float32)
        q_vals = samples.imag.astype(np.float32)

        hues = ((np.angle(samples) + np.pi) / (2 * np.pi)).astype(float)
        brushes = [pg.mkBrush(*(int(v * 255) for v in colorsys.hsv_to_rgb(h, .68, .95)), 130)
                   for h in hues]
        self._scatter.setData(x=i_vals, y=q_vals, brush=brushes)

        # Draw ideal constellation points
        ideal = _get_ideal_constellation(modulation)
        if ideal is not None:
            pwr_i = np.sqrt(np.mean(np.abs(ideal) ** 2))
            if pwr_i > 0:
                ideal = ideal / pwr_i
            self._ideal_scatter.setData(
                x=ideal.real.astype(np.float32),
                y=ideal.imag.astype(np.float32)
            )
        else:
            self._ideal_scatter.setData(x=[], y=[])

        # EVM estimate — with (est.) qualifier to reflect approximation
        if ideal is not None and len(ideal) > 0:
            evm = _estimate_evm(samples, ideal)
            self._lbl_evm.setText(f"EVM (est.): {evm:.1f}%")   # Fix #14
        else:
            self._lbl_evm.setText("EVM: —")

        self._lbl_mod.setText(f"Modulation: {modulation or '—'}")
        pts_label = f"Points: {len(i_vals):,}"
        if subsampled:
            pts_label += f" of {total:,} (subsampled)"
        self._lbl_points.setText(pts_label)
        # Only auto-range on first plot — preserve user zoom on updates
        if not self._has_data:
            self._plot_widget.setTitle("")
            self._plot_widget.autoRange()
            self._has_data = True

    def _reset_zoom(self):
        """Reset zoom to fit all IQ points."""
        self._plot_widget.autoRange()

    def clear(self):
        self._scatter.setData(x=[], y=[])
        self._ideal_scatter.setData(x=[], y=[])
        self._lbl_mod.setText("Modulation: —")
        self._lbl_points.setText("Points: —")
        self._lbl_evm.setText("EVM: —")
        self._has_data = False
        self._plot_widget.setTitle(
            "<span style='color:#636366;font-size:13px'>"
            "Demodulate a signal to view the constellation</span>")


def _get_ideal_constellation(modulation: str) -> np.ndarray:
    """Return ideal constellation points for a given modulation."""
    if not modulation:
        return None
    mod = modulation.upper().replace('-', '').replace('_', '')

    if mod == 'BPSK':
        return np.array([-1+0j, 1+0j])
    elif mod == 'QPSK':
        return np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
    elif mod == '8PSK':
        angles = np.pi * np.arange(8) / 4
        return np.exp(1j * angles)
    elif mod == 'QAM16':
        levels = [-3, -1, 1, 3]
        return np.array([i + 1j*q for q in levels for i in levels],
                        dtype=np.complex64)
    elif mod == 'QAM64':
        levels = [-7, -5, -3, -1, 1, 3, 5, 7]
        return np.array([i + 1j*q for q in levels for i in levels],
                        dtype=np.complex64)
    elif mod == 'APSK16':
        r1, r2 = 1.0, 2.5
        inner  = [r1 * np.exp(1j*(2*np.pi*k/4+np.pi/4)) for k in range(4)]
        outer  = [r2 * np.exp(1j*(2*np.pi*k/12))        for k in range(12)]
        return np.array(inner + outer, dtype=np.complex64)
    elif mod == 'APSK32':
        r1, r2, r3 = 1.0, 2.6, 4.5
        inner  = [r1 * np.exp(1j*(2*np.pi*k/4+np.pi/4)) for k in range(4)]
        middle = [r2 * np.exp(1j*(2*np.pi*k/12))        for k in range(12)]
        outer  = [r3 * np.exp(1j*(2*np.pi*k/16))        for k in range(16)]
        return np.array(inner + middle + outer, dtype=np.complex64)
    elif mod == 'OOK':
        return np.array([0+0j, 1+0j])
    elif mod == 'PAM4':
        return np.array([-3+0j, -1+0j, 1+0j, 3+0j]) / 3.0
    return None


def _estimate_evm(received: np.ndarray,
                  ideal: np.ndarray) -> float:
    """Estimate EVM (%) by nearest-neighbor matching."""
    if len(received) == 0 or len(ideal) == 0:
        return 0.0
    total_error = 0.0
    total_power = 0.0
    for sym in received[:500]:  # limit for speed
        dists = np.abs(ideal - sym) ** 2
        nearest = ideal[np.argmin(dists)]
        total_error += float(np.abs(sym - nearest) ** 2)
        total_power += float(np.abs(nearest) ** 2)
    if total_power == 0:
        return 0.0
    return float(np.sqrt(total_error / total_power) * 100)
