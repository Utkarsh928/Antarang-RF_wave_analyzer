"""
Waterfall Widget — professional time-frequency heatmap.
Uses pyqtgraph ImageItem with proper axis scaling.
No ColorBarItem (unstable API) — uses a custom gradient legend instead.
"""
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QSizePolicy, QPushButton)
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QLinearGradient, QColor, QPainter, QBrush, QPen, QPalette
from gui.theme import apply_window_chrome


# ── Waterfall colormap (black→charcoal→white) ───────────────────────────────
_WATERFALL_COLORS = [
    (6, 8, 28, 255), (25, 30, 110, 255), (36, 88, 190, 255),
    (35, 170, 210, 255), (113, 205, 118, 255), (239, 216, 68, 255),
    (244, 119, 35, 255), (190, 35, 66, 255),
]

_WATERFALL_POSITIONS = [0.0, 0.12, 0.26, 0.42, 0.58, 0.74, 0.88, 1.0]


def _build_colormap() -> pg.ColorMap:
    return pg.ColorMap(
        pos=np.array(_WATERFALL_POSITIONS),
        color=np.array(_WATERFALL_COLORS, dtype=np.ubyte)
    )


# ── Gradient legend widget ─────────────────────────────────────────────────────
class _GradientLegend(QWidget):
    """Thin vertical color bar with min/max dB labels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._vmin = -80.0
        self._vmax = 0.0
        self._chrome = QColor('#8E8E93')
        self.setFixedWidth(56)
        self.setSizePolicy(QSizePolicy.Policy.Fixed,
                           QSizePolicy.Policy.Expanding)

    def set_levels(self, vmin: float, vmax: float):
        self._vmin = vmin
        self._vmax = vmax
        self.update()

    def set_chrome(self, color: str):
        self._chrome = QColor(color)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        bar_x = 8
        bar_w = 18
        bar_top = 20          # extra top margin so "dB" label fits
        bar_bot = h - 16

        # Draw gradient bar
        grad = QLinearGradient(bar_x, bar_top, bar_x, bar_bot)
        for pos, color in zip(_WATERFALL_POSITIONS[::-1], _WATERFALL_COLORS[::-1]):
            grad.setColorAt(pos, QColor(*color))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(self._chrome, 1))
        painter.drawRect(bar_x, bar_top, bar_w, bar_bot - bar_top)

        # Labels
        painter.setPen(QPen(self._chrome))
        fm = painter.fontMetrics()
        top_lbl = f"{self._vmax:.0f}"
        bot_lbl = f"{self._vmin:.0f}"
        mid_lbl = f"{(self._vmin+self._vmax)/2:.0f}"
        unit_lbl = "dB"

        lx = bar_x + bar_w + 3
        painter.drawText(lx, bar_top + fm.ascent(), top_lbl)
        mid_y = (bar_top + bar_bot) // 2
        painter.drawText(lx, mid_y + fm.ascent() // 2, mid_lbl)
        painter.drawText(lx, bar_bot, bot_lbl)

        # "dB" unit above the bar — safely within widget bounds
        painter.setPen(QPen(self._chrome))
        painter.drawText(bar_x, fm.ascent() + 2, unit_lbl)

        painter.end()


# ── Main waterfall widget ──────────────────────────────────────────────────────
class WaterfallWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._vmin = -80.0
        self._vmax = 0.0
        self._cmap = _build_colormap()
        self._lut = self._cmap.getLookupTable(nPts=256, alpha=False)
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # ── Info bar (row 1: stats + reset zoom) ─────────────────────
        info_layout = QHBoxLayout()
        info_layout.setSpacing(12)

        self._lbl_time = QLabel("Duration: —")
        self._lbl_freq = QLabel("Freq range: —")
        self._lbl_res  = QLabel("Resolution: —")
        # Hint uses a narrow eliding label so it doesn't crowd smaller windows
        self._lbl_hint = QLabel("Intensity: low to high")
        self._lbl_hint.setStyleSheet("color: #636366; font-size: 10px;")
        self._lbl_hint.setMaximumWidth(240)

        for lbl in [self._lbl_time, self._lbl_freq, self._lbl_res]:
            lbl.setObjectName("label_key")
            info_layout.addWidget(lbl)

        info_layout.addStretch()
        info_layout.addWidget(self._lbl_hint)

        self._btn_reset_zoom = QPushButton("Reset Zoom")
        self._btn_reset_zoom.setObjectName("btn_reset_zoom")
        self._btn_reset_zoom.setMaximumHeight(22)
        self._btn_reset_zoom.setToolTip("Reset zoom to fit full waterfall")
        self._btn_reset_zoom.clicked.connect(self._reset_zoom)
        info_layout.addWidget(self._btn_reset_zoom)
        outer.addLayout(info_layout)

        # ── Plot row (plot + gradient bar) ────────────────────────────
        plot_row = QHBoxLayout()
        plot_row.setSpacing(2)

        # pyqtgraph plot
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setLabel('left',   'Time',      units='s',
                                   color='#8E8E93')
        self._plot_widget.setLabel('bottom', 'Frequency', units='Hz',
                                   color='#8E8E93')
        self.apply_theme('dark')
        self._install_export_dialog_theme_hook()
        self._plot_widget.showGrid(x=True, y=True, alpha=0.15)

        # ImageItem
        self._img = pg.ImageItem()
        self._img.setLookupTable(self._lut)
        self._plot_widget.addItem(self._img)

        # Crosshair cursor
        self._vline = pg.InfiniteLine(angle=90, movable=False,
                                       pen=pg.mkPen('#ffffff', width=1,
                                                    style=Qt.PenStyle.DashLine))
        self._hline = pg.InfiniteLine(angle=0,  movable=False,
                                       pen=pg.mkPen('#ffffff', width=1,
                                                    style=Qt.PenStyle.DashLine))
        self._plot_widget.addItem(self._vline)
        self._plot_widget.addItem(self._hline)
        self._vline.hide()
        self._hline.hide()

        # Mouse hover
        self._plot_widget.scene().sigMouseMoved.connect(self._on_mouse_move)

        plot_row.addWidget(self._plot_widget, stretch=1)

        # Gradient legend
        self._legend = _GradientLegend()
        plot_row.addWidget(self._legend)

        outer.addLayout(plot_row, stretch=1)

        # ── Cursor info row (row 3: below plot, right-aligned) ────────
        cursor_row = QHBoxLayout()
        cursor_row.setContentsMargins(0, 0, 0, 0)
        cursor_row.addStretch()
        self._lbl_cursor = QLabel("")
        self._lbl_cursor.setStyleSheet(
            "color: #AEAEB2; background: #18181A; "
            "padding: 2px 8px; border-radius: 3px; font-size: 11px;")
        self._lbl_cursor.setAlignment(Qt.AlignmentFlag.AlignRight)
        cursor_row.addWidget(self._lbl_cursor)
        outer.addLayout(cursor_row)

        # State
        self._freq_min = 0.0
        self._freq_max = 1.0
        self._time_min = 0.0
        self._time_max = 1.0

    def _install_export_dialog_theme_hook(self):
        """Apply the application theme to pyqtgraph's built-in export dialog."""
        scene = self._plot_widget.scene()
        original_show_export_dialog = scene.showExportDialog

        def show_export_dialog():
            original_show_export_dialog()
            self._style_export_dialog(scene.exportDialog)

        # GraphicsScene connected the original bound method during
        # initialization, so replacing the attribute alone would not affect
        # the context-menu action.
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
        """Keep heatmap colours fixed while adapting plot readability."""
        self._theme_mode = mode
        bg, text, cursor_bg = (('#050505', '#8E8E93', '#18181A') if mode == 'dark'
                               else ('#FFFFFF', '#4B5563', '#F1F3F6'))
        self._plot_widget.setBackground(bg)
        for axis in ('left', 'bottom'):
            ax = self._plot_widget.getAxis(axis)
            ax.setPen(pg.mkPen(text))
            ax.setTextPen(pg.mkPen(text))
        if hasattr(self, '_legend'):
            self._legend.set_chrome(text)
        if hasattr(self, '_lbl_hint'):
            self._lbl_hint.setStyleSheet(f"color: {text}; font-size: 10px;")
        if hasattr(self, '_lbl_cursor'):
            self._lbl_cursor.setStyleSheet(
                f"color: {text}; background: {cursor_bg}; padding: 2px 8px; "
                "border-radius: 3px; font-size: 11px;")
        if hasattr(self, '_vline'):
            self._vline.setPen(pg.mkPen(text, width=1, style=Qt.PenStyle.DashLine))
            self._hline.setPen(pg.mkPen(text, width=1, style=Qt.PenStyle.DashLine))
        scene = self._plot_widget.scene()
        if getattr(scene, "exportDialog", None) is not None:
            self._style_export_dialog(scene.exportDialog)

    # ── Public API ────────────────────────────────────────────────────────────

    def plot(self, freqs: np.ndarray, times: np.ndarray,
             power_db: np.ndarray):
        """
        Update waterfall.
        power_db shape: (n_freqs, n_times) — rows=frequencies, cols=time
        """
        if power_db.size == 0:
            return

        # Clip/normalize levels with percentile stretch
        self._vmin = float(np.percentile(power_db,  2))
        self._vmax = float(np.percentile(power_db, 98))

        # Avoid zero-range
        if self._vmax - self._vmin < 1.0:
            self._vmax = self._vmin + 1.0

        # pyqtgraph ImageItem expects shape (width, height) = (n_freqs, n_times)
        # power_db is already (n_freqs, n_times) from compute_spectrogram
        img_data = np.ascontiguousarray(power_db.astype(np.float32))

        self._freq_min = float(freqs[0])
        self._freq_max = float(freqs[-1])
        self._time_min = float(times[0])
        self._time_max = float(times[-1])

        freq_span = self._freq_max - self._freq_min
        time_span = self._time_max - self._time_min

        # Set image with levels
        self._img.setImage(img_data, levels=(self._vmin, self._vmax),
                           autoLevels=False)

        # Position image correctly in data coordinates
        # setRect(x, y, w, h): x=freq_min, y=time_min
        rect = QRectF(
            float(self._freq_min),
            float(self._time_min),
            float(max(freq_span, 1e-6)),
            float(max(time_span, 1e-6))
        )
        self._img.setRect(rect)

        # Update legend
        self._legend.set_levels(self._vmin, self._vmax)

        # Auto-range to fit image
        self._plot_widget.setXRange(self._freq_min, self._freq_max,
                                     padding=0.01)
        self._plot_widget.setYRange(self._time_min, self._time_max,
                                     padding=0.01)

        # Update info labels
        dur = time_span
        freq_step = freq_span / max(len(freqs) - 1, 1)
        time_step = time_span / max(len(times) - 1, 1)

        self._lbl_time.setText(f"Duration: {dur:.3f} s")
        self._lbl_freq.setText(
            f"{_format_freq(self._freq_min)} → {_format_freq(self._freq_max)}")
        self._lbl_res.setText(
            f"Δf: {_format_freq(freq_step)}  Δt: {time_step*1000:.1f} ms")

    def clear(self):
        self._img.clear()
        self._lbl_time.setText("Duration: —")
        self._lbl_freq.setText("Freq range: —")
        self._lbl_res.setText("Resolution: —")
        self._lbl_cursor.setText("")
        self._vline.hide()
        self._hline.hide()

    def _reset_zoom(self):
        """Fix #17: Reset zoom to full data range."""
        if self._freq_min != self._freq_max:
            self._plot_widget.setXRange(self._freq_min, self._freq_max, padding=0.01)
            self._plot_widget.setYRange(self._time_min, self._time_max, padding=0.01)

    # ── Mouse cursor ──────────────────────────────────────────────────────────

    def _on_mouse_move(self, pos):
        """Show crosshair and data values under cursor."""
        try:
            vb = self._plot_widget.getViewBox()
            if self._plot_widget.sceneBoundingRect().contains(pos):
                mouse_point = vb.mapSceneToView(pos)
                freq = mouse_point.x()
                t    = mouse_point.y()

                if (self._freq_min <= freq <= self._freq_max and
                        self._time_min <= t <= self._time_max):
                    self._vline.setPos(freq)
                    self._hline.setPos(t)
                    self._vline.show()
                    self._hline.show()
                    self._lbl_cursor.setText(
                        f"f: {_format_freq(freq)}  t: {t:.3f}s")
                else:
                    self._vline.hide()
                    self._hline.hide()
                    self._lbl_cursor.setText("")
        except Exception:
            pass


def _format_freq(hz: float) -> str:
    if abs(hz) >= 1_000_000:
        return f"{hz/1_000_000:.3f} MHz"
    elif abs(hz) >= 1_000:
        return f"{hz/1_000:.2f} kHz"
    return f"{hz:.1f} Hz"
