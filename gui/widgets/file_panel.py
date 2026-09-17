"""
File Panel — signal metadata, detected parameters, pipeline status.
Premium UI: SNR quality dot, visual weight hierarchy, translatable.
Includes a small, static drop zone at the bottom of the panel.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QGroupBox, QGridLayout, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QFontMetrics, QColor, QPainter
from core.signal_info import SignalInfo
from gui.language_manager import LM
from gui.drop_overlay import DropOverlay


class _ElidingLabel(QLabel):
    """Label that elides text with '…' when too narrow, with tooltip showing full text."""
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._full_text = text
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def setText(self, text: str):
        self._full_text = text
        super().setToolTip(text if text not in ("—", "") else "")
        self._update_elided()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_elided()

    def _update_elided(self):
        fm = QFontMetrics(self.font())
        elided = fm.elidedText(self._full_text, Qt.TextElideMode.ElideRight,
                               max(self.width() - 2, 10))
        super().setText(elided)


class _PipelineStatusLabel(_ElidingLabel):
    """Text status with a compact painted indicator, not a Unicode icon."""
    _COLORS = {'pending': '#8E8E93', 'running': '#1677E8', 'done': '#2E9D62', 'error': '#C63D3D'}

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._status = 'pending'
        self.setContentsMargins(13, 0, 0, 0)

    def set_status(self, status: str):
        self._status = status
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        color = QColor(self._COLORS.get(self._status, self._COLORS['pending']))
        y = self.height() // 2
        if self._status == 'pending':
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(color)
        else:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
        painter.drawEllipse(3, y - 3, 6, 6)
        painter.end()


class FilePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("file_panel_root")
        # Panel itself: min 160px, no fixed max — splitter controls width
        self.setMinimumWidth(160)
        self.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Expanding)
        self._status_by_step = {}
        self._build_ui()
        LM.language_changed.connect(self.retranslate)

    # ─────────────────────────────────────────────────────────────────────
    # Build
    # ─────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── Modulation hero card ──────────────────────────────────────────
        self._hero_card = _make_card()
        self._hero_card.setObjectName("antarang_hero")
        hero_lay = QVBoxLayout(self._hero_card)
        hero_lay.setContentsMargins(10, 8, 10, 8)
        hero_lay.setSpacing(3)

        # Top: label
        self._lbl_mod_title = QLabel("MODULATION")
        self._lbl_mod_title.setObjectName("label_key")
        self._lbl_mod_title.setStyleSheet(
            "color: #8E8E93; font-size: 10px; font-weight: 700; "
            "letter-spacing: 1px; background: transparent;")
        hero_lay.addWidget(self._lbl_mod_title)

        # Big modulation value + SNR dot on same row
        mod_row = QHBoxLayout()
        self._lbl_mod_value = QLabel("—")
        self._lbl_mod_value.setObjectName("label_param_big")
        # Font size via CSS (label_param_big = 14px) — no hardcoded QFont
        self._lbl_mod_value.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        mod_row.addWidget(self._lbl_mod_value, stretch=1)
        mod_row.addStretch()
        self._lbl_snr_dot = QLabel("●")
        self._lbl_snr_dot.setObjectName("snr_dot_none")
        self._lbl_snr_dot.setToolTip("Signal quality indicator — loads after analysis")
        mod_row.addWidget(self._lbl_snr_dot)
        hero_lay.addLayout(mod_row)

        # Confidence + SNR inline
        conf_row = QHBoxLayout()
        self._lbl_conf = QLabel("Confidence: —")
        self._lbl_conf.setObjectName("label_key")
        self._lbl_conf.setToolTip(
            "ML classifier confidence — how certain the model is about\n"
            "the detected modulation type.\n"
            "≥80% = high  |  50–80% = moderate  |  <50% = uncertain")
        conf_row.addWidget(self._lbl_conf)
        conf_row.addStretch()
        self._lbl_snr_val = QLabel("SNR: —")
        self._lbl_snr_val.setObjectName("label_key")
        conf_row.addWidget(self._lbl_snr_val)
        hero_lay.addLayout(conf_row)

        root.addWidget(self._hero_card)

        # ── File Info group ───────────────────────────────────────────────
        self._file_group = QGroupBox()
        self._file_group.setObjectName("side_panel")
        fg = QGridLayout(self._file_group)
        fg.setContentsMargins(8, 4, 8, 6)
        fg.setSpacing(4)
        fg.setColumnStretch(1, 1)

        self._file_rows = [
            ('name',     'file_panel.name'),
            ('type',     'file_panel.type'),
            ('size',     'file_panel.size'),
            ('sr',       'file_panel.sample_rate'),
            ('duration', 'file_panel.duration'),
            ('dtype',    'file_panel.data_type'),
        ]
        self._file_key_labels = {}
        self._file_labels     = {}
        for i, (key, tk) in enumerate(self._file_rows):
            lbl = QLabel(LM.t(tk) + ":")
            lbl.setObjectName("label_key")
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            lbl.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
            val = _ElidingLabel("—")
            val.setObjectName("label_value")
            fg.addWidget(lbl, i, 0)
            fg.addWidget(val, i, 1)
            self._file_key_labels[key] = lbl
            self._file_labels[key]     = val
        root.addWidget(self._file_group)

        # ── Signal Parameters group ───────────────────────────────────────
        self._param_group = QGroupBox()
        self._param_group.setObjectName("side_panel")
        pg = QGridLayout(self._param_group)
        pg.setContentsMargins(8, 4, 8, 6)
        pg.setSpacing(4)
        pg.setColumnStretch(1, 1)

        self._param_rows = [
            ('band',        'file_panel.band'),
            ('bandwidth',   'file_panel.bandwidth'),
            ('center_freq', 'file_panel.center_freq'),
            ('symbol_rate', 'file_panel.symbol_rate'),
            ('bit_rate',    'file_panel.bit_rate'),
        ]
        self._param_key_labels = {}
        self._param_labels     = {}
        for i, (key, tk) in enumerate(self._param_rows):
            lbl = QLabel(LM.t(tk) + ":")
            lbl.setObjectName("label_key")
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            lbl.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
            val = _ElidingLabel("—")
            val.setObjectName("label_value")
            pg.addWidget(lbl, i, 0)
            pg.addWidget(val, i, 1)
            self._param_key_labels[key] = lbl
            self._param_labels[key]     = val
        root.addWidget(self._param_group)

        # ── Pipeline Status group ─────────────────────────────────────────
        self._status_group = QGroupBox()
        self._status_group.setObjectName("side_panel")
        self._status_group.setToolTip(
            "Pipeline steps run in order:\n"
            "1. Demodulate → 2. De-interleave → 3. FEC Decode → 4. Correlate\n"
            "Use the control panel below to run each step.")
        sg = QGridLayout(self._status_group)
        sg.setContentsMargins(8, 4, 8, 6)
        sg.setSpacing(5)
        sg.setColumnStretch(1, 1)

        self._status_rows = [
            ('demod',    'file_panel.demodulation'),
            ('deinter',  'file_panel.de_interleave'),
            ('fec',      'file_panel.fec_decode'),
            ('correlate','file_panel.correlation'),
        ]
        self._status_key_labels = {}
        self._status_labels     = {}
        for i, (key, tk) in enumerate(self._status_rows):
            lbl = QLabel(LM.t(tk) + ":")
            lbl.setObjectName("label_key")
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            lbl.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
            val = _PipelineStatusLabel(LM.t("file_panel.pending"))
            val.setObjectName("label_value")
            sg.addWidget(lbl, i, 0)
            sg.addWidget(val, i, 1)
            self._status_key_labels[key] = lbl
            self._status_labels[key]     = val
        root.addWidget(self._status_group)

        # ── Drop zone (small, at bottom of panel) ────────────────────────
        self.drop_zone = DropOverlay(self)
        root.addWidget(self.drop_zone)

        root.addStretch()

        self._apply_group_titles()

    def _apply_group_titles(self):
        self._file_group.setTitle(LM.t("file_panel.file_info"))
        self._param_group.setTitle(LM.t("file_panel.detected_params"))
        self._status_group.setTitle(LM.t("file_panel.pipeline_status"))

    # ─────────────────────────────────────────────────────────────────────
    # Retranslation
    # ─────────────────────────────────────────────────────────────────────
    def retranslate(self):
        self._apply_group_titles()
        for key, tk in self._file_rows:
            self._file_key_labels[key].setText(LM.t(tk) + ":")
        for key, tk in self._param_rows:
            self._param_key_labels[key].setText(LM.t(tk) + ":")
        for key, tk in self._status_rows:
            self._status_key_labels[key].setText(LM.t(tk) + ":")
        for key, status in self._status_by_step.items():
            self._set_status_text(key, status)

    # ─────────────────────────────────────────────────────────────────────
    # Data updates
    # ─────────────────────────────────────────────────────────────────────
    def update_file_info(self, info: SignalInfo):
        self._file_labels['name'].setText(info.file_name or "—")
        self._file_labels['type'].setText(info.file_type or "—")
        self._file_labels['size'].setText(_format_size(info.file_size_bytes))
        self._file_labels['sr'].setText(_format_freq(info.sample_rate))
        self._file_labels['duration'].setText(f"{info.duration_sec:.3f} s")
        self._file_labels['dtype'].setText(info.dtype or "—")

    def update_parameters(self, info: SignalInfo):
        mod  = info.modulation or "—"
        conf = f"{info.mod_confidence * 100:.0f}%" if info.modulation else "—"
        band = info.band or "—"
        bw   = _format_freq(info.bandwidth_hz) if info.bandwidth_hz else "—"
        snr  = f"{info.snr_db:.1f} dB" if info.snr_db is not None else "—"
        cf   = _format_freq(info.center_freq_hz) if info.center_freq_hz else "—"
        sr   = _format_freq(info.symbol_rate_hz) if info.symbol_rate_hz else "—"
        br   = f"{info.bit_rate_bps:.0f} bps" if info.bit_rate_bps else "—"

        # Hero card
        self._lbl_mod_value.setText(mod)
        self._lbl_conf.setText(f"Confidence: {conf}")
        self._lbl_snr_val.setText(f"SNR: {snr}")

        self._lbl_mod_value.setStyleSheet(
            "font-weight: 600; background: transparent;")

        # SNR quality dot
        snr_val = info.snr_db if info.snr_db is not None else 0
        if info.snr_db is None:
            self._lbl_snr_dot.setObjectName("snr_dot_none")
            self._lbl_snr_dot.setToolTip("Signal quality: awaiting analysis")
        elif snr_val >= 20:
            self._lbl_snr_dot.setObjectName("snr_dot_good")
            self._lbl_snr_dot.setToolTip(f"SNR {snr_val:.1f} dB — Excellent")
        elif snr_val >= 10:
            self._lbl_snr_dot.setObjectName("snr_dot_ok")
            self._lbl_snr_dot.setToolTip(f"SNR {snr_val:.1f} dB — Good")
        else:
            self._lbl_snr_dot.setObjectName("snr_dot_bad")
            self._lbl_snr_dot.setToolTip(f"SNR {snr_val:.1f} dB — Poor")
        # Force stylesheet refresh for objectName change
        self._lbl_snr_dot.style().unpolish(self._lbl_snr_dot)
        self._lbl_snr_dot.style().polish(self._lbl_snr_dot)

        # Parameter rows
        self._param_labels['band'].setText(band)
        self._param_labels['bandwidth'].setText(bw)
        self._param_labels['center_freq'].setText(cf)
        self._param_labels['symbol_rate'].setText(sr)
        self._param_labels['bit_rate'].setText(br)

        self._param_labels['band'].setStyleSheet(
            "font-weight: 600; background: transparent;")

    def set_pipeline_status(self, step: str, status: str):
        if step not in self._status_labels:
            return
        self._set_status_text(step, status)

    def _set_status_text(self, step: str, status: str):
        label = self._status_labels[step]
        self._status_by_step[step] = status
        label.set_status(status)
        icons = {
            'pending': (LM.t("file_panel.pending"), 'background: transparent;'),
            'running': (LM.t("file_panel.running"), 'font-weight: 600; background: transparent;'),
            'done':    (LM.t("file_panel.done"),    'font-weight: 600; background: transparent;'),
            'error':   (LM.t("file_panel.error"),   'font-weight: 600; background: transparent;'),
        }
        text, style = icons.get(status, ('—', 'background: transparent;'))
        label.setText(text)
        label.setStyleSheet(style)

    def reset(self):
        self._lbl_mod_value.setText("—")
        self._lbl_mod_value.setStyleSheet(
            "font-weight: 600; background: transparent;")
        self._lbl_conf.setText("Confidence: —")
        self._lbl_snr_val.setText("SNR: —")
        self._lbl_snr_dot.setObjectName("snr_dot_none")
        self._lbl_snr_dot.setToolTip("Signal quality indicator — loads after analysis")
        for lbl in self._file_labels.values():
            lbl.setText("—")
            lbl.setStyleSheet("")
        for lbl in self._param_labels.values():
            lbl.setText("—")
            lbl.setStyleSheet("")
        for key in self._status_labels:
            self._set_status_text(key, 'pending')

    def apply_theme(self, mode: str):
        self.drop_zone.apply_theme(mode)


# ─────────────────────────────────────────────────────────────────────────────
# Helper widgets / functions
# ─────────────────────────────────────────────────────────────────────────────
def _make_card() -> QFrame:
    """Subtle summary surface; its global object style supplies the chrome."""
    card = QFrame()
    return card


def _format_size(n: int) -> str:
    if n > 1_000_000:
        return f"{n/1_000_000:.2f} MB"
    elif n > 1_000:
        return f"{n/1_000:.1f} KB"
    return f"{n} B"


def _format_freq(hz: float) -> str:
    if hz == 0:
        return "0 Hz"
    if abs(hz) >= 1_000_000:
        return f"{hz/1_000_000:.3f} MHz"
    elif abs(hz) >= 1_000:
        return f"{hz/1_000:.3f} kHz"
    return f"{hz:.1f} Hz"
