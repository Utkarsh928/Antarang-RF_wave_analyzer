"""
Control Panel — pipeline controls: demodulate, de-interleave, FEC, correlate.
Fully translatable via LanguageManager.
Fix #13: Wrapped in QScrollArea for small screens.
"""
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QPushButton,
                               QComboBox, QLabel, QGroupBox, QGridLayout,
                               QSpinBox, QFrame, QCheckBox, QScrollArea,
                               QSizePolicy, QAbstractSpinBox)
from PyQt6.QtCore import pyqtSignal, Qt
from gui.language_manager import LM

class ControlPanel(QWidget):
    # Signals emitted when user clicks pipeline buttons
    sig_demodulate       = pyqtSignal(str)
    sig_deinterleave     = pyqtSignal(str, int, int)
    sig_fec_decode       = pyqtSignal(str)
    sig_correlate        = pyqtSignal(str)
    sig_protocol_analysis = pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pipeline_container")
        self._build_ui()
        LM.language_changed.connect(self.retranslate)

    # ─────────────────────────────────────────────────────────────────────
    # Build
    # ─────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Outer layout holds the scroll area
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # Scroll area wraps the inner widget for small screens
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setObjectName("pipeline_scroll")
        # No setMaximumHeight — let the layout determine height naturally
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Minimum)

        inner = QWidget()
        inner.setObjectName("pipeline_row")
        inner.setSizePolicy(QSizePolicy.Policy.MinimumExpanding,
                            QSizePolicy.Policy.Minimum)
        layout = QHBoxLayout(inner)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        self._inner_widget = inner
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        # ── 1. Demodulate ────────────────────────────────────────────────
        self._demod_group = QGroupBox()
        self._demod_group.setObjectName("step_group")
        self._demod_group.setMinimumWidth(140)
        self._demod_group.setSizePolicy(QSizePolicy.Policy.Preferred,
                                        QSizePolicy.Policy.Minimum)
        dg = QGridLayout(self._demod_group)
        dg.setSpacing(3)
        dg.setContentsMargins(6, 4, 6, 4)

        self._lbl_mod = QLabel()
        dg.addWidget(self._lbl_mod, 0, 0)
        self._combo_mod = QComboBox()
        self._combo_mod.addItems([
            'Auto-detect',
            'BPSK', 'QPSK', '8PSK',
            'QAM16', 'QAM64', 'QAM256',
            'FSK2', 'FSK4', 'GFSK', 'CPFSK',
            'OFDM',
            'AM-DSB', 'AM-SSB', 'WBFM',
            'APSK16', 'APSK32',
            'PAM4', 'OOK'
        ])
        dg.addWidget(self._combo_mod, 0, 1)

        self._btn_demod = QPushButton()
        self._btn_demod.setObjectName("btn_step")
        self._btn_demod.clicked.connect(self._on_demodulate)
        self._btn_demod.setEnabled(False)
        dg.addWidget(self._btn_demod, 1, 0, 1, 2)

        layout.addWidget(self._demod_group)
        layout.addWidget(_make_arrow())

        # ── 2. De-interleave ─────────────────────────────────────────────
        self._inter_group = QGroupBox()
        self._inter_group.setObjectName("step_group")
        self._inter_group.setMinimumWidth(140)
        self._inter_group.setSizePolicy(QSizePolicy.Policy.Preferred,
                                        QSizePolicy.Policy.Minimum)
        ig = QGridLayout(self._inter_group)
        ig.setSpacing(3)
        ig.setContentsMargins(6, 4, 6, 4)

        self._lbl_method = QLabel()
        ig.addWidget(self._lbl_method, 0, 0)
        self._combo_inter = QComboBox()
        self._combo_inter.addItems([
            'Auto-detect', 'Block', 'Convolutional',
            'Diagonal', 'Pseudo-random'
        ])
        ig.addWidget(self._combo_inter, 0, 1)

        self._lbl_rows = QLabel()
        ig.addWidget(self._lbl_rows, 1, 0)
        self._spin_rows = QSpinBox()
        self._spin_rows.setRange(2, 256)
        self._spin_rows.setValue(8)
        self._spin_rows.setButtonSymbols(
            QAbstractSpinBox.ButtonSymbols.PlusMinus)
        self._spin_rows.setToolTip("Block rows or interleaver depth")
        ig.addWidget(self._spin_rows, 1, 1)

        # Fix #7: Add columns spinbox
        self._lbl_cols = QLabel()
        ig.addWidget(self._lbl_cols, 2, 0)
        self._spin_cols = QSpinBox()
        self._spin_cols.setRange(0, 256)
        self._spin_cols.setValue(0)
        self._spin_cols.setButtonSymbols(
            QAbstractSpinBox.ButtonSymbols.PlusMinus)
        self._spin_cols.setToolTip("Block columns (0 = auto / same as rows)")
        ig.addWidget(self._spin_cols, 2, 1)

        self._btn_inter = QPushButton()
        self._btn_inter.setObjectName("btn_step")
        self._btn_inter.clicked.connect(self._on_deinterleave)
        self._btn_inter.setEnabled(False)
        ig.addWidget(self._btn_inter, 3, 0, 1, 2)

        layout.addWidget(self._inter_group)
        layout.addWidget(_make_arrow())

        # ── 3. FEC Decode ────────────────────────────────────────────────
        self._fec_group = QGroupBox()
        self._fec_group.setObjectName("step_group")
        self._fec_group.setMinimumWidth(140)
        self._fec_group.setSizePolicy(QSizePolicy.Policy.Preferred,
                                      QSizePolicy.Policy.Minimum)
        fg = QGridLayout(self._fec_group)
        fg.setSpacing(3)
        fg.setContentsMargins(6, 4, 6, 4)

        self._lbl_fec = QLabel()
        fg.addWidget(self._lbl_fec, 0, 0)
        self._combo_fec = QComboBox()
        self._combo_fec.addItems([
            'Auto-detect', 'Viterbi', 'Reed-Solomon',
            'LDPC', 'Concatenated'
        ])
        fg.addWidget(self._combo_fec, 0, 1)

        self._btn_fec = QPushButton()
        self._btn_fec.setObjectName("btn_step")
        self._btn_fec.clicked.connect(self._on_fec)
        self._btn_fec.setEnabled(False)
        fg.addWidget(self._btn_fec, 1, 0, 1, 2)

        layout.addWidget(self._fec_group)
        layout.addWidget(_make_arrow())

        # ── 4. Bit Correlation ───────────────────────────────────────────
        self._corr_group = QGroupBox()
        self._corr_group.setObjectName("step_group")
        self._corr_group.setMinimumWidth(140)
        self._corr_group.setSizePolicy(QSizePolicy.Policy.Preferred,
                                       QSizePolicy.Policy.Minimum)
        cg = QGridLayout(self._corr_group)
        cg.setSpacing(3)
        cg.setContentsMargins(6, 4, 6, 4)

        self._lbl_sync = QLabel()
        cg.addWidget(self._lbl_sync, 0, 0)
        self._combo_sync = QComboBox()
        self._refresh_sync_combo()
        cg.addWidget(self._combo_sync, 0, 1)

        self._btn_corr = QPushButton()
        self._btn_corr.setObjectName("btn_step")
        self._btn_corr.clicked.connect(self._on_correlate)
        self._btn_corr.setEnabled(False)
        cg.addWidget(self._btn_corr, 1, 0, 1, 2)

        self._btn_sync_db = QPushButton()
        self._btn_sync_db.setObjectName("btn_sync_db")
        self._btn_sync_db.setToolTip(
            "Add custom protocol sync words for unknown signal formats")
        self._btn_sync_db.clicked.connect(self._open_sync_editor)
        cg.addWidget(self._btn_sync_db, 2, 0, 1, 2)

        layout.addWidget(self._corr_group)
        layout.addWidget(_make_arrow())

        # ── 5. Protocol Analysis ─────────────────────────────────────────
        self._proto_group = QGroupBox()
        self._proto_group.setObjectName("step_group")
        self._proto_group.setMinimumWidth(140)
        self._proto_group.setSizePolicy(QSizePolicy.Policy.Preferred,
                                        QSizePolicy.Policy.Minimum)
        pg = QGridLayout(self._proto_group)
        pg.setSpacing(3)
        pg.setContentsMargins(6, 4, 6, 4)

        self._lbl_proto_hint = QLabel()
        pg.addWidget(self._lbl_proto_hint, 0, 0)
        self._combo_protocol = QComboBox()
        self._combo_protocol.addItems([
            'Auto-detect', 'AX.25', 'CCSDS', 'Custom'
        ])
        pg.addWidget(self._combo_protocol, 0, 1)

        self._chk_security = QCheckBox()
        self._chk_security.setChecked(True)
        self._chk_security.setToolTip(
            "Detect encryption and analyze traffic patterns")
        pg.addWidget(self._chk_security, 1, 0, 1, 2)

        self._btn_protocol = QPushButton()
        self._btn_protocol.setObjectName("btn_step")
        self._btn_protocol.clicked.connect(self._on_protocol_analysis)
        self._btn_protocol.setEnabled(False)
        pg.addWidget(self._btn_protocol, 2, 0, 1, 2)

        layout.addWidget(self._proto_group)
        # No addStretch — scroll area handles overflow naturally

        # Apply translated text on first build
        self.retranslate()

    # ─────────────────────────────────────────────────────────────────────
    # Retranslation
    # ─────────────────────────────────────────────────────────────────────
    def retranslate(self):
        """Update all user-visible text to the current language."""
        # Group titles
        self._demod_group.setTitle(LM.t("control_panel.demodulate_group"))
        self._inter_group.setTitle(LM.t("control_panel.deinterleave_group"))
        self._fec_group.setTitle(LM.t("control_panel.fec_group"))
        self._corr_group.setTitle(LM.t("control_panel.correlate_group"))
        self._proto_group.setTitle(LM.t("control_panel.protocol_group"))

        # Labels
        self._lbl_mod.setText(LM.t("control_panel.modulation_label"))
        self._lbl_method.setText(LM.t("control_panel.method_label"))
        self._lbl_rows.setText(LM.t("control_panel.rows_label"))
        self._lbl_cols.setText(LM.t("control_panel.cols_label"))
        self._lbl_fec.setText(LM.t("control_panel.fec_label"))
        self._lbl_sync.setText(LM.t("control_panel.sync_label"))
        self._lbl_proto_hint.setText(LM.t("control_panel.protocol_hint_label"))
        # Buttons
        self._btn_demod.setText(LM.t("control_panel.btn_demodulate"))
        self._btn_inter.setText(LM.t("control_panel.btn_deinterleave"))
        self._btn_fec.setText(LM.t("control_panel.btn_fec"))
        self._btn_corr.setText(LM.t("control_panel.btn_correlate"))
        self._btn_sync_db.setText(LM.t("control_panel.btn_sync_db"))
        self._btn_protocol.setText(LM.t("control_panel.btn_protocol"))

        # Checkbox — correct key for "Run Security Analysis"
        self._chk_security.setText(LM.t("control_panel.run_security_label"))

    # ─────────────────────────────────────────────────────────────────────
    # Slots
    # ─────────────────────────────────────────────────────────────────────
    def _on_demodulate(self):
        mod = self._combo_mod.currentText()
        if mod == 'Auto-detect':
            mod = 'auto'
        self.sig_demodulate.emit(mod)

    def _on_deinterleave(self):
        method = self._combo_inter.currentText()
        if method == 'Auto-detect':
            method = 'auto'
        self.sig_deinterleave.emit(method, self._spin_rows.value(),
                                    self._spin_cols.value())  # Fix #7: pass real cols

    def _on_fec(self):
        fec = self._combo_fec.currentText()
        if fec == 'Auto-detect':
            fec = 'auto'
        self.sig_fec_decode.emit(fec)

    def _on_correlate(self):
        sync = self._combo_sync.currentText()
        if 'Auto' in sync:
            sync = 'auto'
        elif 'HDLC' in sync:
            sync = 'HDLC/AX.25'
        elif 'CCSDS' in sync:
            sync = 'CCSDS-32'
        elif 'Barker-7' in sync:
            sync = 'Barker-7'
        elif 'Barker-11' in sync:
            sync = 'Barker-11'
        elif 'Barker-13' in sync:
            sync = 'Barker-13'
        elif '802.11' in sync:
            sync = '802.11-SFD'
        self.sig_correlate.emit(sync)

    def _on_protocol_analysis(self):
        protocol = self._combo_protocol.currentText()
        auto_detect = (protocol == 'Auto-detect')
        protocol_hint = None if auto_detect else protocol
        self.sig_protocol_analysis.emit(auto_detect, protocol_hint or "")

    def _refresh_sync_combo(self):
        try:
            from core.bitstream.sync_db import get_builtin_names, load_custom
            current = self._combo_sync.currentText() \
                if self._combo_sync.count() > 0 else 'Auto-detect'
            self._combo_sync.clear()
            items = ['Auto-detect'] + get_builtin_names()
            custom = load_custom()
            if custom:
                items += list(custom.keys())
            self._combo_sync.addItems(items)
            idx = self._combo_sync.findText(current)
            if idx >= 0:
                self._combo_sync.setCurrentIndex(idx)
        except Exception:
            if self._combo_sync.count() == 0:
                self._combo_sync.addItems([
                    'Auto-detect', 'HDLC (0x7E)', 'CCSDS',
                    'Barker-7', 'Barker-11', 'Barker-13', '802.11'])

    def _open_sync_editor(self):
        try:
            from gui.sync_editor import SyncWordEditor
            dlg = SyncWordEditor(self.window())
            dlg.exec()
            self._refresh_sync_combo()
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self.window(), "Sync Editor",
                                f"Could not open editor: {e}")

    # ─────────────────────────────────────────────────────────────────────
    # State control
    # ─────────────────────────────────────────────────────────────────────
    def enable_demodulate(self, enabled: bool = True):
        self._btn_demod.setEnabled(enabled)

    def enable_deinterleave(self, enabled: bool = True):
        self._btn_inter.setEnabled(enabled)

    def enable_fec(self, enabled: bool = True):
        self._btn_fec.setEnabled(enabled)

    def enable_correlate(self, enabled: bool = True):
        self._btn_corr.setEnabled(enabled)

    def enable_protocol_analysis(self, enabled: bool = True):
        self._btn_protocol.setEnabled(enabled)

    def set_detected_modulation(self, modulation: str):
        idx = self._combo_mod.findText(
            modulation, Qt.MatchFlag.MatchFixedString)
        if idx >= 0:
            self._combo_mod.setCurrentIndex(idx)

    def reset(self):
        self.enable_demodulate(False)
        self.enable_deinterleave(False)
        self.enable_fec(False)
        self.enable_correlate(False)
        self.enable_protocol_analysis(False)
        self._combo_mod.setCurrentIndex(0)
        self._combo_inter.setCurrentIndex(0)
        self._combo_fec.setCurrentIndex(0)
        self._combo_sync.setCurrentIndex(0)
        self._combo_protocol.setCurrentIndex(0)

    def apply_theme(self):
        """Re-polish the scroll-hosted cards after the shared theme changes."""
        style = self._inner_widget.style()
        style.unpolish(self._inner_widget)
        style.polish(self._inner_widget)
        self._inner_widget.update()


def _make_arrow() -> QLabel:
    """Styled → arrow separator between pipeline steps."""
    lbl = QLabel("›")
    lbl.setObjectName("pipeline_arrow")
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setFixedWidth(18)
    return lbl


def _make_vsep() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.VLine)
    sep.setStyleSheet("color: #242426;")
    return sep
