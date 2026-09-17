"""
Signal History / Session Log Panel — Fix #19
A dockable panel that records every analyzed signal in the current session.
Clicking an entry shows a summary. Double-clicking re-opens the file.
"""
from datetime import datetime
from typing import List, Optional, Callable

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QListWidget, QListWidgetItem, QTextEdit,
    QSplitter, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from core.signal_info import SignalInfo


class _HistoryEntry:
    """Lightweight snapshot of one analyzed signal."""
    def __init__(self, info: SignalInfo):
        self.timestamp = datetime.now().strftime("%H:%M:%S")
        self.file_path = getattr(info, 'file_path', '')
        self.file_name = info.file_name or '—'
        self.file_type = info.file_type or '—'
        self.modulation = info.modulation or '—'
        self.mod_confidence = info.mod_confidence
        self.sample_rate = info.sample_rate
        self.duration = info.duration_sec
        self.snr_db = info.snr_db
        self.bandwidth = info.bandwidth_hz
        self.bit_rate = info.bit_rate_bps
        self.protocol = getattr(info, 'protocol_type', None)
        self.encrypted = getattr(info, 'crypto_detected', None)
        self.shannon_entropy = getattr(info, 'crypto_entropy', None)
        self.num_frames = getattr(info, 'num_frames', None)
        self.fec_type = info.fec_type or '—'
        self.errors_detected = info.fec_errors_detected
        self.errors_corrected = info.fec_errors_corrected

    def list_label(self) -> str:
        return (f"[{self.timestamp}]  {self.file_name}"
                f"  —  {self.modulation}  |  SNR {self.snr_db:.1f} dB")

    def detail_html(self) -> str:
        def _row(k, v): return f"<tr><td style='padding:3px 8px'>{k}</td><td style='padding:3px 8px'>{v}</td></tr>"
        def _hz(v):
            if v >= 1e6: return f"{v/1e6:.3f} MHz"
            if v >= 1e3: return f"{v/1e3:.3f} kHz"
            return f"{v:.1f} Hz"

        rows = [
            _row("File", self.file_name),
            _row("Type", self.file_type),
            _row("Time", self.timestamp),
            _row("Sample Rate", _hz(self.sample_rate) if self.sample_rate else "—"),
            _row("Duration", f"{self.duration:.3f} s"),
            _row("Modulation", f"{self.modulation}  ({self.mod_confidence*100:.1f}%)"),
            _row("SNR", f"{self.snr_db:.1f} dB"),
            _row("Bandwidth", _hz(self.bandwidth) if self.bandwidth else "—"),
            _row("Bit Rate", f"{self.bit_rate:.0f} bps" if self.bit_rate else "—"),
            _row("FEC", self.fec_type),
        ]
        if self.fec_type != '—':
            rows.append(_row("FEC Errors", f"detected={self.errors_detected}  corrected={self.errors_corrected}"))
        if self.protocol:
            rows.append(_row("Protocol", self.protocol))
        if self.num_frames is not None:
            rows.append(_row("Frames", str(self.num_frames)))
        if self.encrypted is not None:
            enc_text = "DETECTED" if self.encrypted else "Not Detected"
            rows.append(f"<tr><td style='padding:3px 8px'>Encryption</td>"
                        f"<td style='padding:3px 8px'><b>{enc_text}</b></td></tr>")
        if self.shannon_entropy is not None:
            rows.append(_row("Entropy", f"{self.shannon_entropy:.4f} bits/byte"))

        return (
            f"<h3 style='margin-bottom:4px'>{self.file_name}</h3>"
            f"<table style='border-collapse:collapse'>{''.join(rows)}</table>"
        )


class HistoryPanel(QWidget):
    """
    Dockable panel showing all signals analyzed in this session.
    Emits sig_reopen(path) when user wants to reload a file.
    """
    sig_reopen = pyqtSignal(str)   # emits file path on double-click

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries: List[_HistoryEntry] = []
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # Header
        hdr = QHBoxLayout()
        lbl = QLabel("Session History")
        lbl.setObjectName("label_section")
        hdr.addWidget(lbl)
        hdr.addStretch()
        self._lbl_count = QLabel("0 signals")
        self._lbl_count.setStyleSheet("color: #636366; font-size: 11px;")
        hdr.addWidget(self._lbl_count)
        btn_clear = QPushButton("Clear")
        btn_clear.setMaximumHeight(22)
        btn_clear.setMaximumWidth(50)
        btn_clear.clicked.connect(self._clear)
        hdr.addWidget(btn_clear)
        root.addLayout(hdr)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("separator")
        root.addWidget(sep)

        # Splitter: list on top, detail below
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(4)

        # List
        self._list = QListWidget()
        self._list.setFont(QFont("Segoe UI", 11))
        self._list.currentRowChanged.connect(self._on_row_changed)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        splitter.addWidget(self._list)

        # Detail panel
        self._detail = QTextEdit()
        self._detail.setReadOnly(True)
        self._detail.setMinimumHeight(120)
        self._detail.setHtml(
            "<p style='font-style:italic'>"
            "Select an entry to see details. "
            "Double-click to reload the file.</p>")
        splitter.addWidget(self._detail)
        splitter.setSizes([180, 140])

        root.addWidget(splitter, stretch=1)

        # Hint
        hint = QLabel("Double-click an entry to reload that file")
        hint.setStyleSheet("color: #636366; font-size: 10px;")
        root.addWidget(hint)

    def add_entry(self, info: SignalInfo):
        """Called by MainWindow whenever analysis completes."""
        entry = _HistoryEntry(info)
        self._entries.insert(0, entry)       # newest first

        item = QListWidgetItem(entry.list_label())
        item.setData(Qt.ItemDataRole.UserRole, len(self._entries) - 1)

        self._list.insertItem(0, item)
        self._list.setCurrentRow(0)
        self._lbl_count.setText(f"{len(self._entries)} signal{'s' if len(self._entries)!=1 else ''}")

    def _on_row_changed(self, row: int):
        if row < 0 or row >= len(self._entries):
            return
        self._detail.setHtml(self._entries[row].detail_html())

    def _on_double_click(self, item: QListWidgetItem):
        row = self._list.row(item)
        if row < 0 or row >= len(self._entries):
            return
        entry = self._entries[row]
        if entry.file_path and entry.file_path != 'live_sdr':
            self.sig_reopen.emit(entry.file_path)

    def _clear(self):
        self._entries.clear()
        self._list.clear()
        self._detail.setHtml(
            "<p style='font-style:italic'>History cleared.</p>")
        self._lbl_count.setText("0 signals")

    def session_summary(self) -> str:
        """Plain-text summary of all signals in this session."""
        if not self._entries:
            return "No signals analyzed this session."
        lines = [f"Session summary — {len(self._entries)} signal(s)\n" + "="*50]
        for e in self._entries:
            lines.append(
                f"[{e.timestamp}] {e.file_name}  |  {e.modulation}  |  "
                f"SNR {e.snr_db:.1f} dB  |  {e.duration:.3f} s")
        return "\n".join(lines)
