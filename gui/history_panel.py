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
        return f"{self.file_name}\n{self.timestamp}  •  {self.modulation}  •  SNR {self.snr_db:.1f} dB"

    def detail_html(self) -> str:
        def _row(k, v, alt=False):
            bg = "#18181A" if alt else "#121214"
            return (f"<tr style='background:{bg};'>"
                    f"<td style='padding:4px 8px; color:#8E8E93; font-size:11px; font-weight:500; border-bottom:1px solid #242426; width:90px;'>{k}</td>"
                    f"<td style='padding:4px 8px; color:#F2F2F7; font-size:11px; font-family:Consolas, monospace; border-bottom:1px solid #242426;'>{v}</td>"
                    f"</tr>")

        def _hz(v):
            if v >= 1e6: return f"{v/1e6:.3f} MHz"
            if v >= 1e3: return f"{v/1e3:.3f} kHz"
            return f"{v:.1f} Hz"

        rows_data = [
            ("File", self.file_name),
            ("Type", self.file_type),
            ("Time", self.timestamp),
            ("Sample Rate", _hz(self.sample_rate) if self.sample_rate else "—"),
            ("Duration", f"{self.duration:.3f} s"),
            ("Modulation", f"{self.modulation} ({self.mod_confidence*100:.1f}%)"),
            ("SNR", f"{self.snr_db:.1f} dB"),
            ("Bandwidth", _hz(self.bandwidth) if self.bandwidth else "—"),
            ("Bit Rate", f"{self.bit_rate:.0f} bps" if self.bit_rate else "—"),
            ("FEC", self.fec_type),
        ]
        if self.fec_type != '—':
            rows_data.append(("FEC Errors", f"det={self.errors_detected}  corr={self.errors_corrected}"))
        if self.protocol:
            rows_data.append(("Protocol", self.protocol))
        if self.num_frames is not None:
            rows_data.append(("Frames", str(self.num_frames)))
        if self.encrypted is not None:
            enc_text = "DETECTED" if self.encrypted else "Not Detected"
            rows_data.append(("Encryption", enc_text))
        if self.shannon_entropy is not None:
            rows_data.append(("Entropy", f"{self.shannon_entropy:.4f} b/B"))

        rows_html = "".join(_row(k, v, i % 2 == 1) for i, (k, v) in enumerate(rows_data))

        return (
            f"<div style='font-family:Segoe UI, sans-serif; padding:4px;'>"
            f"<div style='color:#F2F2F7; font-weight:600; font-size:12px; margin-bottom:6px; word-break:break-all;'>"
            f"{self.file_name}</div>"
            f"<table style='width:100%; border-collapse:collapse; border:1px solid #2C2C2E; border-radius:6px; overflow:hidden;'>"
            f"{rows_html}"
            f"</table>"
            f"</div>"
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
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Header toolbar
        hdr = QHBoxLayout()
        hdr.setSpacing(6)

        lbl = QLabel("Session History")
        lbl.setStyleSheet("font-size: 13px; font-weight: 600; color: #F2F2F7;")
        hdr.addWidget(lbl)

        self._lbl_count = QLabel("0 signals")
        self._lbl_count.setStyleSheet(
            "background: #1C1C1E; color: #8E8E93; border: 1px solid #2C2C2E; "
            "border-radius: 9px; padding: 2px 7px; font-size: 11px; font-weight: 500;")
        hdr.addWidget(self._lbl_count)
        hdr.addStretch()

        self._btn_open = QPushButton("Open")
        self._btn_open.setToolTip("Reload selected signal into workspace")
        self._btn_open.setFixedHeight(26)
        self._btn_open.setStyleSheet(
            "QPushButton { background: #1677E8; color: #FFFFFF; border: none; "
            "border-radius: 6px; padding: 2px 10px; font-size: 11px; font-weight: 600; } "
            "QPushButton:hover { background: #1468D5; }")
        self._btn_open.clicked.connect(self._on_open_clicked)
        hdr.addWidget(self._btn_open)

        self._btn_clear = QPushButton("Clear")
        self._btn_clear.setToolTip("Clear session history log")
        self._btn_clear.setFixedHeight(26)
        self._btn_clear.setStyleSheet(
            "QPushButton { background: #242426; color: #8E8E93; border: 1px solid #2C2C2E; "
            "border-radius: 6px; padding: 2px 10px; font-size: 11px; } "
            "QPushButton:hover { background: #2C2C2E; color: #F2F2F7; }")
        self._btn_clear.clicked.connect(self._clear)
        hdr.addWidget(self._btn_clear)

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
        self._list.setWordWrap(True)
        self._list.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self._list.setStyleSheet(
            "QListWidget { background: #101011; border: 1px solid #2C2C2E; border-radius: 8px; padding: 2px; } "
            "QListWidget::item { padding: 6px 8px; border-bottom: 1px solid #1C1C1E; border-radius: 6px; margin: 1px 2px; line-height: 1.3; } "
            "QListWidget::item:selected { background: #242426; color: #F2F2F7; border: 1px solid #38383A; } "
            "QListWidget::item:hover:!selected { background: #18181A; }")
        self._list.currentRowChanged.connect(self._on_row_changed)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        splitter.addWidget(self._list)

        # Detail panel
        self._detail = QTextEdit()
        self._detail.setReadOnly(True)
        self._detail.setMinimumHeight(120)
        self._detail.setStyleSheet(
            "QTextEdit { background: #101011; border: 1px solid #2C2C2E; border-radius: 8px; padding: 4px; }")
        self._detail.setHtml(
            "<div style='font-family:Segoe UI, sans-serif; color:#636366; font-size:11px; font-style:italic; padding:8px;'>"
            "Select an entry above to view parameters.<br>Double-click or press 'Open' to reload signal."
            "</div>")
        splitter.addWidget(self._detail)
        splitter.setSizes([180, 160])

        root.addWidget(splitter, stretch=1)

        # Bottom Hint
        hint = QLabel("💡 Double-click or select & press Open to reload file")
        hint.setStyleSheet("color: #636366; font-size: 10px;")
        root.addWidget(hint)

    def add_entry(self, info: SignalInfo):
        """Called by MainWindow whenever analysis completes."""
        entry = _HistoryEntry(info)
        self._entries.insert(0, entry)       # newest first

        item = QListWidgetItem(entry.list_label())
        item.setToolTip(
            f"{entry.file_name}\nModulation: {entry.modulation} ({entry.mod_confidence*100:.1f}%)\n"
            f"SNR: {entry.snr_db:.1f} dB\nDuration: {entry.duration:.3f} s\nPath: {entry.file_path}")
        item.setData(Qt.ItemDataRole.UserRole, len(self._entries) - 1)

        self._list.insertItem(0, item)
        self._list.setCurrentRow(0)
        self._lbl_count.setText(f"{len(self._entries)} signal{'s' if len(self._entries)!=1 else ''}")

    def _on_open_clicked(self):
        item = self._list.currentItem()
        if item:
            self._on_double_click(item)

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
            "<div style='font-family:Segoe UI, sans-serif; color:#636366; font-size:11px; font-style:italic; padding:8px;'>"
            "History cleared."
            "</div>")
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
