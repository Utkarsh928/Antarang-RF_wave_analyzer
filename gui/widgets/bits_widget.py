"""
Bits Widget — displays raw bits, decoded bits, header and payload with hex/ASCII view.
"""
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QTextEdit, QTabWidget, QPushButton,
                               QGroupBox, QGridLayout, QFrame)
from PyQt6.QtGui import QFont, QColor, QTextCharFormat, QTextCursor, QFontDatabase
from PyQt6.QtCore import Qt
from core.bitstream.correlator import bits_to_hex, bits_to_ascii
from gui.language_manager import LM


class BitsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._raw_bits = None
        self._corrected_bits = None
        self._header_offset = 0
        self._payload_offset = 0
        self._theme_mode = 'dark'
        self._build_ui()
        self.apply_theme('dark')

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Stats bar
        stats_layout = QHBoxLayout()
        self._lbl_total = QLabel("Total bits: —")
        self._lbl_total.setObjectName("label_key")
        self._lbl_errors = QLabel("Errors: —")
        self._lbl_errors.setObjectName("label_key")
        self._lbl_header = QLabel("Header @ bit: —")
        self._lbl_header.setObjectName("label_key")
        self._lbl_payload = QLabel("Payload @ bit: —")
        self._lbl_payload.setObjectName("label_key")
        for lbl in [self._lbl_total, self._lbl_errors,
                    self._lbl_header, self._lbl_payload]:
            stats_layout.addWidget(lbl)
        stats_layout.addStretch()
        layout.addLayout(stats_layout)

        # Use system monospace font for reliable column alignment
        _mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        _mono.setPointSize(11)

        # Tabs: Binary | Hex | ASCII
        self._tabs = QTabWidget()
        self._tabs.setObjectName("bits_tabs")

        # Binary tab
        self._bin_view = QTextEdit()
        self._bin_view.setReadOnly(True)
        self._bin_view.setFont(_mono)
        self._bin_view.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self._tabs.addTab(self._bin_view, "Binary")

        # Hex tab
        self._hex_view = QTextEdit()
        self._hex_view.setReadOnly(True)
        self._hex_view.setFont(_mono)
        self._tabs.addTab(self._hex_view, "Hex")

        # ASCII tab
        self._ascii_view = QTextEdit()
        self._ascii_view.setReadOnly(True)
        self._ascii_view.setFont(_mono)
        self._tabs.addTab(self._ascii_view, "ASCII")

        layout.addWidget(self._tabs)

        # Legend — uses CSS colours, text comes from LM so it translates
        legend_layout = QHBoxLayout()
        self._legend_items = [
            ("■ Header", "bits_widget.legend_header"),
            ("■ Payload", "bits_widget.legend_payload"),
            ("■ Raw bits", "bits_widget.legend_raw"),
        ]
        self._legend_labels = []
        for default_text, _tk in self._legend_items:
            lbl = QLabel(default_text)
            legend_layout.addWidget(lbl)
            self._legend_labels.append(lbl)
        legend_layout.addStretch()
        layout.addLayout(legend_layout)

    def set_raw_bits(self, bits: np.ndarray, bit_rate: float = 0.0):
        """Display raw demodulated bits."""
        self._raw_bits = bits
        self._corrected_bits = None
        self._header_offset = 0
        self._payload_offset = 0
        total = len(bits)
        self._lbl_total.setText(f"Total bits: {total:,}")
        self._lbl_errors.setText("Errors: — (run FEC Decode to check)")
        self._lbl_header.setText("Header @ bit: —")
        self._lbl_payload.setText("Payload @ bit: —")
        self._render_bits(bits, header_off=None, payload_off=None,
                          color=self._muted_color)

    def set_corrected_bits(self, bits: np.ndarray,
                            errors_detected: int,
                            errors_corrected: int):
        """Update with FEC-corrected bits."""
        self._corrected_bits = bits
        self._lbl_errors.setText(
            f"Errors detected: {errors_detected}  corrected: {errors_corrected}")
        self._render_bits(bits,
                          header_off=self._header_offset or None,
                          payload_off=self._payload_offset or None,
                          color=self._text_color)

    def set_correlation_result(self, header_offset: int,
                                payload_offset: int, sync_name: str = ""):
        """Highlight header and payload regions."""
        self._header_offset = header_offset
        self._payload_offset = payload_offset
        self._lbl_header.setText(f"Header @ bit: {header_offset}")
        self._lbl_payload.setText(f"Payload @ bit: {payload_offset}"
                                   + (f"  [{sync_name}]" if sync_name else ""))

        bits = self._corrected_bits if self._corrected_bits is not None \
            else self._raw_bits
        if bits is not None:
            self._render_bits(bits, header_off=header_offset,
                              payload_off=payload_offset,
                              color=self._text_color if self._corrected_bits is not None
                              else self._muted_color)

    def apply_theme(self, mode: str):
        """Re-render rich bit text so existing analysis stays readable."""
        self._theme_mode = mode
        self._text_color = '#F2F2F7' if mode == 'dark' else '#111318'
        self._muted_color = '#AEAEB2' if mode == 'dark' else '#4B5563'
        legend_colors = ('#38BDF8', '#A78BFA', self._muted_color)
        for label, color in zip(self._legend_labels, legend_colors):
            label.setStyleSheet(f"color: {color}; font-size: 11px;")
        bits = self._corrected_bits if self._corrected_bits is not None else self._raw_bits
        if bits is not None:
            self._render_bits(bits, self._header_offset or None,
                              self._payload_offset or None,
                              self._text_color if self._corrected_bits is not None else self._muted_color)

    def _render_bits(self, bits: np.ndarray,
                     header_off: int, payload_off: int,
                     color: str):
        """Render bits in all three views with color highlights."""
        if bits is None or len(bits) == 0:
            self._bin_view.clear()
            self._hex_view.clear()
            self._ascii_view.clear()
        else:
            max_display = 8192
            total = len(bits)
            display_bits = bits[:max_display]

            # Show truncation notice in stats bar
            if total > max_display:
                self._lbl_total.setText(
                    f"Total bits: {total:,}  "
                    f"(showing first {max_display:,})")

            # Binary view
            self._bin_view.clear()
            cursor = self._bin_view.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            _render_binary(cursor, display_bits, header_off, payload_off, color,
                           dark=self._theme_mode == 'dark')
            self._bin_view.setTextCursor(cursor)

            # Hex view
            self._hex_view.clear()
            hex_str = bits_to_hex(display_bits)
            self._hex_view.setPlainText(hex_str)

            # ASCII view
            self._ascii_view.clear()
            ascii_str = bits_to_ascii(display_bits)
            self._ascii_view.setPlainText(ascii_str)

    def clear(self):
        self._raw_bits = None
        self._corrected_bits = None
        self._bin_view.clear()
        self._hex_view.clear()
        self._ascii_view.clear()
        self._lbl_total.setText("Total bits: —")
        self._lbl_errors.setText("Errors: —")
        self._lbl_header.setText("Header @ bit: —")
        self._lbl_payload.setText("Payload @ bit: —")


def _render_binary(cursor: QTextCursor, bits: np.ndarray,
                   header_off, payload_off, base_color: str, *, dark: bool):
    """Write bits to cursor with color highlights for header/payload."""
    fmt_default = QTextCharFormat()
    fmt_default.setForeground(QColor(base_color))

    fmt_header = QTextCharFormat()
    fmt_header.setForeground(QColor("#F2F2F7" if dark else "#0F5CB8"))
    fmt_header.setBackground(QColor("#242426" if dark else "#E7F0FD"))

    fmt_payload = QTextCharFormat()
    fmt_payload.setForeground(QColor("#AEAEB2" if dark else "#4B5563"))
    fmt_payload.setBackground(QColor("#1C1C1E" if dark else "#F5F5F7"))

    chunk_size = 8
    line_chunks = 8  # 64 bits per line

    for i, bit in enumerate(bits):
        # Choose format
        if header_off is not None and payload_off is not None:
            if header_off <= i < payload_off:
                fmt = fmt_header
            elif i >= payload_off:
                fmt = fmt_payload
            else:
                fmt = fmt_default
        else:
            fmt = fmt_default

        cursor.setCharFormat(fmt)
        cursor.insertText(str(int(bit)))

        # Space between bytes
        if (i + 1) % chunk_size == 0:
            reset_fmt = QTextCharFormat()
            cursor.setCharFormat(reset_fmt)
            if (i + 1) % (chunk_size * line_chunks) == 0:
                cursor.insertText("\n")
            else:
                cursor.insertText(" ")
