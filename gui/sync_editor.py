"""
Sync Word Database Editor — GUI for adding/removing custom protocols.
Accessible from the Bit Correlation step in the control panel.
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QGroupBox,
    QGridLayout, QMessageBox, QHeaderView, QAbstractItemView,
    QComboBox, QTextEdit
)
from PyQt6.QtCore import Qt
from gui.theme import apply_window_chrome, window_theme_mode
import numpy as np


class SyncWordEditor(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Antarang — Sync Word Database")
        self.setMinimumSize(700, 520)
        self.setStyleSheet(parent.styleSheet() if parent else "")
        apply_window_chrome(self, window_theme_mode(parent))
        self._build_ui()
        self._load_table()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ── Table of all sync words ──────────────────────────────────────
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ['Name', 'Bits (hex)', 'Description', 'Band', 'Type'])
        self._table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self._table)

        # ── Add new sync word ────────────────────────────────────────────
        add_group = QGroupBox("Add Custom Sync Word")
        ag = QGridLayout(add_group)

        ag.addWidget(QLabel("Name:"), 0, 0)
        self._inp_name = QLineEdit()
        self._inp_name.setPlaceholderText("e.g. MY-PROTOCOL-SYNC")
        ag.addWidget(self._inp_name, 0, 1, 1, 3)

        ag.addWidget(QLabel("Bits (choose format):"), 1, 0)
        self._combo_fmt = QComboBox()
        self._combo_fmt.addItems(['Hex (e.g. 7E A5)', 'Binary (e.g. 01111110)'])
        ag.addWidget(self._combo_fmt, 1, 1)

        self._inp_bits = QLineEdit()
        self._inp_bits.setPlaceholderText("7E  or  01111110 10100101")
        ag.addWidget(self._inp_bits, 1, 2, 1, 2)

        ag.addWidget(QLabel("Description:"), 2, 0)
        self._inp_desc = QLineEdit()
        self._inp_desc.setPlaceholderText("Optional description")
        ag.addWidget(self._inp_desc, 2, 1, 1, 3)

        ag.addWidget(QLabel("Band:"), 3, 0)
        self._combo_band = QComboBox()
        self._combo_band.addItems(['any', 'HF', 'VHF', 'UHF', 'SHF'])
        ag.addWidget(self._combo_band, 3, 1)

        ag.addWidget(QLabel("Protocol:"), 3, 2)
        self._inp_protocol = QLineEdit()
        self._inp_protocol.setPlaceholderText("e.g. Custom")
        self._inp_protocol.setText("Custom")
        ag.addWidget(self._inp_protocol, 3, 3)

        self._btn_add = QPushButton("+ Add to Database")
        self._btn_add.setObjectName("btn_primary")
        self._btn_add.clicked.connect(self._add_entry)
        ag.addWidget(self._btn_add, 4, 0, 1, 4)

        layout.addWidget(add_group)

        # ── Buttons ──────────────────────────────────────────────────────
        btn_layout = QHBoxLayout()

        self._btn_delete = QPushButton("Delete Selected")
        self._btn_delete.clicked.connect(self._delete_selected)

        self._btn_test = QPushButton("Test Selected on Current Bits")
        self._btn_test.clicked.connect(self._test_selected)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)

        btn_layout.addWidget(self._btn_delete)
        btn_layout.addWidget(self._btn_test)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

    def _load_table(self):
        from core.bitstream.sync_db import (
            BUILTIN_SYNC_WORDS, load_custom, bits_from_hex)

        self._table.setRowCount(0)

        # Built-in entries
        for name, entry in BUILTIN_SYNC_WORDS.items():
            bits = entry['bits']
            hex_str = ' '.join(
                f'{int("".join(str(b) for b in bits[i:i+8]), 2):02X}'
                for i in range(0, len(bits), 8)
                if len(bits[i:i+8]) == 8
            )
            self._add_row(name, hex_str, entry.get('description', ''),
                          entry.get('band', 'any'),
                          entry.get('protocol', 'Built-in'),
                          deletable=False)

        # Custom entries
        custom = load_custom()
        for name, entry in custom.items():
            bits = entry['bits']
            hex_str = ' '.join(
                f'{int("".join(str(b) for b in bits[i:i+8]), 2):02X}'
                for i in range(0, len(bits), 8)
                if len(bits[i:i+8]) == 8
            )
            self._add_row(name, hex_str, entry.get('description', ''),
                          entry.get('band', 'any'),
                          entry.get('protocol', 'Custom'),
                          deletable=True)

    def _add_row(self, name, hex_str, desc, band, protocol, deletable):
        row = self._table.rowCount()
        self._table.insertRow(row)
        for col, text in enumerate([name, hex_str, desc, band, protocol]):
            item = QTableWidgetItem(text)
            if not deletable:
                item.setForeground(Qt.GlobalColor.gray)
            self._table.setItem(row, col, item)
        # Store deletable flag in user data
        self._table.item(row, 0).setData(Qt.ItemDataRole.UserRole, deletable)

    def _add_entry(self):
        from core.bitstream.sync_db import (
            add_custom, bits_from_hex, bits_from_binary_str)

        name = self._inp_name.text().strip()
        bits_str = self._inp_bits.text().strip()
        desc  = self._inp_desc.text().strip()
        band  = self._combo_band.currentText()
        proto = self._inp_protocol.text().strip() or 'Custom'
        fmt   = self._combo_fmt.currentIndex()

        if not name:
            QMessageBox.warning(self, "Input Error", "Name is required.")
            return
        if not bits_str:
            QMessageBox.warning(self, "Input Error", "Bits are required.")
            return

        try:
            if fmt == 0:
                bits = bits_from_hex(bits_str)
            else:
                bits = bits_from_binary_str(bits_str)
        except Exception as e:
            QMessageBox.warning(self, "Parse Error", f"Cannot parse bits: {e}")
            return

        if len(bits) < 4:
            QMessageBox.warning(self, "Input Error",
                                 "Sync word must be at least 4 bits.")
            return

        add_custom(name, bits, desc, band, proto)
        self._inp_name.clear()
        self._inp_bits.clear()
        self._inp_desc.clear()
        self._load_table()
        QMessageBox.information(
            self, "Added",
            f"'{name}' added to custom database.\n"
            f"Length: {len(bits)} bits")

    def _delete_selected(self):
        from core.bitstream.sync_db import remove_custom

        rows = self._table.selectedItems()
        if not rows:
            return
        row = self._table.currentRow()
        name_item = self._table.item(row, 0)
        if not name_item:
            return

        deletable = name_item.data(Qt.ItemDataRole.UserRole)
        if not deletable:
            QMessageBox.information(
                self, "Cannot Delete",
                "Built-in sync words cannot be deleted.\n"
                "Only custom entries can be removed.")
            return

        name = name_item.text()
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete '{name}' from custom database?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            remove_custom(name)
            self._load_table()

    def _test_selected(self):
        """Quick test: correlate selected sync word against a random bit stream."""
        row = self._table.currentRow()
        if row < 0:
            QMessageBox.information(self, "No Selection",
                                     "Select a sync word to test.")
            return

        from core.bitstream.sync_db import BUILTIN_SYNC_WORDS, load_custom
        from core.bitstream.correlator import correlate_bits

        name = self._table.item(row, 0).text()
        all_words = {**BUILTIN_SYNC_WORDS}
        all_words.update(load_custom())

        if name not in all_words:
            QMessageBox.warning(self, "Not Found",
                                  f"'{name}' not found in database.")
            return

        entry = all_words[name]
        bits_list = entry if isinstance(entry, list) else entry.get('bits', [])
        sync = np.array(bits_list, dtype=np.uint8)

        # Create test bit stream with sync injected at position 20
        test_bits = np.random.randint(0, 2, 200).astype(np.uint8)
        test_bits[20:20 + len(sync)] = sync

        corr, h_off, p_off, matched = correlate_bits(test_bits, sync)
        peak = float(corr.max()) if len(corr) > 0 else 0.0

        QMessageBox.information(
            self, "Test Result",
            f"Sync word: {name}\n"
            f"Length:    {len(sync)} bits\n"
            f"Injected at: bit 20\n"
            f"Detected at: bit {h_off}\n"
            f"Peak correlation: {peak:.1f}/{len(sync)}\n"
            f"Payload starts at: bit {p_off}\n\n"
            f"{'PASS — correctly detected' if abs(h_off - 20) <= 2 else 'MISS'}"
        )
