"""
LanguageSelectorWidget — toolbar button that opens a language-picker dialog.

Usage:
    from gui.language_selector import LanguageSelectorWidget
    toolbar_layout.addWidget(LanguageSelectorWidget())
"""
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QDialog,
    QVBoxLayout, QHBoxLayout as QHBox, QLabel,
    QListWidget, QListWidgetItem, QFrame
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont
from gui.theme import apply_window_chrome, window_theme_mode

from gui.language_manager import LM


# ─────────────────────────────────────────────────────────────────────────────
# Toolbar button
# ─────────────────────────────────────────────────────────────────────────────
class LanguageSelectorWidget(QWidget):
    """Compact toolbar widget.  Click → language picker dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._btn = QPushButton()
        self._btn.setObjectName("btn_language")
        self._btn.setToolTip("Select application language / \u092d\u093e\u0937\u093e \u091a\u0941\u0928\u0947\u0902")
        self._btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn.setMinimumWidth(130)
        self._btn.clicked.connect(self._open_dialog)
        layout.addWidget(self._btn)

        LM.language_changed.connect(self._update_label)
        self._update_label()

    def _update_label(self):
        self._btn.setText(LM.t("toolbar.language") + "  [" + LM.current_name + "]")

    def _open_dialog(self):
        dlg = _LanguageDialog(self)
        dlg.exec()


# ─────────────────────────────────────────────────────────────────────────────
# Language-picker dialog
# ─────────────────────────────────────────────────────────────────────────────
class _LanguageDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(LM.t("language_dialog.title"))
        self.setMinimumWidth(400)
        self.setMinimumHeight(480)
        self.setModal(True)
        self.setStyleSheet(parent.window().styleSheet() if parent else "")
        apply_window_chrome(self, window_theme_mode(parent))
        self._build_ui()

    # ── Build ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(18, 18, 18, 18)

        # ── Header ──────────────────────────────────────────────────────
        title_lbl = QLabel(LM.t("language_dialog.title"))
        font = QFont()
        font.setPointSize(15)
        font.setBold(True)
        title_lbl.setFont(font)
        title_lbl.setStyleSheet("padding-bottom: 4px;")
        root.addWidget(title_lbl)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("separator")
        root.addWidget(sep)

        instr = QLabel(LM.t("language_dialog.select_label"))
        instr.setStyleSheet("font-size: 12px; padding: 2px 0;")
        root.addWidget(instr)

        # ── Language list ────────────────────────────────────────────────
        self._list = QListWidget()
        self._list.setSpacing(2)

        for lang in LM.available_languages():
            text = "  " + lang["name"]
            if lang["code"] == LM.current_code:
                text += "   \u2713"          # ✓ checkmark
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, lang["code"])
            if lang["code"] == LM.current_code:
                f = QFont()
                f.setBold(True)
                item.setFont(f)
            self._list.addItem(item)

        self._list.itemDoubleClicked.connect(self._apply_and_close)
        root.addWidget(self._list, stretch=1)

        # ── Note ────────────────────────────────────────────────────────
        note = QLabel(LM.t("language_dialog.restart_note"))
        note.setStyleSheet("font-size: 11px; padding-top: 2px;")
        root.addWidget(note)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setObjectName("separator")
        root.addWidget(sep2)

        # ── Buttons (plain QPushButtons — no QDialogButtonBox) ───────────
        btn_row = QHBox()
        btn_row.setSpacing(10)
        btn_row.addStretch()

        self._btn_close = QPushButton(LM.t("dialogs.close"))
        self._btn_close.clicked.connect(self.reject)
        btn_row.addWidget(self._btn_close)

        self._btn_apply = QPushButton("Apply")
        self._btn_apply.setObjectName("btn_primary")
        self._btn_apply.setDefault(True)
        self._btn_apply.clicked.connect(self._apply_and_close)
        btn_row.addWidget(self._btn_apply)

        root.addLayout(btn_row)

        # Pre-select the current language
        self._select_current()

    # ── Helpers ────────────────────────────────────────────────────────────
    def _select_current(self):
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == LM.current_code:
                self._list.setCurrentRow(i)
                self._list.scrollToItem(item)
                break

    def _apply_and_close(self):
        item = self._list.currentItem()
        if item is None:
            self.accept()
            return
        code = item.data(Qt.ItemDataRole.UserRole)
        if code and code != LM.current_code:
            LM.set_language(code)      # fires language_changed → whole UI retranslates
        self.accept()
