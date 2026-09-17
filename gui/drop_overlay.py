"""
DropZone — a small, quiet drop-target card shown in the file panel
when no signal is loaded. No animation, no full-screen overlay.

Highlights with a subtle border glow only while a file is actually
being dragged over it (the standard OS drag-enter / drag-leave cycle).
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore    import Qt, pyqtSignal
from PyQt6.QtGui     import QPainter, QColor, QPen, QBrush, QFont
from gui.language_manager import LM
from gui.icons import icon


class DropOverlay(QWidget):
    """
    Compact drop-zone card.
    API kept identical to the old full-screen overlay so main_window.py
    needs zero changes:  show_overlay() / hide_overlay() / sig_open_file
    """

    sig_open_file = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._drag_active = False
        self._theme_mode = 'dark'
        self._build_ui()
        LM.language_changed.connect(self._retranslate)

    # ── Build ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.setMinimumHeight(72)
        self.setMaximumHeight(88)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Click to open a file, or drag a .wav / .iq file here")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._icon_lbl = QLabel("↥")
        self._icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon_lbl.setStyleSheet("background: transparent;")
        lay.addWidget(self._icon_lbl)

        self._main_lbl = QLabel()
        self._main_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._main_lbl.setStyleSheet(
            "font-size: 11px; font-weight: 600; "
            "letter-spacing: 0.3px; background: transparent;")
        lay.addWidget(self._main_lbl)

        self._sub_lbl = QLabel(".wav  ·  .iq  ·  .bin")
        self._sub_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sub_lbl.setStyleSheet("font-size: 10px; background: transparent;")
        lay.addWidget(self._sub_lbl)

        self._retranslate()
        self.apply_theme('dark')

    def apply_theme(self, mode: str):
        self._theme_mode = mode
        color = '#111318' if mode == 'light' else '#F2F2F7'
        self._icon_lbl.setPixmap(icon('open', color).pixmap(18, 18))
        self.update()

    def _retranslate(self):
        try:
            self._main_lbl.setText(LM.t("drop_zone.drop_here"))
        except Exception:
            self._main_lbl.setText("Drop file here")

    # ── Public API (unchanged from old overlay) ───────────────────────────
    def show_overlay(self):
        self.setVisible(True)

    def hide_overlay(self):
        self.setVisible(False)

    # ── Painting ──────────────────────────────────────────────────────────
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        r    = 8          # corner radius

        light = self._theme_mode == 'light'
        if self._drag_active:
            p.setBrush(QBrush(QColor("#E7F0FD" if light else "#202023")))
            pen = QPen(QColor("#1677E8" if light else "#AEAEB2"), 1.5)
        else:
            p.setBrush(QBrush(QColor("#FAFAFA" if light else "#101011")))
            pen = QPen(QColor("#D1D1D6" if light else "#38383A"), 1.0, Qt.PenStyle.DashLine)
            pen.setDashPattern([6, 4])

        p.setPen(pen)
        p.drawRoundedRect(2, 2, w - 4, h - 4, r, r)
        p.end()

    # ── Interaction ───────────────────────────────────────────────────────
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.sig_open_file.emit()

    def enterEvent(self, event):
        # Light hover — just redraw with slightly brighter border
        self._drag_active = False
        self.update()

    def leaveEvent(self, event):
        self._drag_active = False
        self.update()

    # ── Drag & drop ───────────────────────────────────────────────────────
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            url  = event.mimeData().urls()[0]
            path = url.toLocalFile().lower()
            if any(path.endswith(ext) for ext in
                   ('.wav', '.iq', '.bin', '.cf32', '.cs16', '.cs8', '.cfile')):
                event.acceptProposedAction()
                self._drag_active = True
                self.update()
                return
        event.ignore()

    def dragLeaveEvent(self, event):
        self._drag_active = False
        self.update()

    def dropEvent(self, event):
        self._drag_active = False
        self.update()
        urls = event.mimeData().urls()
        if not urls:
            event.ignore()
            return
        path = urls[0].toLocalFile()
        event.acceptProposedAction()
        mw = self.window()
        if hasattr(mw, '_load_file'):
            mw._load_file(path)
