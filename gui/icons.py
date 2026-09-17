"""Small monochrome, Qt-drawn icons used by Antarang's application chrome."""
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap


def icon(name: str, color: str, size: int = 18) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color), 1.7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    r = size - 4
    if name == "open":
        painter.drawRect(3, 6, r - 2, r - 4); painter.drawLine(3, 6, 7, 2); painter.drawLine(7, 2, r + 1, 2)
    elif name == "recent":
        painter.drawArc(3, 3, r, r, 35 * 16, 290 * 16); painter.drawLine(size // 2, size // 2, size // 2, 5); painter.drawLine(size // 2, size // 2, size - 5, size // 2)
    elif name == "radio":
        painter.drawEllipse(7, 7, 4, 4); painter.drawArc(4, 4, 10, 10, 45 * 16, 270 * 16); painter.drawArc(1, 1, 16, 16, 45 * 16, 270 * 16)
    elif name == "export":
        painter.drawLine(size // 2, 2, size // 2, 11); painter.drawLine(5, 8, size // 2, 11); painter.drawLine(size - 5, 8, size // 2, 11); painter.drawRect(3, 12, r, 3)
    elif name == "compare":
        painter.drawRect(2, 4, 6, 10); painter.drawRect(10, 4, 6, 10); painter.drawLine(5, 7, 5, 11); painter.drawLine(13, 7, 13, 11)
    elif name == "history":
        painter.drawArc(3, 3, r, r, 35 * 16, 290 * 16); painter.drawLine(3, 5, 3, 1); painter.drawLine(3, 1, 7, 1); painter.drawLine(size // 2, size // 2, size // 2, 5)
    elif name == "settings":
        painter.drawEllipse(5, 5, 8, 8); painter.drawEllipse(7, 7, 4, 4); [painter.drawLine(*line) for line in ((9, 1, 9, 4), (9, 14, 9, 17), (1, 9, 4, 9), (14, 9, 17, 9))]
    elif name == "brain":
        painter.drawEllipse(3, 3, 12, 12); painter.drawLine(6, 8, 12, 8); painter.drawLine(9, 5, 9, 13)
    elif name == "info":
        painter.drawEllipse(2, 2, 14, 14); painter.drawPoint(9, 5); painter.drawLine(9, 8, 9, 13)
    elif name == "train":
        painter.drawLine(3, 14, 7, 10); painter.drawLine(7, 10, 10, 12); painter.drawLine(10, 12, 15, 5); painter.drawEllipse(2, 13, 2, 2); painter.drawEllipse(14, 4, 2, 2)
    elif name == "reset":
        painter.drawArc(3, 3, r, r, 40 * 16, 275 * 16); painter.drawLine(3, 7, 3, 3); painter.drawLine(3, 3, 7, 3)
    elif name == "mark":
        painter.drawArc(2, 2, 14, 14, 20 * 16, 140 * 16); painter.drawArc(2, 2, 14, 14, 200 * 16, 140 * 16)
    painter.end()
    return QIcon(pixmap)
