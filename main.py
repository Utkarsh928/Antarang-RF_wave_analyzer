"""Antarang application entry point."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")
if sys.platform == "win32":
    os.environ.setdefault("QT_MEDIA_BACKEND", "windows")

# Initialize local inference runtime safely before Qt C runtime initialization
try:
    from core.ai.runtime_manager import safe_backend_init
    safe_backend_init()
except Exception:
    pass

from startup import ensure_runtime_dependencies

try:
    ensure_runtime_dependencies()
except RuntimeError as exc:
    print(f"Unable to start Antarang:\n{exc}", file=sys.stderr)
    sys.exit(1)

from PyQt6.QtCore import Qt, QSettings, QTimer, QUrl
from PyQt6.QtWidgets import QApplication, QLabel, QProgressBar, QVBoxLayout, QWidget

try:
    from PyQt6.QtMultimedia import QMediaFormat, QMediaPlayer
    from PyQt6.QtMultimediaWidgets import QVideoWidget
except ImportError:  # Keep startup working on installations without multimedia.
    QMediaFormat = QMediaPlayer = QVideoWidget = None

from gui.branding import SPLASH_VIDEO, logo_pixmap, window_icon
from gui.main_window import MainWindow
from gui.theme import theme_stylesheet


class AntarangSplash(QWidget):
    """Frameless startup video with a theme-aware static fallback."""
    def __init__(self, mode: str):
        super().__init__(flags=Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
        self._player = None
        self.setFixedSize(420, 472)
        is_light = mode == "light"
        self.setStyleSheet("background: #FFFFFF;" if is_light else "background: #000000;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._video = QVideoWidget() if QVideoWidget else None
        self._fallback = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self._fallback.setFixedHeight(420)
        self._fallback.setPixmap(logo_pixmap(mode, 360, 104))
        self._fallback.hide()
        if self._video:
            self._video.setFixedHeight(420)
            self._video.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
            self._video.setStyleSheet("background: #000000;")
            layout.addWidget(self._video)
        layout.addWidget(self._fallback)
        self._loading = QLabel("Loading signal analysis workspace…",
                               alignment=Qt.AlignmentFlag.AlignCenter)
        self._loading.setFixedHeight(49)
        self._loading.setStyleSheet(
            f"background: {'#FFFFFF' if is_light else '#000000'}; "
            f"color: {'#3A3A3C' if is_light else '#AEAEB2'}; font-size: 12px;")
        layout.addWidget(self._loading)
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(3)
        self._progress.setStyleSheet(
            "QProgressBar { border: 0; background: #242426; }"
            "QProgressBar::chunk { background: #5E5CE6; }")
        layout.addWidget(self._progress)
        self._start_video()

    def _start_video(self):
        if not self._video or not SPLASH_VIDEO.is_file() or not QMediaFormat().supportedFileFormats(
                QMediaFormat.ConversionMode.Decode):
            self._show_fallback()
            return
        try:
            self._player = QMediaPlayer(self)
            self._player.setVideoOutput(self._video)
            self._player.errorOccurred.connect(self._show_fallback)
            self._player.setSource(QUrl.fromLocalFile(str(SPLASH_VIDEO)))
            self._player.play()
        except Exception:
            self._show_fallback()

    def _show_fallback(self, *_):
        if self._player:
            self._player.stop()
        if self._video:
            self._video.hide()
        self._fallback.show()

    def showEvent(self, event):
        super().showEvent(event)
        screen = self.screen() or QApplication.primaryScreen()
        if screen:
            self.move(screen.availableGeometry().center() - self.rect().center())

    def closeEvent(self, event):
        if self._player:
            self._player.stop()
        super().closeEvent(event)


def main():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Antarang.SignalAnalyzer.1.0")
        except Exception:
            pass

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName("Antarang")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("SignalAnalyzerPro")
    mode = str(QSettings("SignalAnalyzerPro", "SignalAnalyzerPro").value("ui/theme", "dark"))
    app.setWindowIcon(window_icon(mode))
    app.setStyleSheet(theme_stylesheet(mode))

    splash = AntarangSplash(mode)
    splash.show()

    def launch_main_window():
        window = MainWindow()
        app.main_window = window
        window.show()
        splash.close()

    QTimer.singleShot(1500, launch_main_window)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
