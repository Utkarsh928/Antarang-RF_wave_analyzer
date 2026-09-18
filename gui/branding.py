"""Project-local Antarang branding assets."""
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QPixmap


BRANDING_DIR = Path(__file__).resolve().parent.parent / "assets" / "branding"
LIGHT_LOGO = BRANDING_DIR / "antarang-light.png"
DARK_LOGO = BRANDING_DIR / "antarang-dark.png"
SPLASH_VIDEO = BRANDING_DIR / "tarang-splash.mp4"

# The light and dark source files place the wordmark at slightly different
# positions inside their square canvases.
_WORDMARK_CROP = {
    "light": (.16, .415, .675, .19),
    "dark": (.187, .35, .644, .18),
}


def logo_pixmap(mode: str, width: int, height: int) -> QPixmap:
    """Return a sharp, compact version of the supplied logo for Qt labels."""
    normalized_mode = "light" if mode == "light" else "dark"
    source = QPixmap(str(LIGHT_LOGO if normalized_mode == "light" else DARK_LOGO))
    if source.isNull():
        return source
    crop_x, crop_y, crop_width, crop_height = _WORDMARK_CROP[normalized_mode]
    cropped = source.copy(
        int(source.width() * crop_x), int(source.height() * crop_y),
        int(source.width() * crop_width), int(source.height() * crop_height),
    )
    return cropped.scaled(width, height, Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)


def window_icon(mode: str = "dark") -> QIcon:
    """Return the Antarang mark for the native window title bar and taskbar."""
    ico_path = BRANDING_DIR / "tarang.ico"
    if ico_path.is_file():
        icon = QIcon(str(ico_path))
        if not icon.isNull():
            return icon
    source = QPixmap(str(LIGHT_LOGO if mode == "light" else DARK_LOGO))
    if not source.isNull():
        return QIcon(source.copy(
            int(source.width() * .22), int(source.height() * .39),
            int(source.width() * .22), int(source.height() * .22),
        ))
    return QIcon()
