"""
LanguageManager — Singleton for application-wide translation support.

Usage:
    from gui.language_manager import LM
    LM.set_language('hi')
    label = LM.t('toolbar.open_file')       # "फ़ाइल खोलें"
    LM.language_changed.connect(my_slot)    # fired on every language switch
"""
import json
import os
import re
from typing import Any, Callable, Dict, List

from PyQt6.QtCore import QObject, QSettings, pyqtSignal


def _translations_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "translations")


class _LanguageManager(QObject):
    """
    Singleton.  Access via module-level `LM`.
    Emits `language_changed` whenever the active language switches so that
    every widget can update its labels in-place without restarting.
    """
    language_changed = pyqtSignal()          # connected by every retranslatable widget

    # ---------------------------------------------------------------------------
    # All supported languages (code → display name in that language)
    # ---------------------------------------------------------------------------
    LANGUAGES: List[Dict[str, str]] = [
        {"code": "en", "name": "English"},
        {"code": "hi", "name": "हिन्दी"},
        {"code": "bn", "name": "বাংলা"},
        {"code": "te", "name": "తెలుగు"},
        {"code": "mr", "name": "मराठी"},
        {"code": "ta", "name": "தமிழ்"},
        {"code": "gu", "name": "ગુજરાતી"},
        {"code": "kn", "name": "ಕನ್ನಡ"},
        {"code": "ml", "name": "മലയാളം"},
        {"code": "pa", "name": "ਪੰਜਾਬੀ"},
        {"code": "or", "name": "ଓଡ଼ିଆ"},
        {"code": "as", "name": "অসমীয়া"},
        {"code": "ur", "name": "اردو"},
    ]

    def __init__(self):
        super().__init__()
        self._code: str = "en"
        self._data: Dict = {}
        self._en_data: Dict = {}        # English fallback always loaded
        self._settings = QSettings("SignalAnalyzerPro", "SignalAnalyzerPro")

        # Load English fallback first
        self._en_data = self._load_file("en")

        # Restore last-used language
        saved = self._settings.value("language", "en")
        self.set_language(saved, emit=False)

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def current_code(self) -> str:
        return self._code

    @property
    def current_name(self) -> str:
        return self._data.get("lang_name", "English")

    def set_language(self, code: str, emit: bool = True) -> bool:
        """Switch to `code`.  Returns True on success."""
        if code == self._code and self._data:
            return True
        data = self._load_file(code)
        if data is None:
            return False
        self._code = code
        self._data = data
        self._settings.setValue("language", code)
        if emit:
            self.language_changed.emit()
        return True

    def t(self, key: str, **kwargs) -> str:
        """
        Translate a dot-separated key, e.g. t('toolbar.open_file').
        Falls back to English, then returns the key itself if missing.
        Supports keyword substitution: t('status.loading', filename='x.wav')
        """
        value = self._lookup(self._data, key)
        if value is None:
            value = self._lookup(self._en_data, key)
        if value is None:
            return key                          # last-resort: show the key
        if kwargs:
            try:
                return value.format(**kwargs)
            except (KeyError, ValueError):
                return value
        return value

    def available_languages(self) -> List[Dict[str, str]]:
        """Return list of {code, name} for languages with a translation file."""
        result = []
        td = _translations_dir()
        for lang in self.LANGUAGES:
            if os.path.exists(os.path.join(td, f"{lang['code']}.json")):
                result.append(lang)
        return result

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _load_file(self, code: str) -> Dict | None:
        path = os.path.join(_translations_dir(), f"{code}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return self._without_icons(json.load(f))
        except Exception:
            return None

    @staticmethod
    def _without_icons(value):
        """Translations are labels, not icon assets; keep their text emoji-free."""
        if isinstance(value, dict):
            return {k: _LanguageManager._without_icons(v) for k, v in value.items()}
        if isinstance(value, str):
            return re.sub(r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF]", "", value).strip()
        return value

    @staticmethod
    def _lookup(data: Dict, key: str) -> Any:
        """Walk dot-separated key through nested dicts."""
        parts = key.split(".")
        node = data
        for part in parts:
            if not isinstance(node, dict):
                return None
            node = node.get(part)
            if node is None:
                return None
        return node if isinstance(node, str) else None


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere
# ---------------------------------------------------------------------------
LM = _LanguageManager()


def tr(key: str, **kwargs) -> str:
    """Convenience alias:  from gui.language_manager import tr"""
    return LM.t(key, **kwargs)
