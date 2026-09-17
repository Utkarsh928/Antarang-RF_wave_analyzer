"""
Configuration management for Tarang AI.
Loads settings from .env, environment variables, and persistent QSettings.
Never logs or prints secret API keys.
"""
try:
    from core.ai.runtime_manager import safe_backend_init
    safe_backend_init()
except Exception:
    pass

import os
from pathlib import Path
from typing import Optional, Dict, Any


def _load_dotenv_file(filepath: Path) -> Dict[str, str]:
    """Lightweight zero-dependency .env parser."""
    env_vars = {}
    if not filepath.exists() or not filepath.is_file():
        return env_vars
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key:
                    env_vars[key] = val
    except Exception:
        pass
    return env_vars


class AIConfig:
    """Centralized AI configuration."""

    # Provider identifiers
    PROVIDER_GEMINI = "Gemini"
    PROVIDER_GROQ = "Groq"
    PROVIDER_OPENROUTER = "OpenRouter"
    PROVIDER_CEREBRAS = "Cerebras"
    PROVIDER_LLAVA = "Offline AI"  # Retained constant name for internal backwards compatibility
    PROVIDER_OFFLINE = "Offline AI"

    # Modes
    MODE_AUTO = "AUTO"
    MODE_ONLINE = "ONLINE"
    MODE_OFFLINE = "OFFLINE"

    # Thinking budgets for Gemini 3.8 Flash
    THINKING_BUDGETS = {
        "low": 1024,
        "medium": 4096,
        "high": 16384,
    }

    # Defaults
    DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
    DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
    DEFAULT_OPENROUTER_MODEL = "openrouter/free"
    DEFAULT_CEREBRAS_MODEL = "llama3.3-70b"
    DEFAULT_OFFLINE_REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    DEFAULT_OFFLINE_MODEL_NAME = "Qwen2.5-1.5B-Instruct"
    DEFAULT_OFFLINE_QUANT = "q4_k_m"

    CONNECT_TIMEOUT = 10
    REQUEST_TIMEOUT = 60
    MAX_RETRIES = 2

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).resolve().parent.parent.parent
        self._dotenv_vars = _load_dotenv_file(self.project_root / ".env")

    def _get_qsetting(self, key: str, default: Any = None) -> Any:
        try:
            from PyQt6.QtCore import QSettings
            s = QSettings("SignalAnalyzerPro", "SignalAnalyzerPro")
            val = s.value(f"ai/{key}", default)
            return val if val is not None else default
        except Exception:
            return default

    def set_qsetting(self, key: str, value: Any):
        try:
            from PyQt6.QtCore import QSettings
            s = QSettings("SignalAnalyzerPro", "SignalAnalyzerPro")
            s.setValue(f"ai/{key}", value)
        except Exception:
            pass

    def get_api_key(self, provider_key_name: str) -> Optional[str]:
        """
        Safely fetch API key without logging or exposing.
        Priority: 1. Process environment, 2. .env file, 3. Persistent QSettings.
        """
        val = os.environ.get(provider_key_name)
        if val and val.strip():
            return val.strip()

        val = self._dotenv_vars.get(provider_key_name)
        if val and val.strip():
            return val.strip()

        val = self._get_qsetting(provider_key_name)
        if val and isinstance(val, str) and val.strip():
            return val.strip()

        return None

    @property
    def gemini_api_key(self) -> Optional[str]:
        return self.get_api_key("GEMINI_API_KEY")

    @property
    def groq_api_key(self) -> Optional[str]:
        return self.get_api_key("GROQ_API_KEY")

    @property
    def openrouter_api_key(self) -> Optional[str]:
        return self.get_api_key("OPENROUTER_API_KEY")

    @property
    def cerebras_api_key(self) -> Optional[str]:
        return self.get_api_key("CEREBRAS_API_KEY")

    @property
    def ai_mode(self) -> str:
        mode = self._get_qsetting("mode", self.MODE_AUTO)
        return str(mode).upper() if mode in (self.MODE_AUTO, self.MODE_ONLINE, self.MODE_OFFLINE) else self.MODE_AUTO

    @property
    def preferred_online_provider(self) -> str:
        prov = self._get_qsetting("preferred_provider", "Automatic")
        return str(prov)

    @property
    def gemini_model(self) -> str:
        return os.environ.get("GEMINI_MODEL") or self._get_qsetting("gemini_model", self.DEFAULT_GEMINI_MODEL)

    @property
    def groq_model(self) -> str:
        return os.environ.get("GROQ_MODEL") or self._get_qsetting("groq_model", self.DEFAULT_GROQ_MODEL)

    @property
    def openrouter_model(self) -> str:
        return os.environ.get("OPENROUTER_MODEL") or self._get_qsetting("openrouter_model", self.DEFAULT_OPENROUTER_MODEL)

    @property
    def cerebras_model(self) -> str:
        return os.environ.get("CEREBRAS_MODEL") or self._get_qsetting("cerebras_model", self.DEFAULT_CEREBRAS_MODEL)

    @property
    def thinking_level(self) -> str:
        level = self._get_qsetting("thinking_level", "medium")
        return str(level).lower() if str(level).lower() in self.THINKING_BUDGETS else "medium"

    @property
    def thinking_budget(self) -> int:
        return self.THINKING_BUDGETS.get(self.thinking_level, 4096)

    @property
    def offline_model_dir(self) -> Path:
        """
        User-local application model directory for Qwen2.5-1.5B-Instruct GGUF.
        Windows: %LOCALAPPDATA%\Tarang\models\qwen2.5-1.5b-instruct\
        Non-Windows: ~/.local/share/Tarang/models/qwen2.5-1.5b-instruct/
        """
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            base = Path(local_app_data) / "Tarang" / "models" / "qwen2.5-1.5b-instruct"
        else:
            base = Path.home() / ".local" / "share" / "Tarang" / "models" / "qwen2.5-1.5b-instruct"
        return base
