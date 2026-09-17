"""Tarang AI Architecture Package."""
try:
    from core.ai.runtime_manager import safe_backend_init
    safe_backend_init()
except Exception:
    pass

from core.ai.provider_base import AIProvider, AIResponse
from core.ai.config import AIConfig
from core.ai.context_builder import AIContextBuilder
from core.ai.system_prompt import TARANG_SYSTEM_PROMPT
from core.ai.compatibility import AICompatibilityChecker, CompatibilityResult
from core.ai.model_manager import AIModelManager, ModelStatus
from core.ai.gemini_provider import GeminiProvider
from core.ai.groq_provider import GroqProvider
from core.ai.openrouter_provider import OpenRouterProvider
from core.ai.cerebras_provider import CerebrasProvider
from core.ai.local_llava_provider import LocalLlavaProvider, OfflineQwenProvider
from core.ai.provider_manager import AIProviderManager

__all__ = [
    "AIProvider",
    "AIResponse",
    "AIConfig",
    "AIContextBuilder",
    "TARANG_SYSTEM_PROMPT",
    "AICompatibilityChecker",
    "CompatibilityResult",
    "AIModelManager",
    "ModelStatus",
    "GeminiProvider",
    "GroqProvider",
    "OpenRouterProvider",
    "CerebrasProvider",
    "LocalLlavaProvider",
    "OfflineQwenProvider",
    "AIProviderManager",
]
