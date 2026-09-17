"""
Provider Manager and Automatic Fallback Orchestrator for Tarang.
Coordinates all online providers (Gemini, Groq, OpenRouter, Cerebras) and offline LLaVA.
Implements bounded retries, exponential backoff, detailed audit records, and safe fallback.
"""
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, Callable

from core.ai.provider_base import AIProvider, AIResponse
from core.ai.config import AIConfig
from core.ai.system_prompt import TARANG_SYSTEM_PROMPT
from core.ai.gemini_provider import GeminiProvider
from core.ai.groq_provider import GroqProvider
from core.ai.openrouter_provider import OpenRouterProvider
from core.ai.cerebras_provider import CerebrasProvider
from core.ai.local_llava_provider import LocalLlavaProvider, OfflineQwenProvider
from core.ai.model_manager import AIModelManager

logger = logging.getLogger("Tarang.AI.ProviderManager")


@dataclass
class FallbackRecord:
    timestamp: str
    provider: str
    model: str
    error_category: str
    error_message: str
    next_provider: Optional[str] = None


class AIProviderManager:
    """
    Central dispatch and lifecycle coordinator for Tarang AI.
    Handles mode transitions (AUTO, ONLINE, OFFLINE) and fault-tolerant routing.
    """

    # Default fallback priority for online providers
    DEFAULT_ONLINE_PRIORITY = [
        AIConfig.PROVIDER_GEMINI,
        AIConfig.PROVIDER_GROQ,
        AIConfig.PROVIDER_OPENROUTER,
        AIConfig.PROVIDER_CEREBRAS,
    ]

    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config or AIConfig()
        self.model_manager = AIModelManager(self.config)

        # Initialize provider instances
        offline_provider = OfflineQwenProvider(self.config, self.model_manager)
        self.providers: Dict[str, AIProvider] = {
            AIConfig.PROVIDER_GEMINI: GeminiProvider(self.config),
            AIConfig.PROVIDER_GROQ: GroqProvider(self.config),
            AIConfig.PROVIDER_OPENROUTER: OpenRouterProvider(self.config),
            AIConfig.PROVIDER_CEREBRAS: CerebrasProvider(self.config),
            AIConfig.PROVIDER_LLAVA: offline_provider,
            AIConfig.PROVIDER_OFFLINE: offline_provider,
        }

        self.last_fallback_records: List[FallbackRecord] = []
        self._current_active_provider: Optional[str] = None

    @property
    def current_mode(self) -> str:
        return self.config.ai_mode

    def get_provider(self, name: str) -> Optional[AIProvider]:
        return self.providers.get(name)

    def get_provider_statuses(self) -> Dict[str, Dict[str, Any]]:
        """
        Fast inspection of local configuration without making network requests.
        Prevents burning API tokens on application startup.
        """
        statuses = {}
        for name, provider in self.providers.items():
            info = provider.get_model_info()
            configured = provider.is_configured()
            statuses[name] = {
                "name": name,
                "model": info.get("model", ""),
                "configured": configured,
                "status_text": "Configured" if configured else "Not configured",
                "multimodal": info.get("multimodal", False),
                "is_offline": (name == AIConfig.PROVIDER_LLAVA),
                "details": info,
            }
        return statuses

    def get_active_provider_info(self) -> Dict[str, Any]:
        """Returns details about the currently active or primary available provider."""
        mode = self.current_mode
        active_name = self._current_active_provider

        if mode == AIConfig.MODE_OFFLINE:
            prov = self.providers[AIConfig.PROVIDER_LLAVA]
            return {
                "provider": prov.name,
                "model": prov.default_model,
                "mode": mode,
                "configured": prov.is_configured(),
                "is_offline": True,
            }

        # Check preferred provider first if specified
        pref = self.config.preferred_online_provider
        if pref in self.providers and self.providers[pref].is_configured() and mode != AIConfig.MODE_OFFLINE:
            p = self.providers[pref]
            return {
                "provider": p.name,
                "model": p.default_model,
                "mode": mode,
                "configured": True,
                "is_offline": False,
            }

        # Check default priority list
        for name in self.DEFAULT_ONLINE_PRIORITY:
            p = self.providers[name]
            if p.is_configured():
                return {
                    "provider": p.name,
                    "model": p.default_model,
                    "mode": mode,
                    "configured": True,
                    "is_offline": False,
                }

        # If in AUTO mode and offline model is installed
        if mode == AIConfig.MODE_AUTO and self.providers[AIConfig.PROVIDER_LLAVA].is_configured():
            p = self.providers[AIConfig.PROVIDER_LLAVA]
            return {
                "provider": p.name,
                "model": p.default_model,
                "mode": mode,
                "configured": True,
                "is_offline": True,
            }

        return {
            "provider": "None",
            "model": "None configured",
            "mode": mode,
            "configured": False,
            "is_offline": False,
        }

    def _determine_execution_queue(self, requires_vision: bool = False) -> List[AIProvider]:
        """
        Determines the ordered list of providers to attempt based on mode,
        configuration, and capabilities.
        CRITICAL RULE: ONLY attempt providers that are actually configured!
        """
        mode = self.current_mode
        queue: List[AIProvider] = []

        if mode == AIConfig.MODE_OFFLINE:
            p = self.providers[AIConfig.PROVIDER_LLAVA]
            if p.is_configured():
                queue.append(p)
            return queue

        # ONLINE or AUTO mode
        pref = self.config.preferred_online_provider
        if pref in self.providers and pref != "Automatic":
            p = self.providers[pref]
            if p.is_configured():
                queue.append(p)

        for name in self.DEFAULT_ONLINE_PRIORITY:
            p = self.providers[name]
            if p.is_configured() and p not in queue:
                queue.append(p)

        # In AUTO mode, if online providers fail, allow offline LLaVA if installed
        if mode == AIConfig.MODE_AUTO:
            local_p = self.providers[AIConfig.PROVIDER_LLAVA]
            if local_p.is_configured() and local_p not in queue:
                queue.append(local_p)

        return queue

    def _categorize_error(self, err: str) -> str:
        err_lower = (err or "").lower()
        if any(w in err_lower for w in ("401", "403", "auth", "api key", "unauthorized", "forbidden", "permission_denied", "denied access")):
            return "AUTHENTICATION"
        if any(w in err_lower for w in ("429", "quota", "rate limit", "credits", "402", "payment", "billing")):
            return "RATE_LIMIT_OR_QUOTA"
        if any(w in err_lower for w in ("timeout", "timed out")):
            return "TIMEOUT"
        if any(w in err_lower for w in ("network", "connection", "connect")):
            return "NETWORK_ERROR"
        if any(w in err_lower for w in ("404", "model_not_found", "decommissioned")):
            return "MODEL_UNAVAILABLE"
        if any(w in err_lower for w in ("500", "502", "503", "504", "server error")):
            return "SERVER_ERROR"
        return "RUNTIME_ERROR"

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        on_token: Optional[Callable[[str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> AIResponse:
        """
        Execute request across configured providers with bounded retries and fallback.
        """
        self.last_fallback_records.clear()
        has_image = bool(image_bytes and len(image_bytes) > 0)
        queue = self._determine_execution_queue(requires_vision=has_image)

        if not queue:
            mode = self.current_mode
            if mode == AIConfig.MODE_OFFLINE:
                msg = (
                    "Offline AI mode is selected, but the local model is not installed.\n"
                    "Click 'Offline AI' in the panel to download and install it."
                )
            else:
                msg = (
                    "Online AI is currently unavailable.\n"
                    "You can also use Offline AI for on-device signal analysis."
                )
            return AIResponse(
                provider="None",
                model="None",
                success=False,
                error=msg,
            )

        active_sys_prompt = system_prompt or TARANG_SYSTEM_PROMPT
        max_retries = self.config.MAX_RETRIES

        for idx, provider in enumerate(queue):
            if cancel_check and cancel_check():
                return AIResponse(provider=provider.name, model=provider.default_model, success=False, error="Cancelled by user.")

            self._current_active_provider = provider.name
            logger.info(f"Attempting inference via provider: {provider.name}")

            # Bounded retry loop for transient network/server hiccups
            for attempt in range(max_retries):
                if cancel_check and cancel_check():
                    return AIResponse(provider=provider.name, model=provider.default_model, success=False, error="Cancelled by user.")

                if has_image:
                    response = provider.generate_with_image(
                        prompt=prompt,
                        image_bytes=image_bytes,
                        context=context,
                        system_prompt=active_sys_prompt,
                        on_token=on_token,
                        cancel_check=cancel_check,
                    )
                else:
                    response = provider.generate(
                        prompt=prompt,
                        context=context,
                        system_prompt=active_sys_prompt,
                        on_token=on_token,
                        cancel_check=cancel_check,
                    )

                if response.success:
                    if idx > 0:
                        response.fallback_from = queue[0].name
                    return response

                err_cat = self._categorize_error(response.error or "")
                logger.warning(
                    f"Provider {provider.name} failed (attempt {attempt + 1}/{max_retries}). "
                    f"Category: {err_cat}. Reason: {response.error}"
                )

                # Fatal auth, quota/billing, or model errors shouldn't retry the same provider
                if err_cat in ("AUTHENTICATION", "MODEL_UNAVAILABLE", "RATE_LIMIT_OR_QUOTA"):
                    break

                # Exponential backoff before next attempt on same provider
                if attempt < max_retries - 1:
                    sleep_sec = 1.0 * (2 ** attempt)
                    time.sleep(sleep_sec)

            # Record fallback event
            next_name = queue[idx + 1].name if idx + 1 < len(queue) else None
            record = FallbackRecord(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                provider=provider.name,
                model=provider.default_model,
                error_category=self._categorize_error(response.error or ""),
                error_message=response.error or "Unknown failure",
                next_provider=next_name,
            )
            self.last_fallback_records.append(record)

            if next_name:
                logger.info(f"Falling back from {provider.name} to {next_name}...")

        # All configured providers failed
        combined_error = (
            "Online AI is currently unavailable.\n\n"
            "Please check your internet connection or use Offline AI for local signal analysis."
        )

        return AIResponse(
            provider="Online AI",
            model="None",
            success=False,
            error=combined_error,
        )

    def test_provider(self, provider_name: str) -> Tuple[bool, str]:
        """Explicit on-demand provider health check invoked from Settings dialog."""
        prov = self.providers.get(provider_name)
        if not prov:
            return False, f"Unknown provider: {provider_name}"
        return prov.health_check()
