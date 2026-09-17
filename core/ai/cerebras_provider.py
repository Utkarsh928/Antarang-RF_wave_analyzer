"""
Cerebras API Provider for Tarang.
Uses the ultra-fast Cerebras Cloud REST API (OpenAI-compatible).
Fast fallback provider for high-throughput inference.
"""
import json
import logging
from typing import Optional, Tuple, Dict, Any, Callable
import requests

from core.ai.provider_base import AIProvider, AIResponse
from core.ai.config import AIConfig

logger = logging.getLogger("Tarang.AI.Cerebras")


class CerebrasProvider(AIProvider):
    """
    Cerebras Cloud REST Provider.
    Endpoint: https://api.cerebras.ai/v1/chat/completions
    """

    BASE_URL = "https://api.cerebras.ai/v1"

    FALLBACK_MODELS = [
        "gpt-oss-120b",
        "qwen-3.8-27b",
        "llama3.3-70b",
    ]

    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config or AIConfig()
        super().__init__(
            name=AIConfig.PROVIDER_CEREBRAS,
            default_model=self.config.cerebras_model,
            is_multimodal=False,
        )

    def is_configured(self) -> bool:
        key = self.config.cerebras_api_key
        return bool(key and len(key.strip()) > 5)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.config.cerebras_model,
            "configured": self.is_configured(),
            "multimodal": self.is_multimodal,
            "notes": "Cerebras wafer-scale fast inference engine.",
        }

    def health_check(self) -> Tuple[bool, str]:
        """Check API key validity by querying Cerebras /models endpoint."""
        if not self.is_configured():
            return False, "Cerebras API key not configured (CEREBRAS_API_KEY)."

        key = self.config.cerebras_api_key
        url = f"{self.BASE_URL}/models"
        headers = {
            "Authorization": f"Bearer {key}",
            "User-Agent": "Tarang-Signal-Analyzer/1.0",
        }

        try:
            resp = requests.get(
                url,
                headers=headers,
                timeout=(self.config.CONNECT_TIMEOUT, 15),
            )
            if resp.status_code == 200:
                return True, f"Connected to Cerebras (Model: {self.config.cerebras_model})"
            elif resp.status_code == 401:
                return False, "Authentication failed: Invalid CEREBRAS_API_KEY."
            elif resp.status_code == 429:
                return False, "Rate limit reached on Cerebras."
            else:
                return False, f"Cerebras returned HTTP {resp.status_code}: {resp.text[:120]}"
        except requests.exceptions.Timeout:
            return False, "Connection timeout to Cerebras API."
        except requests.exceptions.RequestException as e:
            return False, f"Network error connecting to Cerebras: {type(e).__name__}"

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        on_token: Optional[Callable[[str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> AIResponse:
        if not self.is_configured():
            return AIResponse(
                provider=self.name,
                model=self.config.cerebras_model,
                success=False,
                error="Cerebras API key is not configured. Set CEREBRAS_API_KEY in .env or Settings.",
            )

        if cancel_check and cancel_check():
            return AIResponse(provider=self.name, model=self.config.cerebras_model, success=False, error="Cancelled by user.")

        key = self.config.cerebras_api_key
        url = f"{self.BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "Tarang-Signal-Analyzer/1.0",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if context:
            messages.append({"role": "user", "content": f"RF Signal Telemetry:\n```json\n{context}\n```"})
        messages.append({"role": "user", "content": prompt})

        models_to_try = [self.config.cerebras_model]
        for fm in self.FALLBACK_MODELS:
            if fm not in models_to_try:
                models_to_try.append(fm)

        last_error = ""
        for model in models_to_try:
            if cancel_check and cancel_check():
                return AIResponse(provider=self.name, model=model, success=False, error="Cancelled by user.")

            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 4096,
            }

            try:
                logger.info(f"Sending request to Cerebras (model={model})")
                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=(self.config.CONNECT_TIMEOUT, self.config.REQUEST_TIMEOUT),
                )

                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        tokens = data.get("usage", {}).get("total_tokens")
                        if on_token and content:
                            on_token(content)
                        return AIResponse(
                            text=content,
                            provider=self.name,
                            model=model,
                            success=True,
                            tokens_used=tokens,
                        )
                    return AIResponse(
                        provider=self.name,
                        model=model,
                        success=False,
                        error="Empty response choices from Cerebras.",
                    )

                err_msg = resp.text[:200]
                try:
                    err_json = resp.json().get("error", {})
                    err_msg = err_json.get("message", err_msg)
                except Exception:
                    pass

                if resp.status_code == 404 or "model" in str(err_msg).lower():
                    logger.warning(f"Cerebras model {model} not found, trying fallback...")
                    last_error = f"Model {model} unavailable: {err_msg}"
                    continue

                if resp.status_code == 401:
                    return AIResponse(
                        provider=self.name,
                        model=model,
                        success=False,
                        error="Authentication failed: Invalid CEREBRAS_API_KEY.",
                    )
                elif resp.status_code == 429:
                    return AIResponse(
                        provider=self.name,
                        model=model,
                        success=False,
                        error=f"Cerebras rate limit exceeded: {err_msg}",
                    )
                else:
                    return AIResponse(
                        provider=self.name,
                        model=model,
                        success=False,
                        error=f"Cerebras error (HTTP {resp.status_code}): {err_msg}",
                    )

            except requests.exceptions.Timeout:
                return AIResponse(
                    provider=self.name,
                    model=model,
                    success=False,
                    error="Cerebras request timed out.",
                )
            except requests.exceptions.RequestException as e:
                return AIResponse(
                    provider=self.name,
                    model=model,
                    success=False,
                    error=f"Network error with Cerebras: {type(e).__name__}",
                )

        return AIResponse(
            provider=self.name,
            model=self.config.cerebras_model,
            success=False,
            error=f"All attempted Cerebras models failed. Last error: {last_error}",
        )

    def generate_with_image(
        self,
        prompt: str,
        image_bytes: bytes,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        on_token: Optional[Callable[[str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> AIResponse:
        """
        Cerebras text models analyze signal based on structured parameters.
        """
        text_prompt = (
            f"[Note: Plot visualization captured ({len(image_bytes)} bytes PNG). "
            f"As a fast analytical text LLM, analyze the signal parameters provided below.]\n\n"
            f"{prompt}"
        )
        resp = self.generate(
            prompt=text_prompt,
            context=context,
            system_prompt=system_prompt,
            on_token=on_token,
            cancel_check=cancel_check,
        )
        resp.is_multimodal = False
        return resp
