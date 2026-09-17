"""
OpenRouter API Provider for Tarang.
Supports zero-cost routing via 'openrouter/free' and custom model overrides.
Preserves user's zero-cost intention and strictly prevents silent upgrades to paid tiers.
"""
import base64
import json
import logging
from typing import Optional, Tuple, Dict, Any, Callable
import requests

from core.ai.provider_base import AIProvider, AIResponse
from core.ai.config import AIConfig

logger = logging.getLogger("Tarang.AI.OpenRouter")


class OpenRouterProvider(AIProvider):
    """
    OpenRouter REST API Provider.
    Endpoint: https://openrouter.ai/api/v1/chat/completions
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config or AIConfig()
        current_model = self.config.openrouter_model
        super().__init__(
            name=AIConfig.PROVIDER_OPENROUTER,
            default_model=current_model,
            is_multimodal=True,
        )

    def is_configured(self) -> bool:
        key = self.config.openrouter_api_key
        return bool(key and len(key.strip()) > 5)

    @property
    def is_free_tier(self) -> bool:
        model = (self.config.openrouter_model or "").lower()
        return "free" in model or model == "openrouter/free"

    def get_model_info(self) -> Dict[str, Any]:
        model = self.config.openrouter_model
        display_label = "OpenRouter Free" if self.is_free_tier else f"OpenRouter ({model})"
        return {
            "provider": self.name,
            "model": model,
            "display_label": display_label,
            "configured": self.is_configured(),
            "is_free": self.is_free_tier,
            "multimodal": self.is_multimodal,
            "notes": "Preserves zero-cost intention when configured with openrouter/free.",
        }

    def health_check(self) -> Tuple[bool, str]:
        """Check OpenRouter API key validity via /auth/key or lightweight models check."""
        if not self.is_configured():
            return False, "OpenRouter API key not configured (OPENROUTER_API_KEY)."

        key = self.config.openrouter_api_key
        url = f"{self.BASE_URL}/auth/key"
        headers = {
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": "https://tarang.rf",
            "X-Title": "Tarang Signal Analyzer",
            "User-Agent": "Tarang-Signal-Analyzer/1.0",
        }

        try:
            resp = requests.get(
                url,
                headers=headers,
                timeout=(self.config.CONNECT_TIMEOUT, 15),
            )
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                label = "Free" if self.is_free_tier else self.config.openrouter_model
                limit_info = data.get("limit_remaining")
                status_str = f"Connected to OpenRouter ({label})"
                if limit_info is not None:
                    status_str += f" [Credits remaining: {limit_info}]"
                return True, status_str
            elif resp.status_code == 401:
                return False, "Authentication failed: Invalid OPENROUTER_API_KEY."
            elif resp.status_code == 402:
                return False, "Payment required: Insufficient OpenRouter credits."
            elif resp.status_code == 429:
                return False, "Rate limited on OpenRouter."
            else:
                return False, f"OpenRouter returned HTTP {resp.status_code}: {resp.text[:120]}"
        except requests.exceptions.Timeout:
            return False, "Connection timeout to OpenRouter API."
        except requests.exceptions.RequestException as e:
            return False, f"Network error connecting to OpenRouter: {type(e).__name__}"

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
                model=self.config.openrouter_model,
                success=False,
                error="OpenRouter API key is not configured. Set OPENROUTER_API_KEY in .env or Settings.",
            )

        if cancel_check and cancel_check():
            return AIResponse(provider=self.name, model=self.config.openrouter_model, success=False, error="Cancelled by user.")

        key = self.config.openrouter_api_key
        url = f"{self.BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://tarang.rf",
            "X-Title": "Tarang Signal Analyzer",
            "User-Agent": "Tarang-Signal-Analyzer/1.0",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if context:
            messages.append({"role": "user", "content": f"RF Analysis Telemetry:\n```json\n{context}\n```"})
        messages.append({"role": "user", "content": prompt})

        model = self.config.openrouter_model
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 4096,
        }

        try:
            logger.info(f"Sending request to OpenRouter (model={model}, is_free={self.is_free_tier})")
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
                    error="OpenRouter returned empty choices.",
                )

            err_msg = resp.text[:250]
            try:
                err_json = resp.json().get("error", {})
                err_msg = err_json.get("message", err_msg)
            except Exception:
                pass

            if resp.status_code == 401:
                return AIResponse(
                    provider=self.name,
                    model=model,
                    success=False,
                    error="Authentication failed: Invalid OPENROUTER_API_KEY.",
                )
            elif resp.status_code == 402:
                return AIResponse(
                    provider=self.name,
                    model=model,
                    success=False,
                    error="OpenRouter payment required or zero free credits left.",
                )
            elif resp.status_code == 429:
                return AIResponse(
                    provider=self.name,
                    model=model,
                    success=False,
                    error=f"OpenRouter rate limit reached: {err_msg}",
                )
            else:
                return AIResponse(
                    provider=self.name,
                    model=model,
                    success=False,
                    error=f"OpenRouter error (HTTP {resp.status_code}): {err_msg}",
                )

        except requests.exceptions.Timeout:
            return AIResponse(
                provider=self.name,
                model=model,
                success=False,
                error="OpenRouter request timed out.",
            )
        except requests.exceptions.RequestException as e:
            return AIResponse(
                provider=self.name,
                model=model,
                success=False,
                error=f"Network error contacting OpenRouter: {type(e).__name__}",
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
        if not self.is_configured():
            return AIResponse(
                provider=self.name,
                model=self.config.openrouter_model,
                success=False,
                error="OpenRouter API key is not configured.",
            )

        if cancel_check and cancel_check():
            return AIResponse(provider=self.name, model=self.config.openrouter_model, success=False, error="Cancelled by user.")

        key = self.config.openrouter_api_key
        url = f"{self.BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://tarang.rf",
            "X-Title": "Tarang Signal Analyzer",
            "User-Agent": "Tarang-Signal-Analyzer/1.0",
        }

        b64_img = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{b64_img}"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if context:
            messages.append({"role": "user", "content": f"RF Analysis Telemetry:\n```json\n{context}\n```"})

        user_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        messages.append({"role": "user", "content": user_content})

        model = self.config.openrouter_model
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 4096,
        }

        try:
            logger.info(f"Sending multimodal request to OpenRouter (model={model})")
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=(self.config.CONNECT_TIMEOUT, self.config.REQUEST_TIMEOUT + 30),
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
                        is_multimodal=True,
                        tokens_used=tokens,
                    )

            # If current model does not support images (e.g. text-only free model returned an error about image_url)
            err_text = resp.text[:200]
            if "image" in err_text.lower() or "vision" in err_text.lower() or resp.status_code == 400:
                logger.info("OpenRouter model does not support image input; falling back to structured analytical representation")
                return self.generate(
                    prompt=f"[RF Plot was provided ({len(image_bytes)} bytes PNG). Answer based on the analytical telemetry.]\n{prompt}",
                    context=context,
                    system_prompt=system_prompt,
                    on_token=on_token,
                    cancel_check=cancel_check,
                )

            return AIResponse(
                provider=self.name,
                model=model,
                success=False,
                error=f"OpenRouter multimodal error (HTTP {resp.status_code}): {err_text}",
            )

        except Exception as e:
            return AIResponse(
                provider=self.name,
                model=model,
                success=False,
                error=f"OpenRouter multimodal request failed: {type(e).__name__}",
            )
