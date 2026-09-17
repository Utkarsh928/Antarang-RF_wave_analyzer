"""
Groq API Provider for Tarang.
Uses the OpenAI-compatible Groq REST API endpoint.
Fast fallback provider with zero-cost / high-throughput inference.
"""
import json
import logging
from typing import Optional, Tuple, Dict, Any, Callable
import requests

from core.ai.provider_base import AIProvider, AIResponse
from core.ai.config import AIConfig

logger = logging.getLogger("Tarang.AI.Groq")


class GroqProvider(AIProvider):
    """
    Groq OpenAI-compatible provider.
    Endpoint: https://api.groq.com/openai/v1/chat/completions
    """

    BASE_URL = "https://api.groq.com/openai/v1"

    # Common fallback models if default is unavailable
    FALLBACK_MODELS = [
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ]

    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config or AIConfig()
        super().__init__(
            name=AIConfig.PROVIDER_GROQ,
            default_model=self.config.groq_model,
            is_multimodal=False,  # Groq text models; handles images via structured analytical representation
        )

    def is_configured(self) -> bool:
        key = self.config.groq_api_key
        return bool(key and len(key.strip()) > 5)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.config.groq_model,
            "configured": self.is_configured(),
            "multimodal": self.is_multimodal,
            "notes": "Ultra-fast inference engine.",
        }

    def health_check(self) -> Tuple[bool, str]:
        """Check API key validity by querying the /models endpoint."""
        if not self.is_configured():
            return False, "Groq API key not configured (GROQ_API_KEY)."

        key = self.config.groq_api_key
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
                return True, f"Connected to Groq (Model: {self.config.groq_model})"
            elif resp.status_code == 401:
                return False, "Authentication failed: Invalid GROQ_API_KEY."
            elif resp.status_code == 429:
                return False, "Rate limited / Quota exceeded on Groq."
            else:
                return False, f"Groq returned HTTP {resp.status_code}: {resp.text[:120]}"
        except requests.exceptions.Timeout:
            return False, "Connection timeout to Groq API."
        except requests.exceptions.RequestException as e:
            return False, f"Network error connecting to Groq: {type(e).__name__}"

    def _query_available_models(self) -> list:
        """Fetch available model IDs from Groq."""
        key = self.config.groq_api_key
        if not key:
            return []
        try:
            resp = requests.get(
                f"{self.BASE_URL}/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                return [m.get("id") for m in data.get("data", [])]
        except Exception:
            pass
        return []

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
                model=self.config.groq_model,
                success=False,
                error="Groq API key is not configured. Set GROQ_API_KEY in .env or Settings.",
            )

        if cancel_check and cancel_check():
            return AIResponse(provider=self.name, model=self.config.groq_model, success=False, error="Cancelled by user.")

        key = self.config.groq_api_key
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
            messages.append({"role": "user", "content": f"Here is the RF signal analysis telemetry:\n```json\n{context}\n```"})
        messages.append({"role": "user", "content": prompt})

        # Try user-configured model first, then fallback models if model_not_found
        models_to_try = [self.config.groq_model]
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
                logger.info(f"Sending request to Groq (model={model})")
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
                        error="Empty response choices from Groq.",
                    )

                error_data = {}
                try:
                    error_data = resp.json().get("error", {})
                except Exception:
                    pass
                err_msg = error_data.get("message", resp.text[:200])

                if resp.status_code == 404 or "model_not_found" in str(err_msg).lower() or "does not exist" in str(err_msg).lower():
                    logger.warning(f"Groq model {model} not found or decommissioned, trying next fallback...")
                    last_error = f"Model {model} unavailable: {err_msg}"
                    continue

                if resp.status_code == 401:
                    return AIResponse(
                        provider=self.name,
                        model=model,
                        success=False,
                        error="Authentication failed: Invalid GROQ_API_KEY.",
                    )
                elif resp.status_code == 429:
                    return AIResponse(
                        provider=self.name,
                        model=model,
                        success=False,
                        error=f"Groq rate limit exceeded: {err_msg}",
                    )
                else:
                    return AIResponse(
                        provider=self.name,
                        model=model,
                        success=False,
                        error=f"Groq error (HTTP {resp.status_code}): {err_msg}",
                    )

            except requests.exceptions.Timeout:
                return AIResponse(
                    provider=self.name,
                    model=model,
                    success=False,
                    error="Groq request timed out.",
                )
            except requests.exceptions.RequestException as e:
                return AIResponse(
                    provider=self.name,
                    model=model,
                    success=False,
                    error=f"Network error with Groq: {type(e).__name__}",
                )

        return AIResponse(
            provider=self.name,
            model=self.config.groq_model,
            success=False,
            error=f"All attempted Groq models failed. Last error: {last_error}",
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
        Groq text models do not directly accept raw plot image bytes.
        Per Rule 10: 'For text-only providers: send structured numerical/analytical information where possible.
        Do not pretend a text-only model visually inspected an image.'
        """
        text_prompt = (
            f"[Note: Live plot image was captured from the RF visualization display ({len(image_bytes)} bytes PNG). "
            f"As a text-focused LLM, analyze the signal's analytical RF telemetry and parameters provided below to answer the query.]\n\n"
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
