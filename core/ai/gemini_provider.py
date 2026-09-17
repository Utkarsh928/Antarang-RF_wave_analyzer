"""
Google Gemini API Provider for Tarang.
Uses Google's REST endpoint with support for gemini-3.8-flash, thinking budgets, and vision.
"""
import base64
import json
from typing import Optional, Tuple, Dict, Any, Callable
import requests

from core.ai.provider_base import AIProvider, AIResponse
from core.ai.config import AIConfig


class GeminiProvider(AIProvider):
    """Google Gemini AI Provider."""

    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config or AIConfig()
        super().__init__(
            name=AIConfig.PROVIDER_GEMINI,
            default_model=self.config.gemini_model,
            is_multimodal=True,
        )

    def is_configured(self) -> bool:
        return bool(self.config.gemini_api_key)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.config.gemini_model,
            "configured": self.is_configured(),
            "multimodal": True,
            "thinking_level": self.config.thinking_level,
            "thinking_budget": self.config.thinking_budget,
        }

    def health_check(self) -> Tuple[bool, str]:
        if not self.is_configured():
            return False, "Not configured (GEMINI_API_KEY missing)"
        key = self.config.gemini_api_key
        model = self.config.gemini_model
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}?key={key}"
        try:
            resp = requests.get(url, timeout=self.config.CONNECT_TIMEOUT)
            if resp.status_code == 200:
                return True, f"Connected ({model} available)"
            elif resp.status_code in (401, 403):
                return False, "Authentication failed (Invalid API key)"
            elif resp.status_code == 429:
                return False, "Rate limited / Quota exceeded"
            elif resp.status_code == 404:
                return False, f"Model '{model}' not found on endpoint"
            else:
                return False, f"HTTP Error {resp.status_code}: {resp.text[:100]}"
        except requests.exceptions.Timeout:
            return False, "Connection timed out"
        except Exception as e:
            return False, f"Network error: {str(e)[:100]}"

    def _call_api(
        self,
        contents_parts: list,
        system_prompt: Optional[str] = None,
        on_token: Optional[Callable[[str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> AIResponse:
        if not self.is_configured():
            return AIResponse(
                success=False,
                provider=self.name,
                model=self.config.gemini_model,
                error="GEMINI_API_KEY is not configured.",
            )

        if cancel_check and cancel_check():
            return AIResponse(success=False, provider=self.name, error="Cancelled by user")

        key = self.config.gemini_api_key
        model = self.config.gemini_model
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": contents_parts}],
            "generationConfig": {
                "temperature": 0.3,
            },
        }

        # Apply thinking budget if supported
        budget = self.config.thinking_budget
        if budget > 0:
            payload["generationConfig"]["thinkingConfig"] = {
                "thinkingBudget": budget
            }

        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }

        headers = {"Content-Type": "application/json"}

        try:
            resp = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=(self.config.CONNECT_TIMEOUT, self.config.REQUEST_TIMEOUT),
            )

            # If model doesn't support thinkingConfig, retry without it
            if resp.status_code == 400 and "thinkingConfig" in payload["generationConfig"]:
                payload["generationConfig"].pop("thinkingConfig", None)
                resp = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=(self.config.CONNECT_TIMEOUT, self.config.REQUEST_TIMEOUT),
                )

            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    return AIResponse(
                        success=False,
                        provider=self.name,
                        model=model,
                        error="Empty response received from Gemini API",
                    )
                part = candidates[0].get("content", {}).get("parts", [{}])[0]
                text = part.get("text", "").strip()

                # Stream token to callback if provided
                if on_token and text:
                    on_token(text)

                usage = data.get("usageMetadata", {})
                tokens = usage.get("totalTokenCount")

                return AIResponse(
                    text=text,
                    provider=self.name,
                    model=model,
                    success=True,
                    tokens_used=tokens,
                )
            elif resp.status_code in (401, 403):
                return AIResponse(
                    success=False,
                    provider=self.name,
                    model=model,
                    error="Authentication failed: Invalid GEMINI_API_KEY",
                )
            elif resp.status_code == 429:
                return AIResponse(
                    success=False,
                    provider=self.name,
                    model=model,
                    error="Gemini API rate limit or quota exceeded",
                )
            elif resp.status_code == 404:
                return AIResponse(
                    success=False,
                    provider=self.name,
                    model=model,
                    error=f"Model '{model}' is not available on Gemini API endpoint",
                )
            else:
                err_text = resp.text[:200]
                return AIResponse(
                    success=False,
                    provider=self.name,
                    model=model,
                    error=f"Gemini API returned error {resp.status_code}: {err_text}",
                )

        except requests.exceptions.Timeout:
            return AIResponse(
                success=False,
                provider=self.name,
                model=model,
                error="Gemini API request timed out",
            )
        except requests.exceptions.ConnectionError:
            return AIResponse(
                success=False,
                provider=self.name,
                model=model,
                error="Network connection to Gemini API failed",
            )
        except Exception as e:
            return AIResponse(
                success=False,
                provider=self.name,
                model=model,
                error=f"Unexpected error communicating with Gemini: {str(e)[:150]}",
            )

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        on_token: Optional[Callable[[str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> AIResponse:
        full_text = f"{context}\n\nUser Question:\n{prompt}" if context else prompt
        parts = [{"text": full_text}]
        return self._call_api(parts, system_prompt, on_token, cancel_check)

    def generate_with_image(
        self,
        prompt: str,
        image_bytes: bytes,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        on_token: Optional[Callable[[str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> AIResponse:
        full_text = f"{context}\n\nVisual Query:\n{prompt}" if context else prompt
        b64_img = base64.b64encode(image_bytes).decode("utf-8")
        parts = [
            {"text": full_text},
            {
                "inlineData": {
                    "mimeType": "image/png",
                    "data": b64_img,
                }
            },
        ]
        resp = self._call_api(parts, system_prompt, on_token, cancel_check)
        resp.is_multimodal = True
        return resp
