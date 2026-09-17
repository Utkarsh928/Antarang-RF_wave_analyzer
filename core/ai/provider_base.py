"""
Base abstractions for Tarang AI Providers.
All online and offline AI implementations adhere to this interface.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, Callable


@dataclass
class AIResponse:
    """Standardized response from any AI provider."""
    text: str = ""
    provider: str = ""
    model: str = ""
    success: bool = True
    error: Optional[str] = None
    thinking: Optional[str] = None
    is_multimodal: bool = False
    tokens_used: Optional[int] = None
    fallback_from: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AIProvider(ABC):
    """Abstract base class for all Tarang AI providers."""

    def __init__(self, name: str, default_model: str, is_multimodal: bool = False):
        self.name = name
        self.default_model = default_model
        self.is_multimodal = is_multimodal

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if required credentials or local assets are present."""
        pass

    @abstractmethod
    def health_check(self) -> Tuple[bool, str]:
        """
        Lightweight health check without burning excessive quota.
        Returns (is_healthy, status_message).
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return provider metadata, current model ID, and capabilities."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        on_token: Optional[Callable[[str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> AIResponse:
        """
        Execute text generation with optional streaming token callback.
        """
        pass

    @abstractmethod
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
        Execute multimodal generation (e.g. plot image analysis).
        """
        pass
