"""
Offline Qwen Provider for Tarang Signal Analyzer.
Powered by Qwen2.5-1.5B-Instruct GGUF via llama-cpp-python.
General offline technical text assistant for RF, DSP, modulation, and protocol concepts.
Runs 100% locally and privately without internet connectivity once downloaded.
"""
import logging
from typing import Optional, Tuple, Dict, Any, Callable

from core.ai.provider_base import AIProvider, AIResponse
from core.ai.config import AIConfig
from core.ai.model_manager import AIModelManager, ModelStatus

logger = logging.getLogger("Tarang.AI.OfflineQwen")

OFFLINE_SYSTEM_PROMPT = (
    "You are the Tarang Offline Signal Intelligence Assistant powered by Qwen2.5-1.5B.\n"
    "You answer technical RF, DSP, modulation, communications, and protocol questions clearly and accurately.\n"
    "You must never invent or fabricate RF measurements.\n"
    "If signal telemetry is provided in the prompt, refer to those exact parameters to answer questions.\n"
    "If no signal is loaded, provide helpful general explanations of signal concepts."
)


class OfflineQwenProvider(AIProvider):
    """
    Offline Text Assistant Provider (Qwen2.5-1.5B-Instruct GGUF).
    Guarantees 100% private, on-device local inference.
    """

    def __init__(self, config: Optional[AIConfig] = None, model_manager: Optional[AIModelManager] = None):
        self.config = config or AIConfig()
        self.model_manager = model_manager or AIModelManager(self.config)
        super().__init__(
            name=AIConfig.PROVIDER_OFFLINE,
            default_model=AIConfig.DEFAULT_OFFLINE_MODEL_NAME,
            is_multimodal=False,  # Text assistant; does not pretend to visually inspect plots
        )

    def is_configured(self) -> bool:
        """Returns True if model GGUF file is installed locally."""
        return self.model_manager.is_installed()

    def get_model_info(self) -> Dict[str, Any]:
        installed = self.is_configured()
        disk_mb = self.model_manager.get_disk_usage_mb() if installed else 0.0
        return {
            "provider": self.name,
            "model": AIConfig.DEFAULT_OFFLINE_MODEL_NAME,
            "repo_id": AIModelManager.PRIMARY_REPO,
            "configured": installed,
            "multimodal": False,
            "is_offline": True,
            "disk_usage_mb": disk_mb,
            "status": self.model_manager.status,
            "status_message": self.model_manager.status_message,
        }

    def health_check(self) -> Tuple[bool, str]:
        """Verifies local model integrity."""
        if not self.is_configured():
            return False, "Offline AI (Qwen2.5-1.5B) is not installed."

        model_path = self.model_manager.get_installed_model_path()
        if model_path and model_path.exists():
            return True, f"Offline AI installed and ready ({model_path.name})"
        return False, "Model file missing or incomplete."

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
                model=self.default_model,
                success=False,
                error=(
                    "Offline AI is not installed.\n"
                    "Click 'Offline AI' in the panel to download and install it for local analysis."
                ),
            )

        if cancel_check and cancel_check():
            return AIResponse(provider=self.name, model=self.default_model, success=False, error="Cancelled by user.")

        # Lazily load GGUF model into memory
        llm, err = self.model_manager.load_model()
        if llm is None:
            return AIResponse(
                provider=self.name,
                model=self.default_model,
                success=False,
                error=f"Could not load local model: {err}",
            )

        try:
            sys_prompt = system_prompt or OFFLINE_SYSTEM_PROMPT
            user_msg = prompt
            if context:
                user_msg = f"RF Telemetry Context:\n{context}\n\nQuestion: {prompt}"

            # Qwen2.5 ChatML format
            formatted_prompt = (
                f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            logger.info("Running local inference on Qwen2.5-1.5B GGUF...")
            response_chunks = []

            # Stream tokens
            for chunk in llm.create_completion(
                prompt=formatted_prompt,
                max_tokens=1024,
                temperature=0.3,
                top_p=0.9,
                stop=["<|im_end|>", "<|endoftext|>"],
                stream=True,
            ):
                if cancel_check and cancel_check():
                    return AIResponse(
                        provider=self.name,
                        model=self.default_model,
                        success=False,
                        error="Cancelled by user.",
                    )

                text_chunk = chunk.get("choices", [{}])[0].get("text", "")
                if text_chunk:
                    response_chunks.append(text_chunk)
                    if on_token:
                        on_token(text_chunk)

            full_text = "".join(response_chunks).strip()

            return AIResponse(
                text=full_text,
                provider=self.name,
                model=self.default_model,
                success=True,
                is_multimodal=False,
                tokens_used=len(response_chunks),
            )

        except Exception as e:
            logger.exception("Local Qwen inference error")
            return AIResponse(
                provider=self.name,
                model=self.default_model,
                success=False,
                error=f"Local inference error: {str(e)}",
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
        Qwen2.5-1.5B is a text model. It answers signal questions based on
        structured analytical RF telemetry without pretending to inspect images visually.
        """
        text_prompt = (
            f"[Note: Offline AI is an on-device text assistant. Visual plot inspection is available in Online AI.]\n\n"
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


# Backward compatibility alias
LocalLlavaProvider = OfflineQwenProvider
