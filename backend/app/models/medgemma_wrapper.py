"""MedGemma wrapper with 4-bit quantization and multimodal support."""
from __future__ import annotations

import logging
import threading
from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

from app.core.config import Settings

logger = logging.getLogger(__name__)


class MedGemmaWrapper:
    """Lazy-loading wrapper around MedGemma for multimodal inference."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Optional[AutoModelForCausalLM] = None
        self._processor: Optional[AutoProcessor] = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """Return True if model is loaded in memory."""
        return self._model is not None and self._processor is not None

    def _ensure_loaded(self) -> None:
        if self.is_loaded:
            return
        with self._lock:
            if self.is_loaded:
                return
            logger.info("Loading MedGemma model: %s", self._settings.medgemma_model_id)
            quant_config = None
            if self._settings.enable_quantization:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            self._processor = AutoProcessor.from_pretrained(self._settings.medgemma_model_id)
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self._settings.medgemma_model_id,
                    device_map="auto" if torch.cuda.is_available() else None,
                    quantization_config=quant_config,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Quantized load failed, retrying in full precision: %s", exc)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self._settings.medgemma_model_id,
                    device_map="auto" if torch.cuda.is_available() else None,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                )
            self._model.eval()

    def _format_prompt(self, prompt: str, has_image: bool) -> str:
        """Format prompt with image token if needed."""
        if has_image:
            # MedGemma requires <image> token in the prompt to reference the image
            if "<image>" not in prompt.lower():
                # Prepend image token for vision-language understanding
                return f"<image>\n{prompt}"
        return prompt

    def generate(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        max_new_tokens: int = 512,
    ) -> str:
        """Generate a response from MedGemma with optional image input."""
        self._ensure_loaded()
        if self._model is None or self._processor is None:
            raise RuntimeError("MedGemma model failed to load.")

        # Format prompt with image token if image is provided
        formatted_prompt = self._format_prompt(prompt, has_image=image is not None)

        # Process inputs - pass image as list if provided
        if image is not None:
            inputs = self._processor(
                text=formatted_prompt,
                images=[image],
                return_tensors="pt",
            )
        else:
            inputs = self._processor(
                text=formatted_prompt,
                return_tensors="pt",
            )

        device = next(self._model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Decode and remove the input prompt from output
        output = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        # Remove the original prompt from the output if present
        if formatted_prompt in output:
            output = output.replace(formatted_prompt, "").strip()

        return output.strip()
