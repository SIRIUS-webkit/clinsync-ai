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

        inputs = self._processor(text=prompt, images=image, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        output = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return output.strip()
