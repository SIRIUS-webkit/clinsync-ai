"""HuggingFace Transformers-based LLM for local inference."""
from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from app.core.config import Settings

logger = logging.getLogger(__name__)


class HuggingFaceLLM:
    """Lazy-loading HuggingFace LLM wrapper with quantization support."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """Return True if model is loaded."""
        return self._model is not None and self._tokenizer is not None

    def _ensure_loaded(self) -> None:
        if self.is_loaded:
            return
        with self._lock:
            if self.is_loaded:
                return
            logger.info("Loading HuggingFace LLM: %s", self._settings.hf_llm_model_id)

            quant_config = None
            if self._settings.enable_quantization and torch.cuda.is_available():
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )

            self._tokenizer = AutoTokenizer.from_pretrained(self._settings.hf_llm_model_id)

            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self._settings.hf_llm_model_id,
                    device_map="auto" if torch.cuda.is_available() else None,
                    quantization_config=quant_config,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                )
            except Exception as exc:
                logger.warning("Quantized load failed, retrying without quantization: %s", exc)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self._settings.hf_llm_model_id,
                    device_map="auto" if torch.cuda.is_available() else None,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                )

            self._model.eval()
            logger.info("HuggingFace LLM loaded successfully")

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Generate text completion."""
        self._ensure_loaded()
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("HuggingFace LLM failed to load.")

        max_tokens = max_new_tokens or self._settings.hf_llm_max_new_tokens

        # Format as chat if tokenizer supports it
        if hasattr(self._tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        inputs = self._tokenizer(formatted, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,  # Greedy decoding for stability
                pad_token_id=self._tokenizer.eos_token_id or self._tokenizer.pad_token_id or 0,
            )

        # Decode only the new tokens
        input_len = inputs["input_ids"].shape[1]
        output = self._tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
        return output.strip()

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """Generate and parse JSON response."""
        json_prompt = (
            f"{prompt}\n\n"
            "Respond with valid JSON only, no other text."
        )
        response = self.generate(json_prompt)

        # Try to extract JSON from response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            json_match = re.search(r"\{[^{}]*\}", response)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            return {}
