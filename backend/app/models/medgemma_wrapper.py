"""MedGemma wrapper with dual backend support: HuggingFace or Ollama."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

import aiohttp
import requests
import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Available backends for MedGemma inference."""

    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


@dataclass
class MedGemmaConfig:
    """Configuration for MedGemma wrapper."""

    backend: BackendType = BackendType.OLLAMA

    # HuggingFace settings
    model_id: str = "google/medgemma-4b-it"
    device: Optional[str] = None
    enable_quantization: bool = True
    max_memory: Optional[int] = None

    # Ollama settings
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "hf.co/unsloth/medgemma-4b-it-GGUF:Q4_K_M"

    # Common settings
    max_new_tokens: int = 512
    temperature: float = 0.3


class MedGemmaWrapper:
    """Unified wrapper for MedGemma supporting both HuggingFace and Ollama backends."""

    # Base system prompts for different modes
    SYSTEM_PROMPTS = {
        "default": """You are ClinSync AI, a compassionate and knowledgeable medical assistant.

You provide helpful health guidance while being warm, empathetic, and professional.
Always prioritize patient safety and recommend professional medical care when appropriate.

Response Guidelines:
- Be conversational and supportive
- Provide practical, actionable advice when safe to do so
- Always clarify when symptoms warrant in-person medical evaluation
- Never diagnose definitively - suggest possibilities and recommend confirmation by a healthcare provider
""",
        "structured": """You are a medical AI assistant providing structured analysis.

Return ONLY valid JSON. Do not include markdown, explanations, or extra text.

The response MUST follow this schema exactly:
{
  "findings": string[],
  "recommendations": string[],
  "confidence": number (0.0 to 1.0),
  "severity": "low" | "moderate" | "urgent",
  "specialist_recommended": string | null
}
""",
        "radiology": """You are a radiology assistant specialized in medical imaging analysis.

Return ONLY valid JSON following this schema:
{
  "findings": string[],
  "recommendations": string[],
  "confidence": number (0.0 to 1.0),
  "severity": "low" | "moderate" | "urgent",
  "specialist_recommended": "radiologist" | "pulmonologist" | null
}
""",
        "skin": """You are a dermatology-focused medical AI assistant.

When analyzing skin conditions:
- Describe visual characteristics (color, texture, distribution, borders)
- Consider common differentials based on presentation
- Note any concerning features (asymmetry, irregular borders, color variation)
- Recommend appropriate next steps based on severity

Be empathetic - skin conditions can significantly impact quality of life.
""",
        "respiratory": """You are a medical AI assistant focused on respiratory health.

When evaluating respiratory concerns:
- Ask about onset, duration, and progression of symptoms
- Consider environmental factors (allergies, exposures, smoking)
- Note any warning signs (difficulty breathing, chest pain, high fever)
- Provide guidance on when to seek emergency care vs. routine care
""",
        "cardiology": """You are a medical AI assistant focused on cardiovascular health.

When evaluating cardiac concerns:
- Take chest pain and palpitations seriously
- Note risk factors (age, family history, lifestyle)
- Recognize warning signs requiring immediate attention
- Emphasize importance of professional evaluation for cardiac symptoms
""",
        "voice": """You are ClinSync AI, a helpful medical assistant in a real-time video consultation.

IMPORTANT Guidelines:
1. Speak naturally as if in a face-to-face conversation
2. Be warm, empathetic, and reassuring
3. Keep responses concise (3-5 sentences) - they will be spoken aloud
4. Give practical, actionable advice when appropriate
5. For common issues (headaches, colds, minor pain), you CAN suggest over-the-counter remedies
6. Always mention when to see a doctor in person
7. DO NOT use bullet points, numbered lists, or markdown formatting
8. Address the patient by name when known to personalize the interaction
""",
    }

    @classmethod
    def build_consultation_prompt(
        cls,
        patient_context: Optional[Dict[str, Any]] = None,
        consultation_type: str = "general",
        mode: str = "voice",
    ) -> str:
        """
        Build a dynamic system prompt incorporating patient context.

        Args:
            patient_context: Dict with patient info (name, age, symptoms, allergies, etc.)
            consultation_type: Type of consultation (general, skin, respiratory, cardiology)
            mode: Response mode (voice, structured, chat)

        Returns:
            Complete system prompt with patient context
        """
        # Start with base prompt for mode
        if mode == "voice":
            base_prompt = cls.SYSTEM_PROMPTS["voice"]
        elif mode == "structured":
            base_prompt = cls.SYSTEM_PROMPTS["structured"]
        else:
            base_prompt = cls.SYSTEM_PROMPTS["default"]

        # Add consultation-type specific knowledge
        consultation_prompts = {
            "skin": cls.SYSTEM_PROMPTS.get("skin", ""),
            "respiratory": cls.SYSTEM_PROMPTS.get("respiratory", ""),
            "cardiology": cls.SYSTEM_PROMPTS.get("cardiology", ""),
        }

        type_specific = consultation_prompts.get(consultation_type, "")

        # Build patient context section
        patient_section = ""
        if patient_context:
            patient_section = "\n\n=== PATIENT CONTEXT ===\n"

            if patient_context.get("fullName"):
                patient_section += f"Patient: {patient_context['fullName']}\n"
            if patient_context.get("age"):
                patient_section += f"Age: {patient_context['age']} years\n"
            if patient_context.get("gender"):
                patient_section += f"Gender: {patient_context['gender']}\n"

            if patient_context.get("chiefComplaint"):
                patient_section += (
                    f"\nChief Complaint: {patient_context['chiefComplaint']}\n"
                )

            symptoms = patient_context.get("symptoms", [])
            if symptoms:
                patient_section += f"Reported Symptoms: {', '.join(symptoms)}\n"

            if patient_context.get("symptomDuration"):
                patient_section += f"Duration: {patient_context['symptomDuration']}\n"

            pain_level = patient_context.get("painLevel")
            if pain_level:
                patient_section += f"Pain Level: {pain_level}/10\n"

            # Critical safety information
            if patient_context.get("allergies"):
                patient_section += f"\n⚠️ ALLERGIES: {patient_context['allergies']}\n"

            if patient_context.get("currentMedications"):
                patient_section += (
                    f"Current Medications: {patient_context['currentMedications']}\n"
                )

            if patient_context.get("medicalHistory"):
                patient_section += (
                    f"Medical History: {patient_context['medicalHistory']}\n"
                )

            patient_section += "=== END CONTEXT ===\n"

            # Add safety reminder
            patient_section += """
IMPORTANT: Use this patient information to:
- Personalize your responses (use their name when appropriate)
- Consider their allergies when suggesting treatments
- Account for existing medications and potential interactions
- Factor in their medical history when assessing symptoms
"""

        # Combine all parts
        full_prompt = base_prompt
        if type_specific:
            full_prompt += f"\n\n{type_specific}"
        if patient_section:
            full_prompt += patient_section

        return full_prompt

    def __init__(
        self,
        config: Optional[MedGemmaConfig] = None,
        model_id: str = "google/medgemma-4b-it",
        device: Optional[str] = None,
        enable_quantization: bool = True,
        max_memory: Optional[int] = None,
    ) -> None:
        if config is None:
            config = MedGemmaConfig(
                backend=BackendType.OLLAMA,
                model_id=model_id,
                device=device,
                enable_quantization=enable_quantization,
                max_memory=max_memory,
            )

        self.config = config
        self.backend = config.backend

        self._model: Optional[AutoModelForImageTextToText] = None
        self._processor: Optional[AutoProcessor] = None
        self._ollama_available: Optional[bool] = None

        self._executor = ThreadPoolExecutor(max_workers=1)

    @classmethod
    def from_config(cls, config: MedGemmaConfig) -> "MedGemmaWrapper":
        """Create wrapper from configuration object."""
        return cls(config=config)

    @classmethod
    def from_ollama(
        cls,
        host: str = "http://localhost:11434",
        model: str = "hf.co/unsloth/medgemma-4b-it-GGUF:Q4_K_M",
    ) -> "MedGemmaWrapper":
        """Convenience constructor for Ollama backend."""
        config = MedGemmaConfig(
            backend=BackendType.OLLAMA,
            ollama_host=host,
            ollama_model=model,
        )
        return cls(config=config)

    @property
    def is_loaded(self) -> bool:
        """Return True if backend is ready."""
        if self.backend == BackendType.HUGGINGFACE:
            return self._model is not None and self._processor is not None
        else:
            return self._ollama_available is not False

    @property
    def backend_name(self) -> str:
        """Get current backend name."""
        return self.backend.value

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get 4-bit quantization config for HF backend."""
        if not self.config.enable_quantization:
            return None

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    async def _check_ollama(self) -> bool:
        """Check if Ollama is available."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config.ollama_host}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = [m["name"] for m in data.get("models", [])]
                        if self.config.ollama_model in models:
                            logger.info(
                                "✅ Ollama %s available", self.config.ollama_model
                            )
                            return True
                        else:
                            logger.warning(
                                "⚠️ Ollama model %s not found. Available: %s",
                                self.config.ollama_model,
                                models,
                            )
                            return False
                    return False
        except Exception as e:
            logger.error("❌ Ollama not available: %s", e)
            return False

    def load(self) -> None:
        """Load backend (HF model or verify Ollama)."""
        if self.backend == BackendType.HUGGINGFACE:
            self._load_huggingface()

    def _load_huggingface(self) -> None:
        """Load HuggingFace model."""
        if self.is_loaded:
            return

        logger.info("Loading MedGemma from HuggingFace: %s", self.config.model_id)

        self._processor = AutoProcessor.from_pretrained(self.config.model_id)

        load_kwargs: dict[str, Any] = {
            "torch_dtype": (
                torch.float16 if self.config.enable_quantization else torch.float32
            ),
        }

        quant_config = self._get_quantization_config()
        if quant_config:
            load_kwargs["quantization_config"] = quant_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = None

        try:
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.config.model_id,
                **load_kwargs,
            )

            if quant_config is None:
                device = self.config.device or (
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                self._model = self._model.to(device)

            self._model.eval()
            logger.info("✅ HuggingFace MedGemma loaded")

        except torch.cuda.OutOfMemoryError as e:
            logger.error("CUDA OOM: %s", e)
            raise RuntimeError(f"GPU memory insufficient. Error: {e}")
        except Exception as e:
            logger.error("Failed to load MedGemma: %s", e)
            raise

    def unload(self) -> None:
        """Unload backend resources."""
        if self.backend == BackendType.HUGGINGFACE:
            if self._model:
                del self._model
                self._model = None
            if self._processor:
                del self._processor
                self._processor = None
            torch.cuda.empty_cache()
            logger.info("HuggingFace MedGemma unloaded")

    def _prepare_image(self, image: Union[Image.Image, str, bytes]) -> Image.Image:
        """Prepare image for processing."""
        if isinstance(image, str):
            if image.startswith("http"):
                response = requests.get(image, timeout=10)
                image = Image.open(io.BytesIO(response.content))
            elif image.startswith("data:image"):
                # Handle data URL
                image_data = base64.b64decode(image.split(",")[1])
                image = Image.open(io.BytesIO(image_data))
            else:
                image = Image.open(image)
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize if too large
        max_size = 1024
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.LANCZOS)

        return image

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL image to base64 string for Ollama."""
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode()

    def generate(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, str, bytes]] = None,
        system_prompt: Optional[str] = None,
        specialization: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate response (synchronous - HF only)."""
        if self.backend == BackendType.OLLAMA:
            raise RuntimeError(
                "Ollama backend only supports async generation. "
                "Use generate_async() instead."
            )

        if not self.is_loaded:
            self.load()

        return self._generate_huggingface(
            prompt, image, system_prompt, specialization, **kwargs
        )

    def _generate_huggingface(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, str, bytes]] = None,
        system_prompt: Optional[str] = None,
        specialization: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate using HuggingFace backend."""
        try:
            pil_image = None
            if image is not None:
                pil_image = self._prepare_image(image)

            if system_prompt is None and specialization:
                system_prompt = self.SYSTEM_PROMPTS.get(
                    specialization, self.SYSTEM_PROMPTS["default"]
                )
            elif system_prompt is None:
                system_prompt = self.SYSTEM_PROMPTS["default"]

            messages = self._build_messages(prompt, pil_image, system_prompt)

            prompt_text = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )

            processor_kwargs = {
                "text": prompt_text,
                "return_tensors": "pt",
            }
            if pil_image is not None:
                processor_kwargs["images"] = pil_image

            inputs = self._processor(**processor_kwargs)
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            generation_config = {
                "max_new_tokens": max_new_tokens or self.config.max_new_tokens,
                "temperature": temperature or self.config.temperature,
                "do_sample": (temperature or self.config.temperature) > 0,
                "pad_token_id": self._processor.tokenizer.pad_token_id,
            }

            with torch.inference_mode():
                outputs = self._model.generate(**inputs, **generation_config)

            input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
            new_tokens = outputs[0][input_len:] if input_len else outputs[0]
            raw_response = self._processor.decode(
                new_tokens, skip_special_tokens=True
            ).strip()

            if not raw_response:
                full_decoded = self._processor.decode(
                    outputs[0], skip_special_tokens=True
                ).strip()
                if full_decoded and prompt_text in full_decoded:
                    raw_response = full_decoded.split(prompt_text, 1)[-1].strip()

            parsed = self._parse_medical_response(raw_response)
            parsed.update(
                {
                    "model": self.config.model_id,
                    "input_type": "multimodal" if pil_image else "text",
                    "image_size": pil_image.size if pil_image else None,
                    "specialization": specialization or "default",
                    "backend": "huggingface",
                }
            )

            return parsed

        except Exception as e:
            logger.error("HF generation failed: %s", e)
            return self._error_response(str(e))

    async def _generate_ollama(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, str, bytes]] = None,
        system_prompt: Optional[str] = None,
        specialization: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate using Ollama backend (VISION) via /api/chat with message.images."""
        # Check Ollama availability on first call
        if self._ollama_available is None:
            self._ollama_available = await self._check_ollama()

        if not self._ollama_available:
            return self._error_response(
                f"Ollama not available or model {self.config.ollama_model} not found. "
                "Run: ollama pull " + self.config.ollama_model
            )

        try:
            # Prepare image (base64) if provided
            pil_image = None
            image_b64 = None

            if image is not None:
                logger.info("Preparing image for Ollama...")
                pil_image = self._prepare_image(image)
                image_b64 = self._image_to_base64(pil_image)
                logger.info("Image converted to base64, length=%d", len(image_b64))

            # Pick system prompt
            if system_prompt is None and specialization:
                system_prompt = self.SYSTEM_PROMPTS.get(
                    specialization, self.SYSTEM_PROMPTS["default"]
                )
            elif system_prompt is None:
                system_prompt = self.SYSTEM_PROMPTS["default"]

            # ✅ Use /api/chat for vision and put images in the USER message
            user_msg: dict[str, Any] = {
                "role": "user",
                "content": prompt,
            }
            if image_b64:
                user_msg["images"] = [image_b64]

            payload: dict[str, Any] = {
                "model": self.config.ollama_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    user_msg,
                ],
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "num_predict": kwargs.get(
                        "max_new_tokens", self.config.max_new_tokens
                    ),
                },
            }

            payload["messages"][0][
                "content"
            ] += "\n\nIMPORTANT: Respond ONLY with valid JSON."

            logger.debug(
                "Ollama payload: %s",
                json.dumps(
                    {
                        **payload,
                        "messages": [
                            payload["messages"][0],
                            {**user_msg, "images": ["..."] if image_b64 else []},
                        ],
                    }
                ),
            )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.ollama_host}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=180),
                ) as resp:
                    response_text = await resp.text()

                    logger.debug("Ollama response: %s", response_text)

                    if resp.status != 200:
                        logger.error("Ollama error response: %s", response_text)
                        raise RuntimeError(
                            f"Ollama error {resp.status}: {response_text[:500]}"
                        )

                    try:
                        data = json.loads(response_text)
                    except json.JSONDecodeError:
                        logger.error(
                            "Invalid JSON from Ollama: %s", response_text[:500]
                        )
                        raise RuntimeError("Invalid response from Ollama")

                    # ✅ /api/chat returns message.content (not response)
                    raw_response = (data.get("message") or {}).get("content", "")

                    if not raw_response:
                        logger.warning(
                            "Ollama returned empty content. Full data: %s", data
                        )
                        return self._error_response("Ollama returned empty response")

                    logger.info(
                        "Ollama response received, length: %d", len(raw_response)
                    )

                    parsed = self._parse_medical_response(raw_response)
                    parsed.update(
                        {
                            "model": self.config.ollama_model,
                            "input_type": "multimodal" if image_b64 else "text",
                            "image_size": pil_image.size if pil_image else None,
                            "specialization": specialization or "default",
                            "backend": "ollama",
                            "ollama_total_duration": data.get("total_duration"),
                            "ollama_eval_count": data.get("eval_count"),
                            "ollama_eval_duration": data.get("eval_duration"),
                        }
                    )
                    logger.info("Parsed response: %s", parsed)
                    return parsed

        except Exception as e:
            logger.error("Ollama generation failed: %s", e, exc_info=True)
            return self._error_response(str(e))

    async def generate_async(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, str, bytes]] = None,
        system_prompt: Optional[str] = None,
        specialization: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate response (async - works with both backends)."""
        logger.info("Generating response using %s backend", self.backend.value)

        if self.backend == BackendType.HUGGINGFACE:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                self._generate_huggingface,
                prompt,
                image,
                system_prompt,
                specialization,
                **kwargs,
            )
        else:
            return await self._generate_ollama(
                prompt, image, system_prompt, specialization, **kwargs
            )

    def _build_messages(
        self,
        prompt: str,
        image: Optional[Image.Image],
        system_prompt: str,
    ) -> list[dict[str, Any]]:
        """Build chat messages for HF format."""
        messages = []

        if system_prompt:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )

        user_content = []

        if image is not None:
            user_content.append({"type": "image", "image": image})

        user_content.append({"type": "text", "text": prompt})

        messages.append({"role": "user", "content": user_content})

        return messages

    def _parse_medical_response(self, raw_response: str) -> dict[str, Any]:
        cleaned = raw_response.strip()

        prefixes = ["assistant", "medical assistant", "radiology assistant"]
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip().lstrip(":").strip()

        cleaned = self._strip_json_fences(cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error("Model returned non-JSON output: %s", cleaned[:500])
            return self._fallback_parse(raw_response)

        return {
            "raw_response": raw_response,
            "clean_response": cleaned,
            "findings": data.get("findings", []) or ["See detailed response"],
            "recommendations": data.get("recommendations", [])
            or ["Consult healthcare provider"],
            "confidence": float(data.get("confidence", 0.7)),
            "severity": data.get("severity", "low"),
            "specialist_recommended": data.get("specialist_recommended"),
        }

    def _strip_json_fences(self, text: str) -> str:
        """
        Remove Markdown ```json fences if present.
        """
        text = text.strip()

        if text.startswith("```"):
            # Remove opening fence (``` or ```json)
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            # Remove closing fence
            text = re.sub(r"\s*```$", "", text)

        return text.strip()

    def _fallback_parse(self, raw_response: str) -> dict[str, Any]:
        logger.warning("Using fallback medical parser")

        return {
            "raw_response": raw_response,
            "clean_response": raw_response.strip(),
            "findings": ["See detailed response"],
            "recommendations": ["Consult healthcare provider"],
            "confidence": 0.7,
            "severity": "low",
            "specialist_recommended": None,
        }

    def _error_response(self, error_msg: str) -> dict[str, Any]:
        return {
            "error": error_msg,
            "raw_response": "",
            "clean_response": f"Analysis error: {error_msg}",
            "findings": ["Error in analysis"],
            "confidence": 0.0,
            "recommendations": ["Please retry or consult specialist directly"],
            "severity": "unknown",
            "specialist_recommended": None,
            "model": (
                self.config.model_id
                if self.backend == BackendType.HUGGINGFACE
                else self.config.ollama_model
            ),
            "input_type": "unknown",
            "backend": self.backend.value,
        }

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
        return False
