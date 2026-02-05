"""MedGemma wrapper with 4-bit quantization using official HF implementation."""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, Union

import requests
import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)


class MedGemmaWrapper:
    """Lazy-loading wrapper around MedGemma for multimodal inference."""

    # Medical system prompts for different specializations
    SYSTEM_PROMPTS = {
        "default": """You are a medical AI assistant. Analyze the provided medical image carefully.
Describe findings, potential abnormalities, and suggest differential diagnoses if applicable.
Always include confidence levels and recommend when to consult a specialist.""",
        
        "radiology": """You are a radiology assistant. Examine this medical image and provide a structured report.
Include: Findings, Impression, and Recommendations. Be specific about anatomical locations.""",
        
        "dermatology": """You are a dermatology assistant. Analyze this skin image for lesions, rashes, or abnormalities.
Describe: Appearance, Distribution, Pattern, and Suggested Differential Diagnosis.""",
        
        "pathology": """You are a pathology assistant. Analyze this histopathology image.
Describe cellular structures, abnormalities, and potential diagnoses.""",
    }

    def __init__(
        self,
        model_id: str = "google/medgemma-4b-it",
        device: Optional[str] = None,
        enable_quantization: bool = True,
        max_memory: Optional[int] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_quantization = enable_quantization and torch.cuda.is_available()
        self.max_memory = max_memory
        
        self._model: Optional[AutoModelForImageTextToText] = None
        self._processor: Optional[AutoProcessor] = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = asyncio.Lock()
        
    @property
    def is_loaded(self) -> bool:
        """Return True if model is loaded in memory."""
        return self._model is not None and self._processor is not None

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get 4-bit quantization config for memory efficiency."""
        if not self.enable_quantization:
            return None
            
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    def load(self) -> None:
        """Load model and processor. Thread-safe."""
        if self.is_loaded:
            return
            
        logger.info("Loading MedGemma model: %s", self.model_id)
        
        # Load processor first
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        
        # Prepare loading arguments
        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch.float16 if self.enable_quantization else torch.float32,
        }
        
        # Add quantization if enabled
        quant_config = self._get_quantization_config()
        if quant_config:
            load_kwargs["quantization_config"] = quant_config
            load_kwargs["device_map"] = "auto"
            logger.info("Using 4-bit quantization (NF4)")
        else:
            load_kwargs["device_map"] = None
            
        try:
            # Use official HF class for image-text-to-text
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                **load_kwargs,
            )
            
            # Manual device placement if not using device_map
            if quant_config is None:
                self._model = self._model.to(self.device)
            
            self._model.eval()
            logger.info("✅ MedGemma loaded on %s", self.device)
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error("CUDA OOM: %s", e)
            raise RuntimeError(
                f"GPU memory insufficient. Try: "
                f"1) Smaller image size, "
                f"2) enable quantization, or "
                f"3) use CPU. Error: {e}"
            )
        except Exception as e:
            logger.error("Failed to load MedGemma: %s", e)
            raise

    def unload(self) -> None:
        """Unload model to free memory."""
        if self._model:
            del self._model
            self._model = None
        if self._processor:
            del self._processor
            self._processor = None
        torch.cuda.empty_cache()
        logger.info("MedGemma unloaded")

    def _prepare_image(self, image: Union[Image.Image, str, bytes]) -> Image.Image:
        """Prepare image for processing."""
        # Handle different input types
        if isinstance(image, str):
            if image.startswith("http"):
                # Load from URL
                response = requests.get(image, timeout=10)
                image = Image.open(io.BytesIO(response.content))
            elif image.startswith("data:image"):
                # Base64 encoded
                image_data = base64.b64decode(image.split(",")[1])
                image = Image.open(io.BytesIO(image_data))
            else:
                # File path
                image = Image.open(image)
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize if too large (prevent OOM)
        max_size = 1024
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.LANCZOS)
            logger.debug("Resized image to %s", image.size)
        
        return image

    def _build_chat_messages(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        system_prompt: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Build chat messages in HF format."""
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })
        
        # Build user message
        user_content = []
        
        if image is not None:
            user_content.append({
                "type": "image",
                "image": image  # PIL Image object
            })
        
        user_content.append({
            "type": "text",
            "text": prompt
        })
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        return messages

    def generate(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, str, bytes]] = None,
        system_prompt: Optional[str] = None,
        specialization: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
        do_sample: bool = False,
    ) -> dict[str, Any]:
        """
        Generate response from MedGemma using official HF chat template.
        
        Args:
            prompt: User question/instruction
            image: Optional image (PIL, URL, path, or base64)
            system_prompt: Optional system prompt (overrides specialization)
            specialization: 'radiology', 'dermatology', 'pathology', or 'default'
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling (False = greedy)
            
        Returns:
            Structured dict with findings, confidence, recommendations
        """
        if not self.is_loaded:
            self.load()
        
        try:
            # Prepare image if provided
            pil_image = None
            if image is not None:
                pil_image = self._prepare_image(image)
            
            # Select system prompt
            if system_prompt is None and specialization:
                system_prompt = self.SYSTEM_PROMPTS.get(specialization, self.SYSTEM_PROMPTS["default"])
            elif system_prompt is None:
                system_prompt = self.SYSTEM_PROMPTS["default"]
            
            # Build chat messages
            messages = self._build_chat_messages(prompt, pil_image, system_prompt)
            
            # Build prompt text with chat template
            prompt_text = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            
            # Encode text (and image if provided)
            processor_kwargs: dict[str, Any] = {
                "text": prompt_text,
                "return_tensors": "pt",
            }
            if pil_image is not None:
                processor_kwargs["images"] = pil_image
            inputs = self._processor(**processor_kwargs)
            
            # Move to device
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            
            # Generate
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature if do_sample else None,
                "top_p": top_p if do_sample else None,
                "do_sample": do_sample,
                "pad_token_id": self._processor.tokenizer.pad_token_id,
            }
            
            # Remove None values
            generation_config = {k: v for k, v in generation_config.items() if v is not None}
            
            with torch.inference_mode():
                outputs = self._model.generate(
                    **inputs,
                    **generation_config,
                )
            
            # Decode only new tokens (remove input)
            input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
            new_tokens = outputs[0][input_len:] if input_len else outputs[0]
            raw_response = self._processor.decode(new_tokens, skip_special_tokens=True).strip()
            if not raw_response:
                # Fallback to full decode if no new tokens or only special tokens
                full_decoded = self._processor.decode(outputs[0], skip_special_tokens=True).strip()
                if full_decoded:
                    if prompt_text and prompt_text in full_decoded:
                        raw_response = full_decoded.split(prompt_text, 1)[-1].strip()
                    else:
                        raw_response = full_decoded
                if not raw_response:
                    logger.warning(
                        "MedGemma produced empty response (input_len=%s, output_len=%s)",
                        input_len,
                        outputs[0].shape[-1],
                    )
            
            # Parse structured output
            parsed = self._parse_medical_response(raw_response)
            
            # Add metadata
            parsed.update({
                "model": self.model_id,
                "input_type": "multimodal" if pil_image else "text",
                "image_size": pil_image.size if pil_image else None,
                "specialization": specialization or "default",
            })
            
            return parsed
            
        except Exception as e:
            logger.error("Generation failed: %s", e)
            return self._error_response(str(e))

    async def generate_async(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, str, bytes]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Async wrapper that runs generation in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.generate,
            prompt,
            image,
            **kwargs,
        )

    def _parse_medical_response(self, raw_response: str) -> dict[str, Any]:
        """Parse raw model output into structured medical format."""
        logger.info("Raw response: %s", raw_response)
        if not raw_response or not raw_response.strip():
            logger.warning("MedGemma returned empty raw response")
        
        # Clean response
        cleaned = raw_response.strip()
        logger.info("Cleaned response: %r", cleaned)
        
        # Remove common prefixes
        prefixes_to_remove = [
            "assistant",
            "medical assistant", 
            "radiology assistant",
        ]
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                # Remove leading punctuation
                cleaned = cleaned.lstrip(":").strip()
        
        logger.info("After prefix removal: %r", cleaned)
        
        # Extract confidence
        confidence = self._extract_confidence(cleaned)
        
        # Extract findings
        findings = self._extract_findings(cleaned)
        
        # Extract recommendations
        recommendations = self._extract_recommendations(cleaned)
        
        # Determine severity
        severity = self._assess_severity(cleaned)
        
        # Check for specialist referral
        specialist = self._detect_specialist_referral(cleaned)
        
        return {
            "raw_response": raw_response,
            "clean_response": cleaned,
            "findings": findings or ["See detailed response"],
            "confidence": confidence,
            "recommendations": recommendations or ["Consult healthcare provider"],
            "severity": severity,
            "specialist_recommended": specialist,
        }

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from text."""
        patterns = [
            r"confidence[:\\s]+(\\d+(?:\\.\\d+)?)",
            r"(\\d+(?:\\.\\d+)?)\\s*%\\s*(?:confidence|sure|certain)",
            r"(\\d+(?:\\.\\d+))\\s*percent",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                conf = float(match.group(1))
                if conf > 1:
                    conf /= 100
                return min(max(conf, 0.0), 1.0)
        
        return 0.7  # Default moderate confidence

    def _extract_findings(self, text: str) -> list[str]:
        """Extract findings from response."""
        # Look for "Findings:" or "Observations:" section
        finding_section = self._extract_section(text, ["findings", "observations", "description", "analysis"])
        
        if finding_section:
            # Split by bullet points or numbered items
            items = re.split(r'[\n•\-*]|\d+\.', finding_section)
            return [item.strip("- *•\t ") for item in items if len(item.strip()) > 10][:5]
        
        # Fallback: extract sentences that describe observations
        sentences = re.split(r'(?<=[.!?])\\s+', text)
        findings = [s.strip() for s in sentences if any(kw in s.lower() for kw in ["appears", "shows", "indicates", "suggests", "demonstrates"])]
        return findings[:3]

    def _extract_recommendations(self, text: str) -> list[str]:
        """Extract recommendations from response."""
        rec_section = self._extract_section(text, ["recommendations", "impression", "conclusion", "next steps"])
        
        if rec_section:
            items = re.split(r'[\n•\-*]|\d+\.', rec_section)
            return [item.strip("- *•\t ") for item in items if len(item.strip()) > 5][:3]
        
        return []

    def _extract_section(self, text: str, headers: list[str]) -> Optional[str]:
        """Extract text section following header keywords."""
        pattern = r'(?:' + '|'.join(headers) + r')[:\s]*\n?(.*?)(?:\n\n|\Z|$)'
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else None

    def _assess_severity(self, text: str) -> str:
        """Assess severity from text keywords."""
        text_lower = text.lower()
        
        urgent_keywords = ["urgent", "emergency", "critical", "severe", "immediate attention", "life-threatening"]
        moderate_keywords = ["moderate", "concerning", "should be evaluated", "recommended", "abnormal"]
        
        if any(kw in text_lower for kw in urgent_keywords):
            return "urgent"
        elif any(kw in text_lower for kw in moderate_keywords):
            return "moderate"
        return "low"

    def _detect_specialist_referral(self, text: str) -> Optional[str]:
        """Detect if specialist referral is mentioned."""
        specialists = {
            "radiologist": ["radiologist", "imaging specialist"],
            "dermatologist": ["dermatologist", "skin specialist"],
            "cardiologist": ["cardiologist", "heart specialist"],
            "pulmonologist": ["pulmonologist", "lung specialist"],
            "oncologist": ["oncologist", "cancer specialist"],
            "neurologist": ["neurologist", "brain specialist"],
        }
        
        text_lower = text.lower()
        for specialist, keywords in specialists.items():
            if any(kw in text_lower for kw in keywords):
                return specialist
        return None

    def _error_response(self, error_msg: str) -> dict[str, Any]:
        """Generate error response structure."""
        return {
            "error": error_msg,
            "raw_response": "",
            "clean_response": f"Analysis error: {error_msg}",
            "findings": ["Error in analysis"],
            "confidence": 0.0,
            "recommendations": ["Please retry or consult specialist directly"],
            "severity": "unknown",
            "specialist_recommended": None,
            "model": self.model_id,
            "input_type": "unknown",
        }

    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()
        return False