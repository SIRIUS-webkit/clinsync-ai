"""Central AI orchestration service."""
from __future__ import annotations

import asyncio
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional, ParamSpec, TypeVar

from PIL import Image

from app.core.config import Settings
from app.core.security import Anonymizer
from app.models.hear_wrapper import HeARWrapper
from app.models.medasr_wrapper import MedASRWrapper
from app.models.medgemma_wrapper import MedGemmaWrapper
from app.models.ollama_orchestrator import OllamaOrchestrator

logger = logging.getLogger(__name__)
P = ParamSpec("P")
T = TypeVar("T")


class AIOrchestrator:
    """Coordinate multimodal model inference and response synthesis."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._executor = ThreadPoolExecutor(max_workers=settings.max_workers)
        self._medgemma = MedGemmaWrapper(settings)
        self._medasr = MedASRWrapper(settings)
        self._hear = HeARWrapper(settings)
        self._ollama = OllamaOrchestrator(settings)
        self._anonymizer = Anonymizer(settings.anonymization_salt)

    @staticmethod
    def _load_image(image_bytes: bytes) -> Image.Image:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    async def _run_blocking(self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: func(*args, **kwargs))

    async def process_request(
        self,
        text: Optional[str],
        image_bytes: Optional[bytes],
        audio_bytes: Optional[bytes],
    ) -> Dict[str, Any]:
        """Process a multimodal request and return structured response."""
        sanitized_text = self._anonymizer.anonymize_text(text or "")
        logger.info("Sanitized text: %s", sanitized_text)
        has_image = image_bytes is not None
        has_audio = audio_bytes is not None

        routing = await self._ollama.route_decision(
            sanitized_text,
            has_image=has_image,
            has_audio=has_audio,
        )

        logger.info("Routing: %s", routing)

        tasks = {}
        if routing.get("analyze_image") and image_bytes:
            image = self._load_image(image_bytes)
            tasks["image"] = self._run_blocking(
                self._medgemma.generate,
                prompt=sanitized_text or "Analyze the image for clinical findings.",
                image=image,
            )

        if routing.get("analyze_audio") and audio_bytes:
            tasks["transcript"] = self._run_blocking(self._medasr.transcribe, audio_bytes)
            tasks["audio_analysis"] = self._run_blocking(self._hear.analyze, audio_bytes)

        results: Dict[str, Any] = {}
        if tasks:
            completed = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for key, value in zip(tasks.keys(), completed):
                if isinstance(value, Exception):
                    logger.error("Task %s failed: %s", key, value)
                    results[key] = None
                else:
                    results[key] = value

        image_analysis = results.get("image")
        transcript = results.get("transcript")
        audio_analysis = results.get("audio_analysis")

        prompt_parts = [
            "You are ClinSync AI. Provide a concise, clinically safe response.",
            f"Patient text: {sanitized_text}",
        ]
        if transcript:
            prompt_parts.append(f"ASR transcript: {transcript}")
        if image_analysis:
            prompt_parts.append(f"Image analysis: {image_analysis}")
        if audio_analysis:
            prompt_parts.append(f"Audio analysis: {audio_analysis}")

        final_response = await self._ollama.synthesize("\n".join(prompt_parts))

        return {
            "response": final_response,
            "routing": routing,
            "transcript": transcript,
            "image_analysis": image_analysis,
            "audio_analysis": audio_analysis,
        }
