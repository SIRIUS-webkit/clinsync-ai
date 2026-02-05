"""Central AI orchestration service."""
from __future__ import annotations

import asyncio
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from PIL import Image

from app.core.config import Settings
from app.core.security import Anonymizer
from app.models.hear_wrapper import HeARWrapper
from app.models.medasr_wrapper import MedASRWrapper
from app.models.medgemma_wrapper import MedGemmaWrapper
from app.models.ollama_orchestrator import OllamaOrchestrator

logger = logging.getLogger(__name__)


class AIOrchestrator:
    """Coordinate multimodal model inference and response synthesis."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._executor = ThreadPoolExecutor(max_workers=settings.max_workers)
        
        # Pass individual params, not Settings object
        self._medgemma = MedGemmaWrapper(
            model_id=settings.medgemma_model_id,
            device=settings.device,
            enable_quantization=settings.enable_quantization,
        )
        self._medasr = MedASRWrapper(settings)  # If these need Settings, keep as-is
        self._hear = HeARWrapper(settings)
        self._ollama = OllamaOrchestrator(settings)
        self._anonymizer = Anonymizer(settings.anonymization_salt)

    def _load_image(self, image_bytes: bytes) -> Image.Image:
        """Load image from bytes with error handling."""
        try:
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            logger.error("Failed to load image: %s", e)
            raise ValueError(f"Invalid image data: {e}")

    async def _run_in_thread(self, func, *args, **kwargs) -> Any:
        """Run blocking function in thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: func(*args, **kwargs))

    async def process_request(
        self,
        text: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        audio_bytes: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """
        Process a multimodal request and return structured response.
        
        Args:
            text: Optional text query
            image_bytes: Optional image data
            audio_bytes: Optional audio data
            
        Returns:
            Structured dict with response, findings, and metadata
        """
        # Anonymize input text
        sanitized_text = self._anonymizer.anonymize_text(text or "") if text else ""
        has_image = image_bytes is not None
        has_audio = audio_bytes is not None
        
        logger.info("Processing request - text: %s, image: %s, audio: %s", 
                   bool(sanitized_text), has_image, has_audio)

        # Get routing decision from Ollama
        try:
            routing = await self._ollama.route_decision(
                sanitized_text,
                has_image=has_image,
                has_audio=has_audio,
            )
            logger.info("Routing decision: %s", routing)
        except Exception as e:
            logger.error("Routing failed, using fallback: %s", e)
            routing = self._fallback_routing(has_image, has_audio)

        # Execute model calls based on routing
        tasks = {}
        
        if routing.get("analyze_image") and image_bytes:
            tasks["image"] = self._analyze_image(image_bytes, sanitized_text)
            
        if routing.get("analyze_audio") and audio_bytes:
            tasks["transcript"] = self._analyze_audio_transcription(audio_bytes)
            tasks["audio_analysis"] = self._analyze_audio_biomarkers(audio_bytes)

        # Gather results
        results = {}
        if tasks:
            completed_tasks = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for key, result in zip(tasks.keys(), completed_tasks):
                if isinstance(result, Exception):
                    logger.error("Task %s failed: %s", key, result)
                    results[key] = {"error": str(result)}
                else:
                    results[key] = result

        # Extract individual results
        image_result = results.get("image", {})
        transcript_result = results.get("transcript", {})
        audio_analysis_result = results.get("audio_analysis", {})

        # Synthesize final response
        final_response = await self._synthesize_response(
            text=sanitized_text,
            image_result=image_result,
            transcript=transcript_result.get("text") if isinstance(transcript_result, dict) else None,
            audio_analysis=audio_analysis_result if isinstance(audio_analysis_result, dict) else None,
        )
        logger.info("Final response: %s", final_response)

        return {
            "response": final_response,
            "triage_level": self._determine_triage(image_result, audio_analysis_result),
            "findings": self._extract_findings(image_result, audio_analysis_result),
            "confidence": self._calculate_confidence(image_result, audio_analysis_result),
            "recommendations": self._extract_recommendations(image_result, final_response),
            "transcript": transcript_result.get("text") if isinstance(transcript_result, dict) else None,
            "raw_results": {
                "image": image_result,
                "transcript": transcript_result,
                "audio": audio_analysis_result,
            },
            "routing": routing,
        }

    async def _analyze_image(self, image_bytes: bytes, text: str) -> Dict[str, Any]:
        """Analyze image using MedGemma."""
        try:
            image = self._load_image(image_bytes)
            # Use async wrapper if available, otherwise thread
            if hasattr(self._medgemma, 'generate_async'):
                return await self._medgemma.generate_async(
                    prompt=text or "Analyze this medical image for abnormalities.",
                    image=image,
                )
            else:
                return await self._run_in_thread(
                    self._medgemma.generate,
                    prompt=text or "Analyze this medical image for abnormalities.",
                    image=image,
                )
        except Exception as e:
            logger.error("Image analysis failed: %s", e)
            return {"error": str(e), "findings": [], "confidence": 0}

    async def _analyze_audio_transcription(self, audio_bytes: bytes) -> Dict[str, Any]:
        """Transcribe audio using MedASR."""
        try:
            return await self._run_in_thread(self._medasr.transcribe, audio_bytes)
        except Exception as e:
            logger.error("Audio transcription failed: %s", e)
            return {"error": str(e), "text": None}

    async def _analyze_audio_biomarkers(self, audio_bytes: bytes) -> Dict[str, Any]:
        """Analyze audio biomarkers using HeAR."""
        try:
            return await self._run_in_thread(self._hear.analyze, audio_bytes)
        except Exception as e:
            logger.error("Audio biomarker analysis failed: %s", e)
            return {"error": str(e), "biomarkers": {}}

    async def _synthesize_response(
        self,
        text: str,
        image_result: Dict,
        transcript: Optional[str],
        audio_analysis: Optional[Dict],
    ) -> str:
        """Synthesize final response using Ollama."""
        # Build context for synthesis
        context_parts = ["You are ClinSync AI, a medical virtual assistant."]
        
        if text:
            context_parts.append(f"Patient query: {text}")
        if transcript:
            context_parts.append(f"Transcribed speech: {transcript}")
        if image_result and "findings" in image_result:
            findings = image_result["findings"]
            if isinstance(findings, list):
                context_parts.append(f"Image findings: {', '.join(findings[:3])}")
            else:
                context_parts.append(f"Image analysis: {findings}")
        if audio_analysis and "biomarkers" in audio_analysis:
            context_parts.append(f"Audio biomarkers: {audio_analysis['biomarkers']}")

        prompt = "\n".join(context_parts)
        prompt += "\n\nProvide a concise, clinically appropriate response. Include triage level (LOW/MODERATE/HIGH) and next steps."

        try:
            logger.info("Synthesizing response with prompt ollama: %s", prompt)
            return await self._ollama.synthesize(prompt)
        except Exception as e:
            logger.error("Response synthesis failed: %s", e)
            return self._fallback_response(text, image_result, transcript, audio_analysis)

    def _fallback_routing(self, has_image: bool, has_audio: bool) -> Dict[str, bool]:
        """Simple fallback when Ollama routing fails."""
        return {
            "analyze_image": has_image,
            "analyze_audio": has_audio,
            "fallback": True,
        }

    def _fallback_response(
        self,
        text: str,
        image_result: Dict,
        transcript: Optional[str],
        audio_analysis: Optional[Dict],
    ) -> str:
        """Generate fallback response when synthesis fails."""
        parts = ["I'm analyzing your request."]
        if image_result.get("findings"):
            parts.append(f"Image analysis shows: {image_result['findings'][0] if isinstance(image_result['findings'], list) else image_result['findings']}")
        if transcript:
            parts.append(f"You mentioned: {transcript[:100]}...")
        parts.append("Please consult a healthcare provider for definitive diagnosis.")
        return " ".join(parts)

    def _determine_triage(
        self,
        image_result: Dict,
        audio_analysis: Optional[Dict],
    ) -> str:
        """Determine triage level from model outputs."""
        # Check image severity
        image_severity = image_result.get("severity", "low")
        
        # Check audio biomarkers
        audio_risk = "low"
        if audio_analysis and "biomarkers" in audio_analysis:
            biomarkers = audio_analysis["biomarkers"]
            if isinstance(biomarkers, dict):
                # High risk indicators
                if biomarkers.get("cough", {}).get("severity", 0) > 0.7:
                    audio_risk = "high"
                elif biomarkers.get("breathing", {}).get("wheezing_detected"):
                    audio_risk = "moderate"
        
        # Return highest severity
        if "urgent" in [image_severity] or audio_risk == "high":
            return "HIGH"
        elif image_severity == "moderate" or audio_risk == "moderate":
            return "MODERATE"
        return "LOW"

    def _extract_findings(
        self,
        image_result: Dict,
        audio_analysis: Optional[Dict],
    ) -> list:
        """Extract all findings from results."""
        findings = []
        
        if "findings" in image_result:
            img_findings = image_result["findings"]
            if isinstance(img_findings, list):
                findings.extend(img_findings)
            else:
                findings.append(img_findings)
        
        if audio_analysis and "biomarkers" in audio_analysis:
            biomarkers = audio_analysis["biomarkers"]
            if isinstance(biomarkers, dict):
                # Add relevant audio findings
                if biomarkers.get("cough", {}).get("type"):
                    findings.append(f"Cough type: {biomarkers['cough']['type']}")
        
        return findings[:5]  # Limit to top 5

    def _calculate_confidence(
        self,
        image_result: Dict,
        audio_analysis: Optional[Dict],
    ) -> float:
        """Calculate overall confidence score."""
        scores = []
        
        if "confidence" in image_result:
            scores.append(image_result["confidence"])
        
        if audio_analysis and "confidence" in audio_analysis:
            scores.append(audio_analysis["confidence"])
        
        return sum(scores) / len(scores) if scores else 0.5

    def _extract_recommendations(self, image_result: Dict, response: str) -> list:
        """Extract recommendations from results."""
        recs = []
        
        if "recommendations" in image_result:
            img_recs = image_result["recommendations"]
            if isinstance(img_recs, list):
                recs.extend(img_recs)
            else:
                recs.append(img_recs)
        
        # Could also parse from response text
        return recs[:3]


# Singleton instance management
_orchestrator: Optional[AIOrchestrator] = None


async def get_orchestrator(settings: Settings) -> AIOrchestrator:
    """Get or create orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AIOrchestrator(settings)
    return _orchestrator