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
        mode: str = "chat",  # "chat" for structured response, "voice" for conversational
    ) -> Dict[str, Any]:
        """
        Process a multimodal request and return structured response.

        Args:
            text: Optional text query
            image_bytes: Optional image data
            audio_bytes: Optional audio data
            mode: Response mode - "chat" (structured) or "voice" (conversational)

        Returns:
            Structured dict with response, findings, and metadata
        """
        # Anonymize input text
        sanitized_text = self._anonymizer.anonymize_text(text or "") if text else ""
        has_image = image_bytes is not None
        has_audio = audio_bytes is not None

        logger.info(
            "Processing request - text: %s, image: %s, audio: %s",
            bool(sanitized_text),
            has_image,
            has_audio,
        )

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
            completed_tasks = await asyncio.gather(
                *tasks.values(), return_exceptions=True
            )
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

        # Get the user's input text (from direct text or transcription)
        user_input = (
            sanitized_text
            or (
                transcript_result.get("text")
                if isinstance(transcript_result, dict)
                else None
            )
            or ""
        )

        # Classify intent to route appropriately
        intent = await self._classify_intent(user_input)
        logger.info(
            "Detected intent: %s for input: '%s'",
            intent,
            user_input[:50] if user_input else "",
        )

        # Route based on intent
        if intent == "greeting":
            # Simple conversational response - no heavy analysis needed
            final_response = await self._handle_greeting(user_input)
            return {
                "response": final_response,
                "triage_level": "LOW",
                "findings": [],
                "confidence": 1.0,
                "recommendations": [],
                "transcript": (
                    transcript_result.get("text")
                    if isinstance(transcript_result, dict)
                    else None
                ),
                "raw_results": {},
                "routing": {"intent": "greeting"},
            }

        elif intent == "general":
            # General question - use LLM but not full medical analysis
            final_response = await self._handle_general_query(user_input)
            return {
                "response": final_response,
                "triage_level": "LOW",
                "findings": [],
                "confidence": 0.8,
                "recommendations": [],
                "transcript": (
                    transcript_result.get("text")
                    if isinstance(transcript_result, dict)
                    else None
                ),
                "raw_results": {},
                "routing": {"intent": "general"},
            }

        # Medical intent - different handling for chat vs voice mode
        if mode == "voice":
            # Voice mode: Conversational, helpful medical advice
            final_response = await self._synthesize_voice_response(
                user_input=user_input,
                image_result=image_result,
                audio_analysis=audio_analysis_result,
            )
            logger.info("Voice response: %s", final_response)

            return {
                "response": final_response,
                "triage_level": self._determine_triage(
                    image_result, audio_analysis_result
                ),
                "findings": [],  # Not shown in voice mode
                "confidence": self._calculate_confidence(
                    image_result, audio_analysis_result
                ),
                "recommendations": [],
                "transcript": (
                    transcript_result.get("text")
                    if isinstance(transcript_result, dict)
                    else None
                ),
                "raw_results": {},
                "routing": {"intent": "medical", "mode": "voice"},
            }

        # Chat mode: Full structured analysis
        final_response = await self._synthesize_response(
            text=sanitized_text,
            image_result=image_result,
            transcript=(
                transcript_result.get("text")
                if isinstance(transcript_result, dict)
                else None
            ),
            audio_analysis=(
                audio_analysis_result
                if isinstance(audio_analysis_result, dict)
                else None
            ),
        )
        logger.info("Final response: %s", final_response)

        return {
            "response": final_response,
            "triage_level": self._determine_triage(image_result, audio_analysis_result),
            "findings": self._extract_findings(image_result, audio_analysis_result),
            "confidence": self._calculate_confidence(
                image_result, audio_analysis_result
            ),
            "recommendations": self._extract_recommendations(
                image_result, final_response
            ),
            "transcript": (
                transcript_result.get("text")
                if isinstance(transcript_result, dict)
                else None
            ),
            "raw_results": {
                "image": image_result,
                "transcript": transcript_result,
                "audio": audio_analysis_result,
            },
            "routing": {"intent": "medical", **routing},
        }

    async def _classify_intent(self, text: str) -> str:
        """
        Classify user intent using rule-based + LLM approach.
        Returns: 'greeting', 'general', or 'medical'
        """
        if not text:
            return "medical"  # Default to medical if no text (likely image/audio only)

        text_lower = text.lower().strip()

        # Rule-based detection for common patterns (fast path)
        greeting_patterns = [
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "how are you",
            "what's up",
            "howdy",
            "greetings",
            "nice to meet",
            "hi there",
            "hello there",
            "hey there",
        ]

        # Check if it's a simple greeting (short and matches pattern)
        if len(text_lower.split()) <= 6:
            for pattern in greeting_patterns:
                if pattern in text_lower:
                    return "greeting"

        # Medical keywords that indicate health-related query
        medical_keywords = [
            "pain",
            "hurt",
            "ache",
            "fever",
            "cough",
            "sick",
            "symptom",
            "diagnosis",
            "medicine",
            "medication",
            "treatment",
            "doctor",
            "hospital",
            "disease",
            "health",
            "medical",
            "prescription",
            "x-ray",
            "scan",
            "blood",
            "test",
            "headache",
            "stomach",
            "chest",
            "breathing",
            "heart",
            "skin",
            "rash",
            "infection",
            "allergy",
            "injury",
            "broken",
            "swollen",
            "bleeding",
            "nausea",
            "dizzy",
            "fatigue",
            "tired",
            "sleep",
            "appetite",
            "weight",
            "temperature",
            "pulse",
            "pressure",
            "oxygen",
            "diabetes",
            "cancer",
            "surgery",
            "operation",
            "therapy",
            "vaccine",
            "immunization",
        ]

        for keyword in medical_keywords:
            if keyword in text_lower:
                return "medical"

        # For ambiguous cases, use LLM to classify (slow path)
        try:
            classification_prompt = (
                "Classify the following user message into exactly one category:\n"
                "- 'greeting': Simple greetings, pleasantries, or social exchanges\n"
                "- 'general': General questions not related to health/medicine\n"
                "- 'medical': Health-related questions, symptoms, or medical inquiries\n\n"
                f'User message: "{text}"\n\n'
                "Respond with ONLY the category name (greeting, general, or medical):"
            )

            result = await self._ollama.generate(classification_prompt, json_mode=False)
            result_lower = result.lower().strip()

            if "greeting" in result_lower:
                return "greeting"
            elif "general" in result_lower:
                return "general"
            else:
                return "medical"

        except Exception as e:
            logger.warning(
                "Intent classification via LLM failed: %s, defaulting to medical", e
            )
            return "medical"

    async def _handle_greeting(self, text: str) -> str:
        """Handle greeting intent with a friendly conversational response."""
        prompt = (
            "You are ClinSync AI, a friendly medical assistant. "
            "The user has greeted you. Respond warmly and briefly, then ask how you can help them today. "
            "Keep your response under 2 sentences.\n\n"
            f'User said: "{text}"\n\n'
            "Your response:"
        )

        try:
            response = await self._ollama.generate(prompt, json_mode=False)
            return (
                response.strip()
                or "Hello! How can I assist you with your health questions today?"
            )
        except Exception:
            return "Hello! I'm ClinSync AI, your medical assistant. How can I help you today?"

    async def _handle_general_query(self, text: str) -> str:
        """Handle general (non-medical) queries with a helpful response."""
        prompt = (
            "You are ClinSync AI, a medical assistant. "
            "The user has asked a general question that isn't directly about health. "
            "Politely answer briefly if you can, then gently guide them toward health-related topics you can help with.\n\n"
            f'User asked: "{text}"\n\n'
            "Your response:"
        )

        try:
            response = await self._ollama.generate(prompt, json_mode=False)
            return (
                response.strip()
                or "I'm primarily designed to help with health-related questions. Is there anything medical I can assist you with?"
            )
        except Exception:
            return "I'm ClinSync AI, focused on health assistance. Is there a health concern I can help you with today?"

    async def _synthesize_voice_response(
        self,
        user_input: str,
        image_result: Dict,
        audio_analysis: Optional[Dict],
    ) -> str:
        """
        Synthesize a conversational response for real-time voice consultation.
        This is designed to be helpful, natural, and suitable for TTS.
        """
        # Build context about what we know
        context_parts = []

        if user_input:
            context_parts.append(f'Patient says: "{user_input}"')

        # Only mention image if there are meaningful findings
        if image_result and "findings" in image_result:
            findings = image_result.get("findings", [])
            if findings and not any(
                "no" in str(f).lower() or "normal" in str(f).lower()
                for f in (findings[:1] if isinstance(findings, list) else [findings])
            ):
                findings_str = (
                    ", ".join(findings[:2])
                    if isinstance(findings, list)
                    else str(findings)
                )
                context_parts.append(f"Visual observation: {findings_str}")

        if audio_analysis and audio_analysis.get("biomarkers"):
            context_parts.append(f"Audio analysis: {audio_analysis['biomarkers']}")

        context = (
            "\n".join(context_parts)
            if context_parts
            else "No specific context provided."
        )

        prompt = f"""You are ClinSync AI, a helpful medical assistant in a real-time video consultation.

CONTEXT:
{context}

INSTRUCTIONS:
1. Respond as if you're speaking directly to the patient in a video call.
2. Be warm, empathetic, and reassuring.
3. Give practical, actionable advice when appropriate.
4. For common issues (headache, cold, minor pain), you CAN suggest over-the-counter remedies.
5. Always mention when they should see a doctor in person.
6. Keep your response concise (3-5 sentences max) - this will be spoken aloud.
7. DO NOT refuse to help or say you cannot provide medical advice. You ARE providing supportive health guidance.
8. DO NOT use bullet points, numbered lists, or headers - speak naturally.

Example good responses:
- "I'm sorry to hear about your headache. For occasional headaches, you can try ibuprofen or acetaminophen following the package directions. Make sure you're staying hydrated and getting enough rest. If the headache persists for more than a few days or is unusually severe, please see a doctor."
- "Based on what you're describing, it sounds like you might have a mild cold. Rest, fluids, and over-the-counter cold medicine can help with the symptoms. If you develop a high fever or symptoms worsen after a week, it's time to see a healthcare provider."

Now respond to the patient:"""

        try:
            response = await self._ollama.generate(prompt, json_mode=False)
            result = response.strip()

            # Clean up any formatting that shouldn't be in voice response
            result = (
                result.replace("**", "")
                .replace("*", "")
                .replace("#", "")
                .replace("`", "")
            )

            # Remove any numbered lists or bullet points
            import re

            result = re.sub(r"^\d+\.\s*", "", result, flags=re.MULTILINE)
            result = re.sub(r"^[-•]\s*", "", result, flags=re.MULTILINE)

            return (
                result
                or "I understand you're not feeling well. Can you tell me more about your symptoms so I can help you better?"
            )

        except Exception as e:
            logger.error("Voice response synthesis failed: %s", e)
            return "I'm here to help. Please tell me more about what you're experiencing, and I'll do my best to provide guidance."

    async def _analyze_image(self, image_bytes: bytes, text: str) -> Dict[str, Any]:
        """Analyze image using MedGemma."""
        try:
            image = self._load_image(image_bytes)
            # Use async wrapper if available, otherwise thread
            if hasattr(self._medgemma, "generate_async"):
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
            text = await self._run_in_thread(self._medasr.transcribe, audio_bytes)
            return {"text": text}
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
        # Build detailed system prompt for consultation synthesis
        system_instruction = (
            "You are ClinSync AI, an advanced medical virtual assistant conducting a video consultation.\n"
            "Your goal is to provide a professional, empathetic, and clinically sound assessment based on multimodal inputs.\n\n"
            "Analyze the following patient context:\n"
        )

        context_parts = []
        if text:
            context_parts.append(f"- Query/Chat Input: {text}")
        if transcript:
            context_parts.append(f"- Spoken Input (Voice): {transcript}")

        # Handle image findings context intelligently
        if image_result and "findings" in image_result:
            findings = image_result["findings"]
            findings_str = (
                ", ".join(findings) if isinstance(findings, list) else str(findings)
            )
            # If findings are generic/empty, note that
            if not findings or findings_str.lower() in ["none", "[]", "normal"]:
                context_parts.append(
                    "- Visual Analysis: No specific abnormalities detected in current frame."
                )
            else:
                context_parts.append(f"- Visual Analysis (AI Vision): {findings_str}")

        if audio_analysis and "biomarkers" in audio_analysis:
            context_parts.append(f"- Audio Biomarkers: {audio_analysis['biomarkers']}")

        prompt = system_instruction + "\n".join(context_parts)

        prompt += (
            "\n\nGUIDELINES:\n"
            "1. Synthesize the visual findings with the patient's reported symptoms (voice/text).\n"
            "2. If the patient reports symptoms (e.g., fever, pain) that are not visible, acknowledge them and incorporate them into your advice.\n"
            "3. If visual findings exist, explain them simply.\n"
            "4. Assign a **Triage Level** (LOW, MODERATE, HIGH, CRITICAL).\n"
            "5. Provide clear **Next Steps**.\n"
            "6. Keep the response concise (suitable for text-to-speech reading)."
        )

        try:
            logger.info("Synthesizing response with prompt ollama: %s", prompt)
            logger.info(
                "Using LLM Backend for synthesis: %s",
                (
                    self._ollama._settings.ollama_model
                    if self._ollama._settings.llm_backend != "huggingface"
                    else self._ollama._settings.hf_llm_model_id
                ),
            )
            return await self._ollama.synthesize(prompt)
        except Exception as e:
            logger.error("Response synthesis failed: %s", e)
            return self._fallback_response(
                text, image_result, transcript, audio_analysis
            )

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
            parts.append(
                f"Image analysis shows: {image_result['findings'][0] if isinstance(image_result['findings'], list) else image_result['findings']}"
            )
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
