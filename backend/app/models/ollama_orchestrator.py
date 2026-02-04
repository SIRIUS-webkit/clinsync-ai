"""Ollama-based LLM orchestrator using Gemma 3."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import aiohttp

from app.core.config import Settings

logger = logging.getLogger(__name__)


class OllamaOrchestrator:
    """Route and synthesize responses via local Ollama Gemma 3."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self._settings.ollama_base_url}/api/generate"
        timeout = aiohttp.ClientTimeout(total=self._settings.ollama_timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                return await response.json()

    async def generate(self, prompt: str, json_mode: bool = False) -> str:
        """Generate a response using Ollama Gemma 3."""
        payload = {
            "model": self._settings.ollama_model,
            "prompt": prompt,
            "stream": False,
        }
        if json_mode:
            payload["format"] = "json"
        try:
            result = await self._post(payload)
            return str(result.get("response", "")).strip()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Ollama unavailable, fallback response used: %s", exc)
            return ""

    async def route_decision(
        self,
        text: str,
        has_image: bool,
        has_audio: bool,
    ) -> Dict[str, Any]:
        """Return routing decision for multimodal inputs."""
        prompt = (
            "You are routing requests for a medical assistant. "
            "Return JSON with keys analyze_image (bool), analyze_audio (bool), "
            "priority (one of urgent, normal, low). "
            f"Text: {text!r}. Image: {has_image}. Audio: {has_audio}."
        )
        response = await self.generate(prompt, json_mode=True)
        try:
            decision = json.loads(response)
            return {
                "analyze_image": bool(decision.get("analyze_image", has_image)),
                "analyze_audio": bool(decision.get("analyze_audio", has_audio)),
                "priority": decision.get("priority", "normal"),
            }
        except json.JSONDecodeError:
            return self._rule_based_decision(text, has_image, has_audio)

    async def synthesize(self, prompt: str) -> str:
        """Synthesize final medical response."""
        response = await self.generate(prompt, json_mode=False)
        if response:
            return response
        return self._rule_based_summary(prompt)

    @staticmethod
    def _rule_based_decision(text: str, has_image: bool, has_audio: bool) -> Dict[str, Any]:
        priority = "normal"
        urgent_terms = ("chest pain", "shortness of breath", "stroke", "severe")
        if any(term in text.lower() for term in urgent_terms):
            priority = "urgent"
        return {
            "analyze_image": has_image,
            "analyze_audio": has_audio,
            "priority": priority,
        }

    @staticmethod
    def _rule_based_summary(prompt: str) -> str:
        return (
            "ClinSync AI could not reach the local LLM. "
            "A clinician should review the provided data for a definitive response. "
            f"Input summary: {prompt[:500]}"
        )
