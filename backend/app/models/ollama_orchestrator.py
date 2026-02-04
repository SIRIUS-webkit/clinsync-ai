"""LLM orchestrator with Ollama and HuggingFace backends."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

import aiohttp

from app.core.config import Settings

logger = logging.getLogger(__name__)


class OllamaOrchestrator:
    """Route and synthesize responses via Ollama or HuggingFace fallback."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._hf_llm: Optional[Any] = None
        self._ollama_available: Optional[bool] = None

    def _get_hf_llm(self) -> Any:
        """Lazy load HuggingFace LLM."""
        if self._hf_llm is None:
            from app.models.huggingface_llm import HuggingFaceLLM
            self._hf_llm = HuggingFaceLLM(self._settings)
        return self._hf_llm

    async def _check_ollama_available(self) -> bool:
        """Check if Ollama is reachable."""
        if self._settings.llm_backend == "huggingface":
            return False
        if self._settings.llm_backend == "ollama":
            return True  # Assume available, will fail on actual call if not

        try:
            url = f"{self._settings.ollama_base_url}/api/tags"
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    return response.status == 200
        except Exception:
            return False

    async def _post_ollama(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to Ollama API."""
        url = f"{self._settings.ollama_base_url}/api/generate"
        timeout = aiohttp.ClientTimeout(total=self._settings.ollama_timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                return await response.json()

    async def _generate_ollama(self, prompt: str, json_mode: bool = False) -> str:
        """Generate using Ollama."""
        payload = {
            "model": self._settings.ollama_model,
            "prompt": prompt,
            "stream": False,
        }
        if json_mode:
            payload["format"] = "json"
        result = await self._post_ollama(payload)
        return str(result.get("response", "")).strip()

    def _generate_hf_sync(self, prompt: str, json_mode: bool = False) -> str:
        """Generate using HuggingFace (sync, run in executor)."""
        hf = self._get_hf_llm()
        if json_mode:
            result = hf.generate_json(prompt)
            return json.dumps(result)
        return hf.generate(prompt)

    async def _generate_hf(self, prompt: str, json_mode: bool = False) -> str:
        """Generate using HuggingFace (async wrapper)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self._generate_hf_sync(prompt, json_mode)
        )

    async def generate(self, prompt: str, json_mode: bool = False) -> str:
        """Generate a response using configured backend."""
        backend = self._settings.llm_backend

        # Check Ollama availability once
        if self._ollama_available is None and backend in ("ollama", "auto"):
            self._ollama_available = await self._check_ollama_available()
            if self._ollama_available:
                logger.info("Ollama is available, using Ollama backend")
            else:
                logger.info("Ollama not available, will use HuggingFace backend")

        # Try Ollama first if configured
        if backend in ("ollama", "auto") and self._ollama_available:
            try:
                return await self._generate_ollama(prompt, json_mode)
            except Exception as exc:
                logger.warning("Ollama generation failed: %s", exc)
                if backend == "ollama":
                    return ""
                # Fall through to HuggingFace for "auto"

        # Use HuggingFace
        if backend in ("huggingface", "auto"):
            try:
                return await self._generate_hf(prompt, json_mode)
            except Exception as exc:
                logger.warning("HuggingFace generation failed: %s", exc)
                return ""

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
            decision = json.loads(response) if response else {}
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
        """Fallback rule-based routing."""
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
        """Fallback rule-based response."""
        return (
            "ClinSync AI could not reach the LLM backend. "
            "A clinician should review the provided data for a definitive response. "
            f"Input summary: {prompt[:500]}"
        )
