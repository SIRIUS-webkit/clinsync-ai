"""MedASR wrapper for streaming speech-to-text with medical vocabulary."""

from __future__ import annotations

import io
import logging
import threading
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
from transformers import pipeline

from app.core.config import Settings

logger = logging.getLogger(__name__)


class MedASRWrapper:
    """Lazy-loading ASR wrapper tuned for medical vocabulary."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pipeline = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """Return True if ASR pipeline is loaded."""
        return self._pipeline is not None

    def _ensure_loaded(self) -> None:
        if self.is_loaded:
            return
        with self._lock:
            if self.is_loaded:
                return
            logger.info("Loading MedASR model: %s", self._settings.medasr_model_id)
            device = 0 if self._settings.device == "cuda" else -1
            try:
                self._pipeline = pipeline(
                    task="automatic-speech-recognition",
                    model=self._settings.medasr_model_id,
                    device=device,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning(
                    "MedASR load failed, falling back to whisper-small: %s", exc
                )
                self._pipeline = pipeline(
                    task="automatic-speech-recognition",
                    model="openai/whisper-small",
                    device=device,
                )

    @staticmethod
    def _decode_audio(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        import tempfile
        import os
        import librosa

        # Try direct read first (for WAV/FLAC)
        try:
            audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if audio_array.ndim > 1:
                audio_array = np.mean(audio_array, axis=1)
            return audio_array, sample_rate
        except Exception:
            # Fallback to temp file for WebM/Opus/MP3 which librosa/sf handle better as files
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
                tmp.write(audio_bytes)
                tmp.flush()
                tmp_path = tmp.name

            try:
                # librosa.load handles resampling and mono conversion automatically
                audio_array, sample_rate = librosa.load(tmp_path, sr=16000, mono=True)
                return audio_array, sample_rate
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe full audio payload into text."""
        self._ensure_loaded()
        if self._pipeline is None:
            raise RuntimeError("MedASR pipeline failed to load.")

        audio_array, _ = self._decode_audio(audio_bytes)

        # Check if audio is too short or silent
        if len(audio_array) < 1600:  # Less than 0.1 seconds at 16kHz
            logger.warning(
                "Audio too short for transcription: %d samples", len(audio_array)
            )
            return ""

        # Check if audio is mostly silence
        if np.max(np.abs(audio_array)) < 0.01:
            logger.warning("Audio appears to be silence")
            return ""

        result = self._pipeline(audio_array)
        text = result.get("text", "").strip()

        # Clean up any epsilon tokens (from models like MedASR)
        import re

        text = re.sub(r"<epsilon>", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        logger.info("Transcription result: '%s'", text[:100] if text else "(empty)")
        return text

    def transcribe_stream(self, audio_bytes: bytes) -> List[str]:
        """Transcribe audio in streaming chunks."""
        self._ensure_loaded()
        if self._pipeline is None:
            raise RuntimeError("MedASR pipeline failed to load.")
        audio_array, sample_rate = self._decode_audio(audio_bytes)
        chunk_size = int(self._settings.asr_chunk_seconds * sample_rate)
        transcripts: List[str] = []
        for start in range(0, len(audio_array), chunk_size):
            chunk = audio_array[start : start + chunk_size]
            result = self._pipeline(chunk)
            text = result.get("text", "").strip()
            if text:
                transcripts.append(text)
        return transcripts
