"""HeAR wrapper for cough and breathing analysis."""
from __future__ import annotations

import io
import logging
import threading
from typing import Dict, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from app.core.config import Settings

logger = logging.getLogger(__name__)


class HeARWrapper:
    """Lazy-loading HeAR audio classifier with heuristic fallback."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Optional[AutoModelForAudioClassification] = None
        self._extractor: Optional[AutoFeatureExtractor] = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """Return True if the HeAR model is loaded."""
        return self._model is not None and self._extractor is not None

    def _ensure_loaded(self) -> None:
        if self.is_loaded:
            return
        with self._lock:
            if self.is_loaded:
                return
            logger.info("Loading HeAR model: %s", self._settings.hear_model_id)
            self._extractor = AutoFeatureExtractor.from_pretrained(self._settings.hear_model_id)
            self._model = AutoModelForAudioClassification.from_pretrained(
                self._settings.hear_model_id,
            )
            self._model.eval()

    @staticmethod
    def _decode_audio(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)
        return audio_array, sample_rate

    def _heuristic_analysis(self, audio_array: np.ndarray, sample_rate: int) -> Dict[str, float]:
        rms = float(np.mean(librosa.feature.rms(y=audio_array)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=audio_array)))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio_array, sr=sample_rate)))

        cough_score = min(1.0, (zcr * 10.0) + (centroid / 8000.0))
        breathing_irregularity = min(1.0, rms * 8.0)
        return {
            "cough_probability": round(cough_score, 4),
            "breathing_irregularity": round(breathing_irregularity, 4),
        }

    def analyze(self, audio_bytes: bytes) -> Dict[str, float]:
        """Analyze cough and breathing signals."""
        audio_array, sample_rate = self._decode_audio(audio_bytes)
        try:
            self._ensure_loaded()
            if self._model is None or self._extractor is None:
                raise RuntimeError("HeAR model not available.")
            inputs = self._extractor(audio_array, sampling_rate=sample_rate, return_tensors="pt")
            with torch.inference_mode():
                logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            labels = self._model.config.id2label
            scored = {labels[idx]: float(score) for idx, score in enumerate(probs)}
            scored.update(self._heuristic_analysis(audio_array, sample_rate))
            return scored
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("HeAR model inference failed, using heuristics: %s", exc)
            return self._heuristic_analysis(audio_array, sample_rate)
