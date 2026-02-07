"""Speech Recognition service using the SpeechRecognition package.

This module provides speech-to-text functionality using Google's Speech Recognition API
via the SpeechRecognition package. It's designed to be a drop-in replacement for the
Whisper-based transcription.
"""

from __future__ import annotations

import io
import logging
import tempfile
import os
from typing import Optional

import speech_recognition as sr
from pydub import AudioSegment

logger = logging.getLogger(__name__)


class SpeechRecognitionService:
    """Speech-to-text service using SpeechRecognition package."""

    def __init__(self, language: str = "en-US") -> None:
        """Initialize the speech recognition service.

        Args:
            language: Language code for recognition (default: en-US)
        """
        self._recognizer = sr.Recognizer()
        self._language = language

        # Adjust for ambient noise
        self._recognizer.energy_threshold = 300
        self._recognizer.dynamic_energy_threshold = True

        logger.info("SpeechRecognitionService initialized with language: %s", language)

    def _detect_format(self, audio_bytes: bytes) -> str:
        """Detect audio format from bytes header."""
        # Check common audio format signatures
        if audio_bytes[:4] == b"RIFF":
            return "wav"
        elif audio_bytes[:4] == b"OggS":
            return "ogg"
        elif audio_bytes[:3] == b"ID3" or (audio_bytes[0:2] == b"\xff\xfb"):
            return "mp3"
        elif audio_bytes[:4] == b"\x1aE\xdf\xa3":  # EBML header (WebM/Matroska)
            return "webm"
        elif audio_bytes[:4] == b"ftyp" or audio_bytes[4:8] == b"ftyp":
            return "mp4"
        else:
            # Default to webm for browser recordings
            return "webm"

    def _convert_to_wav(self, audio_bytes: bytes) -> bytes:
        """Convert audio bytes to WAV format for recognition.

        Handles WebM, MP3, OGG, and other formats by converting to WAV.
        """
        detected_format = self._detect_format(audio_bytes)
        logger.debug(
            "Detected audio format: %s (size: %d bytes)",
            detected_format,
            len(audio_bytes),
        )

        # Try multiple format suffixes if the detected one fails
        formats_to_try = [detected_format, "webm", "ogg", "mp4", "mp3"]
        # Remove duplicates while preserving order
        formats_to_try = list(dict.fromkeys(formats_to_try))

        last_error = None

        for fmt in formats_to_try:
            tmp_input_path = None
            wav_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=f".{fmt}"
                ) as tmp_input:
                    tmp_input.write(audio_bytes)
                    tmp_input_path = tmp_input.name

                wav_path = tmp_input_path.rsplit(".", 1)[0] + ".wav"

                # Use pydub to handle format conversion
                audio = AudioSegment.from_file(tmp_input_path, format=fmt)

                # Convert to mono 16kHz WAV (optimal for speech recognition)
                audio = audio.set_channels(1).set_frame_rate(16000)
                audio.export(wav_path, format="wav")

                with open(wav_path, "rb") as wav_file:
                    wav_bytes = wav_file.read()

                logger.debug("Successfully converted from %s to WAV", fmt)
                return wav_bytes

            except Exception as e:
                last_error = e
                logger.debug("Format %s failed: %s", fmt, str(e)[:50])
                continue
            finally:
                # Cleanup temp files
                for path in [tmp_input_path, wav_path]:
                    if path and os.path.exists(path):
                        try:
                            os.unlink(path)
                        except Exception:
                            pass

        # All formats failed
        logger.error("Audio conversion failed after trying all formats: %s", last_error)
        raise last_error or Exception("Audio conversion failed")

    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio data (WebM, WAV, MP3, etc.)

        Returns:
            Transcribed text string
        """
        if not audio_bytes or len(audio_bytes) < 100:
            logger.warning("Audio data too short or empty")
            return ""

        try:
            # Convert to WAV format
            wav_bytes = self._convert_to_wav(audio_bytes)

            # Create AudioData from WAV bytes
            with io.BytesIO(wav_bytes) as wav_io:
                with sr.AudioFile(wav_io) as source:
                    audio_data = self._recognizer.record(source)

            # Use Google's free speech recognition API
            try:
                text = self._recognizer.recognize_google(
                    audio_data, language=self._language, show_all=False
                )
                logger.info(
                    "Transcription result: '%s'", text[:100] if text else "(empty)"
                )
                return text.strip() if text else ""

            except sr.UnknownValueError:
                logger.warning("Speech was not understood")
                return ""
            except sr.RequestError as e:
                logger.error("Speech recognition service error: %s", e)
                # Fallback: try with Sphinx (offline, but less accurate)
                try:
                    text = self._recognizer.recognize_sphinx(audio_data)
                    logger.info(
                        "Sphinx fallback result: '%s'",
                        text[:100] if text else "(empty)",
                    )
                    return text.strip() if text else ""
                except Exception:
                    return ""

        except Exception as e:
            logger.error("Transcription failed: %s", e)
            return ""

    def transcribe_realtime(self, audio_bytes: bytes) -> dict:
        """Transcribe audio in real-time mode with additional metadata.

        Args:
            audio_bytes: Raw audio data

        Returns:
            Dict with text and confidence
        """
        text = self.transcribe(audio_bytes)

        return {
            "text": text,
            "confidence": 0.85 if text else 0.0,  # Estimated confidence
            "is_final": True,
        }


# Singleton instance
_service: Optional[SpeechRecognitionService] = None


def get_speech_recognition_service(language: str = "en-US") -> SpeechRecognitionService:
    """Get or create the speech recognition service singleton."""
    global _service
    if _service is None:
        _service = SpeechRecognitionService(language=language)
    return _service
