"""WebRTC AI Handler for real-time video calling with AI.

This module provides real-time audio/video streaming via WebRTC,
enabling low-latency AI consultation similar to Gemini video calls.
"""

from __future__ import annotations

import asyncio
import fractions
import logging
import time
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import io

from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceCandidate,
    MediaStreamTrack,
)
from aiortc.contrib.media import MediaRelay
from av import AudioFrame
from pydub import AudioSegment

from app.services.speech_recognition_service import get_speech_recognition_service
from app.services.ai_orchestrator import AIOrchestrator

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Process incoming audio frames for speech recognition."""

    def __init__(self):
        self.audio_buffer: List[bytes] = []
        self.sample_rate = 16000
        self.channels = 1
        self.silence_threshold = 500  # RMS threshold for silence
        self.silence_duration = 0
        self.silence_max_duration = 1.5  # Seconds of silence to trigger processing
        self.is_speaking = False
        self.last_speech_time = 0
        self.min_speech_duration = 0.5  # Minimum seconds of speech to process

    def add_frame(self, frame: AudioFrame) -> Optional[bytes]:
        """
        Add an audio frame to the buffer.
        Returns the accumulated audio when silence is detected after speech.
        """
        # Convert audio frame to numpy array
        audio_data = frame.to_ndarray()

        # Handle stereo to mono conversion
        if len(audio_data.shape) > 1 and audio_data.shape[0] > 1:
            audio_data = np.mean(audio_data, axis=0)

        audio_data = audio_data.flatten()

        # Calculate RMS for voice activity detection
        rms = np.sqrt(np.mean(audio_data**2))

        current_time = time.time()

        # Voice activity detected
        if rms > self.silence_threshold:
            if not self.is_speaking:
                logger.debug("Speech started")
                self.is_speaking = True
                self.audio_buffer = []  # Clear buffer at speech start

            self.last_speech_time = current_time
            self.silence_duration = 0

            # Add audio to buffer
            # Resample if needed
            if frame.sample_rate != self.sample_rate:
                # Simple resampling using numpy
                audio_data = self._resample(
                    audio_data, frame.sample_rate, self.sample_rate
                )

            # Convert to bytes (16-bit PCM)
            audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
            self.audio_buffer.append(audio_bytes)

        else:
            # Silence detected
            if self.is_speaking:
                self.silence_duration = current_time - self.last_speech_time

                # Still add audio during brief silence
                if frame.sample_rate != self.sample_rate:
                    audio_data = self._resample(
                        audio_data, frame.sample_rate, self.sample_rate
                    )
                audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
                self.audio_buffer.append(audio_bytes)

                # Check if silence duration exceeded threshold
                if self.silence_duration >= self.silence_max_duration:
                    speech_duration = self.last_speech_time - (
                        current_time - self.silence_duration - self.silence_max_duration
                    )

                    if len(self.audio_buffer) > 0:
                        logger.debug(
                            "Speech ended, processing %d chunks", len(self.audio_buffer)
                        )
                        self.is_speaking = False

                        # Combine all audio chunks
                        combined_audio = b"".join(self.audio_buffer)
                        self.audio_buffer = []

                        return combined_audio

        return None

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple resampling using linear interpolation."""
        if orig_sr == target_sr:
            return audio

        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)

        indices = np.linspace(0, len(audio) - 1, target_length)
        return np.interp(indices, np.arange(len(audio)), audio)

    def clear(self):
        """Clear the audio buffer."""
        self.audio_buffer = []
        self.is_speaking = False
        self.silence_duration = 0


class AIAudioTrack(MediaStreamTrack):
    """Audio track that plays AI responses."""

    kind = "audio"

    def __init__(self):
        super().__init__()
        self.audio_queue: asyncio.Queue = asyncio.Queue()
        self.sample_rate = 16000
        self.samples_per_frame = 960  # 60ms at 16kHz
        self._timestamp = 0
        self._playing = False
        self._current_audio: Optional[np.ndarray] = None
        self._audio_position = 0

    async def recv(self) -> AudioFrame:
        """Generate audio frames for the AI response."""
        # Check for new audio in queue
        if self._current_audio is None or self._audio_position >= len(
            self._current_audio
        ):
            try:
                # Non-blocking check for new audio
                self._current_audio = self.audio_queue.get_nowait()
                self._audio_position = 0
                self._playing = True
            except asyncio.QueueEmpty:
                # Generate silence
                self._current_audio = None
                self._playing = False

        # Generate frame
        if self._current_audio is not None and self._audio_position < len(
            self._current_audio
        ):
            # Get chunk of audio
            end_pos = min(
                self._audio_position + self.samples_per_frame, len(self._current_audio)
            )
            chunk = self._current_audio[self._audio_position : end_pos]
            self._audio_position = end_pos

            # Pad if necessary
            if len(chunk) < self.samples_per_frame:
                chunk = np.pad(chunk, (0, self.samples_per_frame - len(chunk)))
        else:
            # Silence
            chunk = np.zeros(self.samples_per_frame, dtype=np.int16)

        # Create audio frame
        frame = AudioFrame.from_ndarray(
            chunk.reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate = self.sample_rate
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, self.sample_rate)

        self._timestamp += self.samples_per_frame

        # Simulate real-time playback
        await asyncio.sleep(self.samples_per_frame / self.sample_rate)

        return frame

    async def add_speech(self, text: str):
        """Convert text to speech and add to queue."""
        # Use pyttsx3 or gTTS for TTS
        try:
            from gtts import gTTS
            import tempfile

            tts = gTTS(text=text, lang="en", slow=False)

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tts.save(f.name)

                # Load and convert to numpy array
                audio = AudioSegment.from_mp3(f.name)
                audio = audio.set_frame_rate(self.sample_rate).set_channels(1)

                samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
                await self.audio_queue.put(samples)

        except ImportError:
            logger.warning("gTTS not available, TTS disabled")
        except Exception as e:
            logger.error("TTS failed: %s", e)


class WebRTCAISession:
    """Manages a WebRTC session with AI processing."""

    def __init__(
        self,
        orchestrator: AIOrchestrator,
        on_transcript: Optional[Callable[[str], None]] = None,
        on_response: Optional[Callable[[str], None]] = None,
        on_state_change: Optional[Callable[[str], None]] = None,
    ):
        self.pc: Optional[RTCPeerConnection] = None
        self.orchestrator = orchestrator
        self.audio_processor = AudioProcessor()
        self.ai_audio_track: Optional[AIAudioTrack] = None
        self.relay = MediaRelay()

        # Callbacks
        self.on_transcript = on_transcript
        self.on_response = on_response
        self.on_state_change = on_state_change

        # State
        self.current_image: Optional[bytes] = None
        self._processing_task: Optional[asyncio.Task] = None
        self._closed = False

    async def create_offer(self) -> RTCSessionDescription:
        """Create WebRTC offer for client."""
        self.pc = RTCPeerConnection()

        # Add AI audio track for responses
        self.ai_audio_track = AIAudioTrack()
        self.pc.addTrack(self.ai_audio_track)

        # Handle incoming tracks
        @self.pc.on("track")
        async def on_track(track: MediaStreamTrack):
            logger.info("Received track: %s", track.kind)

            if track.kind == "audio":
                asyncio.create_task(self._process_audio_track(track))
            elif track.kind == "video":
                asyncio.create_task(self._process_video_track(track))

        @self.pc.on("connectionstatechange")
        async def on_connection_state_change():
            logger.info("Connection state: %s", self.pc.connectionState)
            if self.on_state_change:
                self.on_state_change(self.pc.connectionState)

        # Create offer
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        return self.pc.localDescription

    async def handle_answer(self, sdp: str, sdp_type: str):
        """Handle SDP answer from client."""
        if not self.pc:
            raise ValueError("Peer connection not initialized")

        answer = RTCSessionDescription(sdp=sdp, type=sdp_type)
        await self.pc.setRemoteDescription(answer)

    async def handle_offer(self, sdp: str, sdp_type: str) -> RTCSessionDescription:
        """Handle SDP offer from client and return answer."""
        self.pc = RTCPeerConnection()

        # Add AI audio track for responses
        self.ai_audio_track = AIAudioTrack()
        self.pc.addTrack(self.ai_audio_track)

        # Handle incoming tracks
        @self.pc.on("track")
        async def on_track(track: MediaStreamTrack):
            logger.info("Received track: %s", track.kind)

            if track.kind == "audio":
                asyncio.create_task(self._process_audio_track(track))
            elif track.kind == "video":
                asyncio.create_task(self._process_video_track(track))

        @self.pc.on("connectionstatechange")
        async def on_connection_state_change():
            logger.info("Connection state: %s", self.pc.connectionState)
            if self.on_state_change:
                self.on_state_change(self.pc.connectionState)

        # Set remote description
        offer = RTCSessionDescription(sdp=sdp, type=sdp_type)
        await self.pc.setRemoteDescription(offer)

        # Create and set local answer
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)

        return self.pc.localDescription

    async def add_ice_candidate(self, candidate: dict):
        """Add ICE candidate from client."""
        if self.pc:
            await self.pc.addIceCandidate(RTCIceCandidate(**candidate))

    async def _process_audio_track(self, track: MediaStreamTrack):
        """Process incoming audio track for speech recognition."""
        logger.info("Starting audio processing")

        while not self._closed:
            try:
                frame = await track.recv()

                # Process frame for VAD and buffering
                audio_data = self.audio_processor.add_frame(frame)

                if audio_data:
                    # Speech segment detected, process it
                    asyncio.create_task(self._process_speech(audio_data))

            except Exception as e:
                if not self._closed:
                    logger.error("Audio processing error: %s", e)
                break

        logger.info("Audio processing ended")

    async def _process_video_track(self, track: MediaStreamTrack):
        """Process incoming video track for context capture."""
        logger.info("Starting video processing")

        frame_count = 0

        while not self._closed:
            try:
                frame = await track.recv()
                frame_count += 1

                # Capture frame every 2 seconds for context
                if frame_count % 60 == 0:  # Assuming 30fps
                    try:
                        # Convert frame to JPEG
                        img = frame.to_image()
                        buffer = io.BytesIO()
                        img.save(buffer, format="JPEG", quality=70)
                        self.current_image = buffer.getvalue()
                        logger.debug("Captured video frame for context")
                    except Exception as e:
                        logger.warning("Frame capture failed: %s", e)

            except Exception as e:
                if not self._closed:
                    logger.error("Video processing error: %s", e)
                break

        logger.info("Video processing ended")

    async def _process_speech(self, audio_data: bytes):
        """Process detected speech segment."""
        if self.on_state_change:
            self.on_state_change("processing")

        try:
            # Convert raw PCM to audio file format
            audio_segment = AudioSegment(
                data=audio_data, sample_width=2, frame_rate=16000, channels=1  # 16-bit
            )

            # Export as WAV
            buffer = io.BytesIO()
            audio_segment.export(buffer, format="wav")
            wav_bytes = buffer.getvalue()

            # Transcribe using SpeechRecognition
            stt = get_speech_recognition_service()
            transcript = stt.transcribe(wav_bytes)

            if not transcript:
                logger.debug("Empty transcript, ignoring")
                if self.on_state_change:
                    self.on_state_change("listening")
                return

            logger.info("Transcript: %s", transcript)

            if self.on_transcript:
                self.on_transcript(transcript)

            # Process with AI orchestrator
            result = await self.orchestrator.process_request(
                text=transcript,
                image_bytes=self.current_image,
                audio_bytes=None,  # Already transcribed
                mode="voice",
            )

            response_text = result.get("response", "")

            if response_text:
                logger.info("AI Response: %s", response_text[:100])

                if self.on_response:
                    self.on_response(response_text)

                # Generate TTS and play
                if self.on_state_change:
                    self.on_state_change("speaking")

                if self.ai_audio_track:
                    await self.ai_audio_track.add_speech(response_text)

            if self.on_state_change:
                self.on_state_change("listening")

        except Exception as e:
            logger.error("Speech processing error: %s", e)
            if self.on_state_change:
                self.on_state_change("listening")

    async def close(self):
        """Close the WebRTC session."""
        self._closed = True

        if self.pc:
            await self.pc.close()
            self.pc = None

        self.audio_processor.clear()


# Session manager
class WebRTCAIManager:
    """Manages multiple WebRTC AI sessions."""

    def __init__(self):
        self.sessions: Dict[str, WebRTCAISession] = {}

    def create_session(
        self, session_id: str, orchestrator: AIOrchestrator, **kwargs
    ) -> WebRTCAISession:
        """Create a new WebRTC AI session."""
        session = WebRTCAISession(orchestrator, **kwargs)
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[WebRTCAISession]:
        """Get an existing session."""
        return self.sessions.get(session_id)

    async def close_session(self, session_id: str):
        """Close and remove a session."""
        session = self.sessions.pop(session_id, None)
        if session:
            await session.close()

    async def close_all(self):
        """Close all sessions."""
        for session_id in list(self.sessions.keys()):
            await self.close_session(session_id)


# Global manager instance
_manager: Optional[WebRTCAIManager] = None


def get_webrtc_ai_manager() -> WebRTCAIManager:
    """Get the global WebRTC AI manager."""
    global _manager
    if _manager is None:
        _manager = WebRTCAIManager()
    return _manager
