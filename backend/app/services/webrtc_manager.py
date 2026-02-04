"""WebRTC signaling manager with room support."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional, Set

from app.core.config import Settings

logger = logging.getLogger(__name__)

FrameCallback = Callable[[str], Awaitable[None]]


@dataclass
class RoomState:
    """State for an active WebRTC room."""

    room_id: str
    peers: Set[str] = field(default_factory=set)
    offers: Dict[str, Any] = field(default_factory=dict)
    answers: Dict[str, Any] = field(default_factory=dict)
    ice_candidates: Dict[str, list] = field(default_factory=dict)
    last_activity: float = field(default_factory=lambda: time.time())


class WebRTCManager:
    """Handle WebRTC signaling and room lifecycle."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._rooms: Dict[str, RoomState] = {}
        self._lock = asyncio.Lock()
        self._frame_callback: Optional[FrameCallback] = self._default_frame_capture
        self._background_started = False

    def register_frame_callback(self, callback: FrameCallback) -> None:
        """Register a callback for periodic frame capture."""
        self._frame_callback = callback

    async def _default_frame_capture(self, room_id: str) -> None:
        """Default frame capture hook to keep room activity alive."""
        async with self._lock:
            room = self._rooms.get(room_id)
            if room is not None:
                room.last_activity = time.time()

    async def _ensure_background_tasks(self) -> None:
        if self._background_started:
            return
        self._background_started = True
        asyncio.create_task(self._cleanup_loop())
        asyncio.create_task(self._frame_capture_loop())

    async def create_or_join_room(self, room_id: str, peer_id: str) -> RoomState:
        """Create or join a WebRTC room."""
        await self._ensure_background_tasks()
        async with self._lock:
            if room_id not in self._rooms and len(self._rooms) >= self._settings.webrtc_max_rooms:
                await self._evict_oldest_room()
            room = self._rooms.setdefault(room_id, RoomState(room_id=room_id))
            room.peers.add(peer_id)
            room.last_activity = time.time()
            return room

    async def handle_offer(self, room_id: str, peer_id: str, offer: Any) -> None:
        """Store an SDP offer for a room."""
        room = await self.create_or_join_room(room_id, peer_id)
        room.offers[peer_id] = offer

    async def handle_answer(self, room_id: str, peer_id: str, answer: Any) -> None:
        """Store an SDP answer for a room."""
        room = await self.create_or_join_room(room_id, peer_id)
        room.answers[peer_id] = answer

    async def handle_ice_candidate(self, room_id: str, peer_id: str, candidate: Any) -> None:
        """Store ICE candidates for a room."""
        room = await self.create_or_join_room(room_id, peer_id)
        room.ice_candidates.setdefault(peer_id, []).append(candidate)

    async def remove_peer(self, room_id: str, peer_id: str) -> None:
        """Remove a peer from the room."""
        async with self._lock:
            room = self._rooms.get(room_id)
            if room is None:
                return
            room.peers.discard(peer_id)
            room.offers.pop(peer_id, None)
            room.answers.pop(peer_id, None)
            room.ice_candidates.pop(peer_id, None)
            room.last_activity = time.time()
            if not room.peers:
                self._rooms.pop(room_id, None)

    async def get_room_state(self, room_id: str) -> Optional[RoomState]:
        """Return room state if available."""
        async with self._lock:
            return self._rooms.get(room_id)

    async def _evict_oldest_room(self) -> None:
        oldest_room_id = min(self._rooms, key=lambda rid: self._rooms[rid].last_activity)
        logger.warning("Evicting oldest WebRTC room: %s", oldest_room_id)
        self._rooms.pop(oldest_room_id, None)

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(30)
            cutoff = time.time() - self._settings.webrtc_room_ttl_seconds
            async with self._lock:
                stale_rooms = [rid for rid, room in self._rooms.items() if room.last_activity < cutoff]
                for rid in stale_rooms:
                    logger.info("Cleaning up stale WebRTC room: %s", rid)
                    self._rooms.pop(rid, None)

    async def _frame_capture_loop(self) -> None:
        while True:
            await asyncio.sleep(self._settings.webrtc_frame_interval_seconds)
            if self._frame_callback is None:
                continue
            rooms = list(self._rooms.keys())
            for room_id in rooms:
                try:
                    await self._frame_callback(room_id)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.warning("Frame capture failed for room %s: %s", room_id, exc)
