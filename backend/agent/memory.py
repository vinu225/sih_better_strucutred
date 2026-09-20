import time
import uuid
import threading
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone


class ConversationMemory:
    """
    Manages conversational history, analytical tool outputs, and context for an agent session.
    """
    def __init__(self, max_history: int = 20, session_id: Optional[str] = None):
        self.session_id = session_id or f"session_{uuid.uuid4().hex[:12]}"
        self.max_history = max_history
        self.last_accessed: float = time.time()
        self.messages: List[Dict[str, Any]] = []

    def touch(self):
        """Update last accessed timestamp for TTL management."""
        self.last_accessed = time.time()

    def add_user_message(self, text: str, tile_id: str = "current"):
        self.touch()
        self.messages.append({
            "role": "user",
            "content": text,
            "tile_id": tile_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._trim()

    def add_agent_message(
        self,
        text: str,
        tools_used: Optional[List[str]] = None,
        artifacts: Optional[Dict[str, Any]] = None,
        tile_id: str = "current",
    ):
        self.touch()
        self.messages.append({
            "role": "assistant",
            "content": text,
            "tile_id": tile_id,
            "tools_used": tools_used or [],
            "artifacts": artifacts or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._trim()

    def get_history(self) -> List[Dict[str, Any]]:
        self.touch()
        return list(self.messages)

    def get_history_clean(self) -> List[Dict[str, Any]]:
        """
        Return history without heavy visual artifacts (e.g. base64 images or dense matrices),
        suitable for API responses.
        """
        self.touch()
        clean = []
        for m in self.messages:
            item = {
                "role": m.get("role", "user"),
                "content": m.get("content", ""),
                "tile_id": m.get("tile_id"),
                "tools_used": m.get("tools_used", []),
                "timestamp": m.get("timestamp"),
            }
            clean.append(item)
        return clean

    def get_last_assistant_turn(self, tile_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve the most recent assistant message, optionally filtered by tile_id."""
        for m in reversed(self.messages):
            if m.get("role") == "assistant":
                if tile_id is None or m.get("tile_id") == tile_id:
                    return m
        return None

    def get_last_user_turn(self) -> Optional[Dict[str, Any]]:
        """Retrieve the most recent user query."""
        for m in reversed(self.messages):
            if m.get("role") == "user":
                return m
        return None

    def clear(self):
        self.touch()
        self.messages.clear()

    def _trim(self):
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]


class SessionMemoryStore:
    """
    Thread-safe, LRU-capped, TTL-expiring per-session memory store.
    Prevents cross-session history pollution while keeping single heavy model instances.
    """
    def __init__(self, max_sessions: int = 200, ttl_seconds: float = 3600.0):
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._sessions: OrderedDict[str, ConversationMemory] = OrderedDict()
        self._lock = threading.RLock()

    def _cleanup_expired(self, now: float):
        """Remove sessions idle for more than ttl_seconds."""
        expired_keys = [
            sid for sid, mem in self._sessions.items()
            if (now - mem.last_accessed) > self.ttl_seconds
        ]
        for sid in expired_keys:
            del self._sessions[sid]

    def get_or_create(self, session_id: Optional[str] = None) -> Tuple[str, ConversationMemory]:
        """
        Retrieve existing session memory or create a new one.
        Returns (resolved_session_id, conversation_memory).
        """
        with self._lock:
            now = time.time()
            self._cleanup_expired(now)

            if not session_id or not str(session_id).strip():
                sid = f"session_{uuid.uuid4().hex[:12]}"
            else:
                sid = str(session_id).strip()

            if sid in self._sessions:
                mem = self._sessions[sid]
                if (now - mem.last_accessed) > self.ttl_seconds:
                    mem.clear()
                mem.touch()
                self._sessions.move_to_end(sid)
                return sid, mem

            # Evict LRU if cap reached
            if len(self._sessions) >= self.max_sessions:
                self._sessions.popitem(last=False)

            mem = ConversationMemory(session_id=sid)
            self._sessions[sid] = mem
            return sid, mem

    def get(self, session_id: str) -> Optional[ConversationMemory]:
        """Look up a session without creating a new one."""
        with self._lock:
            now = time.time()
            self._cleanup_expired(now)
            if not session_id:
                return None
            sid = str(session_id).strip()
            if sid in self._sessions:
                mem = self._sessions[sid]
                if (now - mem.last_accessed) > self.ttl_seconds:
                    del self._sessions[sid]
                    return None
                mem.touch()
                self._sessions.move_to_end(sid)
                return mem
            return None

    def delete(self, session_id: str) -> bool:
        """Delete / clear a session."""
        with self._lock:
            if not session_id:
                return False
            sid = str(session_id).strip()
            if sid in self._sessions:
                del self._sessions[sid]
                return True
            return False

    def clear_all(self):
        """Clear all active sessions."""
        with self._lock:
            self._sessions.clear()

    def active_session_count(self) -> int:
        """Return number of currently active non-expired sessions."""
        with self._lock:
            now = time.time()
            self._cleanup_expired(now)
            return len(self._sessions)


# Global default session store shared across backend routes
session_store = SessionMemoryStore(max_sessions=200, ttl_seconds=3600.0)

