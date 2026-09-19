from typing import List, Dict, Any
from datetime import datetime, timezone


class ConversationMemory:
    """
    Manages conversational history, analytical tool outputs, and context for an agent session.
    """
    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.messages: List[Dict[str, Any]] = []

    def add_user_message(self, text: str, tile_id: str = "current"):
        self.messages.append({
            "role": "user",
            "content": text,
            "tile_id": tile_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._trim()

    def add_agent_message(self, text: str, tools_used: List[str] = None, artifacts: Dict[str, Any] = None):
        self.messages.append({
            "role": "assistant",
            "content": text,
            "tools_used": tools_used or [],
            "artifacts": artifacts or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._trim()

    def get_history(self) -> List[Dict[str, Any]]:
        return self.messages

    def clear(self):
        self.messages.clear()

    def _trim(self):
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
