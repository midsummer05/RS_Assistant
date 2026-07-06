from __future__ import annotations

from typing import Any, Dict

from rs_agent.agent.state import Event
from rs_agent.storage.json_store import JsonFileStore


class EventBus:
    def __init__(self, store: JsonFileStore) -> None:
        self.store = store

    def emit(self, task_id: str, event_type: str, payload: Dict[str, Any] | None = None) -> Event:
        event = Event(task_id=task_id, event_type=event_type, payload=payload or {})
        self.store.append_event(event)
        return event

