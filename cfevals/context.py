from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ContextEvent:
    event_time: datetime
    text: str
    available_at: datetime | None = None
    modality: str = "text"
    source: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.event_time, datetime):
            raise TypeError(f"ContextEvent.event_time must be datetime, got {type(self.event_time)!r}")
        if self.available_at is not None and not isinstance(self.available_at, datetime):
            raise TypeError(f"ContextEvent.available_at must be datetime, got {type(self.available_at)!r}")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("ContextEvent.text must be a non-empty string")
        normalized_event_time = _to_naive_utc(self.event_time)
        normalized_available_at = _to_naive_utc(self.available_at) if self.available_at else None
        if normalized_available_at and normalized_available_at < normalized_event_time:
            raise ValueError("ContextEvent.available_at cannot be earlier than event_time")
        object.__setattr__(self, "event_time", normalized_event_time)
        object.__setattr__(self, "available_at", normalized_available_at)

    @property
    def as_of_time(self) -> datetime:
        return self.available_at or self.event_time


def render_context_text(
    events: list[ContextEvent],
    *,
    max_events: int | None = None,
) -> str | None:
    if not events:
        return None
    selected = events
    if max_events is not None and max_events > 0:
        selected = events[-max_events:]
    lines: list[str] = []
    for event in selected:
        parts = [
            f"event_time={event.event_time.isoformat()}",
            f"available_at={event.as_of_time.isoformat()}",
            f"modality={event.modality}",
        ]
        if event.source:
            parts.append(f"source={event.source}")
        lines.append(f"[{', '.join(parts)}] {event.text}")
    return "\n".join(lines)


# TODO(P0): Replace string-only context with structured multimodal payloads
# (text/image/table) and point-in-time availability validation against benchmark calendars.
# TODO(P1): Add retrieval/summarization policies for long context histories with
# deterministic truncation and token budgeting.
# TODO(P2): Add event de-duplication and context cache materialization for large-scale runs.


def _to_naive_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(datetime.UTC).replace(tzinfo=None)
