from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any


class RecorderBase:
    def __init__(self) -> None:
        self._sample_id: str | None = None

    def set_sample_id(self, sample_id: str) -> None:
        self._sample_id = sample_id

    def record_event(self, event_type: str, payload: dict[str, Any], *, sample_id: str | None = None) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None


class NullRecorder(RecorderBase):
    def record_event(self, event_type: str, payload: dict[str, Any], *, sample_id: str | None = None) -> None:
        return None


@dataclass
class LocalRecorder(RecorderBase):
    path: str

    def __post_init__(self) -> None:
        super().__init__()
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")
        self._lock = threading.Lock()
        self._closed = False

    def record_event(self, event_type: str, payload: dict[str, Any], *, sample_id: str | None = None) -> None:
        if self._closed:
            raise RuntimeError("LocalRecorder is already closed")
        event = {
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "sample_id": sample_id or self._sample_id,
            "payload": payload,
        }
        line = json.dumps(event, default=str)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._fh.close()
            self._closed = True


_thread_local = threading.local()


def default_recorder() -> RecorderBase:
    recorder = getattr(_thread_local, "recorder", None)
    if recorder is None:
        recorder = NullRecorder()
        _thread_local.recorder = recorder
    return recorder
