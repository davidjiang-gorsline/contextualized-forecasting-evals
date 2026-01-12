from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml


DEFAULT_REGISTRY_PATHS = [
    os.path.join(os.path.dirname(__file__), "registry"),
    os.path.expanduser("~/.cfevals"),
]


@dataclass
class Registry:
    evals: dict[str, dict[str, Any]] = field(default_factory=dict)
    datasets: dict[str, dict[str, Any]] = field(default_factory=dict)
    forecasters: dict[str, dict[str, Any]] = field(default_factory=dict)
    forecaster_sets: dict[str, dict[str, Any]] = field(default_factory=dict)
    eval_sets: dict[str, dict[str, Any]] = field(default_factory=dict)

    def load(self, paths: list[str] | None = None) -> "Registry":
        for base in paths or DEFAULT_REGISTRY_PATHS:
            if not os.path.isdir(base):
                continue
            for root, _, files in os.walk(base):
                for fname in files:
                    if not fname.endswith(".yaml"):
                        continue
                    full_path = os.path.join(root, fname)
                    with open(full_path, "r", encoding="utf-8") as f:
                        payload = yaml.safe_load(f) or {}
                    self._register_payload(payload)
        return self

    def _register_payload(self, payload: dict[str, Any] | list[Any]) -> None:
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    self._register_payload(item)
            return
        if not isinstance(payload, dict) or "id" not in payload:
            return
        spec_id = payload["id"]
        if "class" in payload:
            if spec_id.startswith("dataset") or payload.get("type") == "dataset":
                self.datasets[spec_id] = payload
            elif payload.get("type") == "forecaster":
                self.forecasters[spec_id] = payload
            elif payload.get("type") == "eval":
                self.evals[spec_id] = payload
            else:
                self.evals[spec_id] = payload
        elif "members" in payload:
            self.forecaster_sets[spec_id] = payload
        elif payload.get("type") == "eval_set":
            self.eval_sets[spec_id] = payload
        else:
            self.evals[spec_id] = payload

    def get_eval(self, eval_id: str) -> dict[str, Any]:
        return self.evals[eval_id]

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        return self.datasets[dataset_id]

    def get_forecaster(self, forecaster_id: str) -> dict[str, Any]:
        return self.forecasters[forecaster_id]

    def get_forecaster_set(self, set_id: str) -> dict[str, Any]:
        return self.forecaster_sets[set_id]

    def get_eval_set(self, set_id: str) -> dict[str, Any]:
        return self.eval_sets[set_id]
