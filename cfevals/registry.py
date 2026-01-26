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
    benchmarks: dict[str, dict[str, Any]] = field(default_factory=dict)
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    benchmark_sets: dict[str, dict[str, Any]] = field(default_factory=dict)

    def load(self, paths: list[str] | None = None) -> "Registry":
        for base in paths or DEFAULT_REGISTRY_PATHS:
            if not os.path.isdir(base):
                continue
            for root, dirs, files in os.walk(base):
                dirs.sort()
                files.sort()
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
        payload_type = payload.get("type")
        if payload_type == "benchmark":
            self.benchmarks[spec_id] = payload
            return
        if payload_type == "model":
            self.models[spec_id] = payload
            return
        if payload_type == "benchmark_set":
            self.benchmark_sets[spec_id] = payload
            return
        if "benchmarks" in payload:
            self.benchmark_sets[spec_id] = payload
            return

    def get_benchmark(self, benchmark_id: str) -> dict[str, Any]:
        return self.benchmarks[benchmark_id]

    def get_model(self, model_id: str) -> dict[str, Any]:
        return self.models[model_id]

    def get_benchmark_set(self, set_id: str) -> dict[str, Any]:
        return self.benchmark_sets[set_id]
