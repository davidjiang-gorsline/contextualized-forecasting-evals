from __future__ import annotations

import os
from dataclasses import dataclass

from cfevals.benchmarks.base import ScenarioBenchmark, ScenarioSample


@dataclass
class ContextIsKeyBenchmark(ScenarioBenchmark):
    dataset_name: str = "ServiceNow/context-is-key"
    split: str = "test"
    max_samples: int | None = None
    cache_dir: str | None = None
    allow_fallback: bool = False

    def load(self) -> list[ScenarioSample]:
        from datasets import load_dataset  # noqa: PLC0415

        cache_dir = self.cache_dir or os.environ.get("CFEVALS_CACHE")
        try:
            ds = load_dataset(self.dataset_name, split=self.split, cache_dir=cache_dir)
        except Exception as exc:  # noqa: BLE001
            if self.allow_fallback:
                return [
                    ScenarioSample(
                        sample_id="cik-fallback-0",
                        history=[1.0, 2.0, 3.0],
                        future=[3.5, 3.7],
                        context_text="Synthetic fallback sample.",
                        roi=(0.0, 10.0),
                    )
                ]
            raise RuntimeError(
                f"Failed to load CiK dataset {self.dataset_name!r}. "
                "Check network access or set CFEVALS_CACHE for offline use."
            ) from exc

        samples: list[ScenarioSample] = []
        limit = self.max_samples or len(ds)
        for idx, row in enumerate(ds):
            if idx >= limit:
                break
            roi = row.get("roi")
            roi_tuple = tuple(roi) if roi else None
            samples.append(
                ScenarioSample(
                    sample_id=row.get("sample_id", f"cik-{idx}"),
                    history=list(row.get("history") or []),
                    future=list(row.get("future") or []),
                    context_text=row.get("context"),
                    roi=roi_tuple,
                    metadata={"dataset": self.dataset_name},
                )
            )
        return samples
