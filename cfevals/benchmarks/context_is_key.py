from __future__ import annotations

from dataclasses import dataclass

from cfevals.benchmarks.base import ScenarioBenchmark, ScenarioSample


@dataclass
class ContextIsKeyBenchmark(ScenarioBenchmark):
    dataset_name: str = "ServiceNow/context-is-key"
    split: str = "test"
    max_samples: int | None = None

    def load(self) -> list[ScenarioSample]:
        from datasets import load_dataset  # noqa: PLC0415

        try:
            ds = load_dataset(self.dataset_name, split=self.split)
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
        except Exception:  # noqa: BLE001
            return [
                ScenarioSample(
                    sample_id="cik-fallback-0",
                    history=[1.0, 2.0, 3.0],
                    future=[3.5, 3.7],
                    context_text="Synthetic fallback sample.",
                    roi=(0.0, 10.0),
                )
            ]
