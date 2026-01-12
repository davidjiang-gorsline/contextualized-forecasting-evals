from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd


def default_cache_dir() -> str:
    return os.environ.get("CFEVALS_CACHE", os.path.expanduser("~/.cfevals/cache"))


class DatasetLoader:
    def load(self) -> list[dict[str, Any]]:
        raise NotImplementedError


@dataclass
class FredUnrateLoader(DatasetLoader):
    target_series: str
    covariate_series: str | None = None
    start_date: str = "1976-01-01"

    def load(self) -> list[dict[str, Any]]:
        cache_dir = default_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"fred_{self.target_series}.json")

        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            target = pd.Series(payload["target"], index=pd.to_datetime(payload["index"]))
            covariate = None
            if self.covariate_series and payload.get("covariate"):
                covariate = pd.Series(payload["covariate"], index=pd.to_datetime(payload["index"]))
        else:
            from pandas_datareader import data as web  # noqa: PLC0415

            target = web.DataReader(self.target_series, "fred", self.start_date)[self.target_series]
            covariate = None
            if self.covariate_series:
                covariate = web.DataReader(self.covariate_series, "fred", self.start_date)[self.covariate_series]
                covariate = covariate.reindex(target.index).ffill()
            payload = {
                "index": [str(ts.date()) for ts in target.index],
                "target": target.tolist(),
                "covariate": covariate.tolist() if covariate is not None else None,
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

        samples = [
            {
                "sample_id": f"{self.target_series}-{idx}",
                "timestamp": str(ts.date()),
                "value": float(val) if val is not None else None,
                "covariate": float(covariate.loc[ts]) if covariate is not None else None,
            }
            for idx, (ts, val) in enumerate(target.items())
        ]
        return samples


@dataclass
class ContextIsKeyLoader(DatasetLoader):
    dataset_name: str = "ServiceNow/context-is-key"
    split: str = "test"
    max_samples: int | None = None

    def load(self) -> list[dict[str, Any]]:
        from datasets import load_dataset  # noqa: PLC0415

        try:
            ds = load_dataset(self.dataset_name, split=self.split)
            samples: list[dict[str, Any]] = []
            limit = self.max_samples or len(ds)
            for idx, row in enumerate(ds):
                if idx >= limit:
                    break
                samples.append(
                    {
                        "sample_id": row.get("sample_id", f"cik-{idx}"),
                        "history": row.get("history"),
                        "future": row.get("future"),
                        "context_text": row.get("context"),
                        "roi": row.get("roi"),
                    }
                )
            return samples
        except Exception:  # noqa: BLE001
            return [
                {
                    "sample_id": "cik-fallback-0",
                    "history": [1.0, 2.0, 3.0],
                    "future": [3.5, 3.7],
                    "context_text": "Synthetic fallback sample.",
                    "roi": [0.0, 10.0],
                }
            ]


def load_dataset_from_spec(spec: dict[str, Any]) -> list[dict[str, Any]]:
    loader_path = spec["class"]
    module_name, class_name = loader_path.split(":")
    module = __import__(module_name, fromlist=[class_name])
    loader_cls = getattr(module, class_name)
    loader = loader_cls(**spec.get("args", {}))
    return loader.load()
