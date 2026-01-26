from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from cfevals.benchmarks.base import TimeSeriesBenchmark, TimeSeriesDataset, TimeSeriesPoint


def default_cache_dir() -> str:
    return os.environ.get("CFEVALS_CACHE", os.path.expanduser("~/.cfevals/cache"))


@dataclass
class FredUnrateBenchmark(TimeSeriesBenchmark):
    target_series: str = "UNRATE"
    covariate_series: str | None = None
    start_date: str = "1976-01-01"
    frequency: str | None = "M"

    def load(self) -> TimeSeriesDataset:
        cache_dir = default_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = self.target_series.lower()
        cache_path = os.path.join(cache_dir, f"fred_{cache_key}.json")

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
                covariate = web.DataReader(self.covariate_series, "fred", self.start_date)[
                    self.covariate_series
                ]
                covariate = covariate.reindex(target.index).ffill()
            payload = {
                "index": [str(ts.date()) for ts in target.index],
                "target": target.tolist(),
                "covariate": covariate.tolist() if covariate is not None else None,
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

        points: list[TimeSeriesPoint] = []
        for ts, val in target.items():
            features: dict[str, float] | None = None
            if covariate is not None and ts in covariate.index:
                features = {"covariate": float(covariate.loc[ts])}
            points.append(
                TimeSeriesPoint(
                    timestamp=_normalize_timestamp(ts),
                    value=float(val) if val is not None else float("nan"),
                    features=features,
                )
            )
        metadata: dict[str, Any] = {
            "target_series": self.target_series,
            "covariate_series": self.covariate_series,
        }
        return TimeSeriesDataset(points=points, frequency=self.frequency, metadata=metadata)


def _normalize_timestamp(ts: Any) -> datetime:
    if isinstance(ts, datetime):
        return ts
    return pd.to_datetime(ts).to_pydatetime()
