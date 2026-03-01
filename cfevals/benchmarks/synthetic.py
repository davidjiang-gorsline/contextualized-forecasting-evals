from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from cfevals.benchmarks.base import TimeSeriesBenchmark, TimeSeriesDataset, TimeSeriesPoint


DEFAULT_SEASONAL_PERIOD_BY_FREQ = {
    "H": 24,
    "D": 7,
    "B": 5,
    "W": 52,
    "M": 12,
    "Q": 4,
    "Y": 1,
    "A": 1,
}


@dataclass
class SyntheticSeasonalBenchmark(TimeSeriesBenchmark):
    start: str = "2010-01-01"
    periods: int = 400
    frequency: str = "D"
    seasonal_period: int | None = None
    level: float = 50.0
    trend: float = 0.05
    seasonal_amplitude: float = 2.5
    covariate_amplitude: float = 1.0

    def load(self) -> TimeSeriesDataset:
        seasonal_period = self.seasonal_period or infer_seasonal_period(self.frequency)
        pandas_frequency = _to_pandas_frequency(self.frequency)
        index = pd.date_range(start=self.start, periods=self.periods, freq=pandas_frequency)
        points: list[TimeSeriesPoint] = []
        for i, ts in enumerate(index):
            seasonal = math.sin((2.0 * math.pi * i) / max(seasonal_period, 1))
            harmonic = math.cos((2.0 * math.pi * i) / max(seasonal_period * 2, 1))
            known_covariate = self.covariate_amplitude * seasonal
            value = self.level + (self.trend * i) + (self.seasonal_amplitude * seasonal) + (0.5 * harmonic)
            points.append(
                TimeSeriesPoint(
                    timestamp=_to_datetime(ts),
                    value=float(value),
                    features={"known_covariate": float(known_covariate)},
                )
            )
        metadata: dict[str, Any] = {
            "generator": "synthetic_seasonal",
            "seasonal_period": seasonal_period,
        }
        return TimeSeriesDataset(points=points, frequency=self.frequency, metadata=metadata)


def infer_seasonal_period(frequency: str | None) -> int:
    if not frequency:
        return 1
    base = frequency.upper()[0]
    return DEFAULT_SEASONAL_PERIOD_BY_FREQ.get(base, 1)


def _to_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return pd.Timestamp(value).to_pydatetime()


def _to_pandas_frequency(frequency: str) -> str:
    if frequency.upper() == "H":
        return "h"
    return frequency
