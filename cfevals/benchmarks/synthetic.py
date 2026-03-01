from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from cfevals.benchmarks.base import TimeSeriesBenchmark, TimeSeriesDataset, TimeSeriesPoint
from cfevals.context import ContextEvent


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
    context_stride: int = 12
    context_lag_steps: int = 0

    def load(self) -> TimeSeriesDataset:
        _validate_config(self)
        seasonal_period = self.seasonal_period or infer_seasonal_period(self.frequency)
        pandas_frequency = _to_pandas_frequency(self.frequency)
        index = pd.date_range(start=self.start, periods=self.periods, freq=pandas_frequency)
        points: list[TimeSeriesPoint] = []
        context_events: list[ContextEvent] = []
        offset = pd.tseries.frequencies.to_offset(pandas_frequency)
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
            if i % max(self.context_stride, 1) == 0:
                event_time = _to_datetime(ts)
                available_at = _to_datetime(ts + (offset * max(self.context_lag_steps, 0)))
                text = _context_text(i=i, seasonal=seasonal, trend=self.trend)
                context_events.append(
                    ContextEvent(
                        event_time=event_time,
                        available_at=available_at,
                        modality="text",
                        source="synthetic_generator",
                        text=text,
                        metadata={"index": i},
                    )
                )
        metadata: dict[str, Any] = {
            "generator": "synthetic_seasonal",
            "seasonal_period": seasonal_period,
            "context_event_count": len(context_events),
        }
        if not context_events:
            raise ValueError(
                "SyntheticSeasonalBenchmark produced no context events. "
                "Adjust context_stride/context_lag_steps/periods."
            )
        return TimeSeriesDataset(
            points=points,
            context_events=context_events,
            frequency=self.frequency,
            metadata=metadata,
        )


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


def _context_text(*, i: int, seasonal: float, trend: float) -> str:
    phase = "rising" if seasonal >= 0 else "cooling"
    trend_dir = "uptrend" if trend >= 0 else "downtrend"
    return f"Cycle signal is {phase}; long-run regime remains {trend_dir} at step {i}."


def _validate_config(config: SyntheticSeasonalBenchmark) -> None:
    if config.periods <= 0:
        raise ValueError("SyntheticSeasonalBenchmark.periods must be > 0")
    if not config.frequency:
        raise ValueError("SyntheticSeasonalBenchmark.frequency must be set")
    if config.seasonal_period is not None and config.seasonal_period <= 0:
        raise ValueError("SyntheticSeasonalBenchmark.seasonal_period must be > 0 when provided")
    if config.context_stride <= 0:
        raise ValueError("SyntheticSeasonalBenchmark.context_stride must be > 0")
    if config.context_lag_steps < 0:
        raise ValueError("SyntheticSeasonalBenchmark.context_lag_steps must be >= 0")
