from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable


@dataclass(frozen=True)
class TimeSeriesPoint:
    timestamp: datetime
    value: float
    features: dict[str, float] | None = None


@dataclass(frozen=True)
class AsOfSlice:
    history: list[float]
    timestamps: list[datetime]
    features: dict[str, list[float]]


@dataclass(frozen=True)
class WalkForwardWindow:
    as_of: datetime
    history: list[float]
    history_timestamps: list[datetime]
    history_features: dict[str, list[float]]
    future: list[float]
    future_timestamps: list[datetime]
    future_features: dict[str, list[float]]
    window_index: int


@dataclass
class TimeSeriesDataset:
    points: list[TimeSeriesPoint]
    frequency: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.points.sort(key=lambda point: point.timestamp)

    def as_of(self, timestamp: datetime) -> AsOfSlice:
        history_points = [p for p in self.points if p.timestamp <= timestamp]
        if not history_points:
            return AsOfSlice(history=[], timestamps=[], features={})
        history = [p.value for p in history_points]
        timestamps = [p.timestamp for p in history_points]
        features = _collect_feature_series(history_points)
        return AsOfSlice(history=history, timestamps=timestamps, features=features)

    def walk_forward_windows(
        self,
        *,
        horizon: int,
        step: int,
        min_train_size: int,
        max_train_size: int | None = None,
        max_windows: int | None = None,
    ) -> Iterable[WalkForwardWindow]:
        total = len(self.points)
        window_index = 0
        start = min_train_size
        while start + horizon <= total:
            history_points = self.points[:start]
            if max_train_size is not None:
                history_points = history_points[-max_train_size:]
            future_points = self.points[start : start + horizon]
            if not history_points or not future_points:
                break
            history = [p.value for p in history_points]
            future = [p.value for p in future_points]
            history_timestamps = [p.timestamp for p in history_points]
            future_timestamps = [p.timestamp for p in future_points]
            history_features = _collect_feature_series(history_points)
            future_features = _collect_feature_series(future_points)
            yield WalkForwardWindow(
                as_of=history_timestamps[-1],
                history=history,
                history_timestamps=history_timestamps,
                history_features=history_features,
                future=future,
                future_timestamps=future_timestamps,
                future_features=future_features,
                window_index=window_index,
            )
            window_index += 1
            if max_windows is not None and window_index >= max_windows:
                break
            start += step


@dataclass(frozen=True)
class ScenarioSample:
    sample_id: str
    history: list[float]
    future: list[float]
    context_text: str | None = None
    roi: tuple[float, float] | None = None
    metadata: dict[str, Any] | None = None


class Benchmark(abc.ABC):
    kind: str

    @abc.abstractmethod
    def load(self) -> Any:
        raise NotImplementedError


class TimeSeriesBenchmark(Benchmark):
    kind = "time_series"

    @abc.abstractmethod
    def load(self) -> TimeSeriesDataset:
        raise NotImplementedError


class ScenarioBenchmark(Benchmark):
    kind = "scenario"

    @abc.abstractmethod
    def load(self) -> list[ScenarioSample]:
        raise NotImplementedError


def _collect_feature_series(points: list[TimeSeriesPoint]) -> dict[str, list[float]]:
    features: dict[str, list[float]] = {}
    for point in points:
        if not point.features:
            continue
        for key, value in point.features.items():
            features.setdefault(key, []).append(float(value))
    return features
