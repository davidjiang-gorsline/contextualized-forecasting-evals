from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ForecastRequest:
    history: list[float]
    horizon: int
    timestamps: list[datetime] | None = None
    features: dict[str, list[float]] | None = None
    context_text: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ForecastResult:
    point_forecast: list[float]
    samples: list[list[float]] | None = None
    quantiles: dict[str, list[float]] | None = None
    metadata: dict[str, Any] | None = None


class Model(abc.ABC):
    def reset(self) -> None:
        return None

    def fit(self, request: ForecastRequest) -> None:
        return None

    @abc.abstractmethod
    def predict(self, request: ForecastRequest) -> ForecastResult:
        raise NotImplementedError
