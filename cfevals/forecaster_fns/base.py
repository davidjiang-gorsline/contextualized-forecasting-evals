from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from cfevals.prompts import ForecastPrompt


@dataclass
class ForecastResult:
    point_forecast: list[float]
    samples: list[list[float]] | None = None
    quantiles: dict[str, list[float]] | None = None
    metadata: dict[str, Any] | None = None


class ForecasterFn(Protocol):
    def __call__(self, prompt: ForecastPrompt) -> ForecastResult:  # pragma: no cover - interface
        ...
