from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np

from cfevals.forecaster_fns.base import ForecastResult
from cfevals.prompts import ForecastPrompt


@dataclass
class TimesFMForecaster:
    model_name: str = "google/timesfm-1.0-200m"

    def __post_init__(self) -> None:
        if importlib.util.find_spec("timesfm") is None:
            raise RuntimeError("timesfm is not installed")
        from timesfm import TimesFm  # noqa: PLC0415

        self.model = TimesFm(model_name=self.model_name)

    def __call__(self, prompt: ForecastPrompt) -> ForecastResult:
        history = np.asarray(prompt.history, dtype=float)
        forecast = self.model.forecast(history, horizon=prompt.horizon)
        values = forecast.tolist() if hasattr(forecast, "tolist") else list(forecast)
        return ForecastResult(point_forecast=[float(v) for v in values])
