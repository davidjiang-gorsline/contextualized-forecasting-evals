from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np

from cfevals.forecaster_fns.base import ForecastResult
from cfevals.prompts import ForecastPrompt


@dataclass
class ChronosForecaster:
    model_name: str = "amazon/chronos-t5-small"

    def __post_init__(self) -> None:
        if importlib.util.find_spec("chronos") is None:
            raise RuntimeError("chronos-forecasting is not installed")
        from chronos import ChronosPipeline  # noqa: PLC0415

        self.pipeline = ChronosPipeline.from_pretrained(self.model_name)

    def __call__(self, prompt: ForecastPrompt) -> ForecastResult:
        history = np.asarray(prompt.history, dtype=float)
        forecast = self.pipeline.predict(history, prediction_length=prompt.horizon)
        values = forecast.mean(axis=0).tolist()
        samples = forecast.tolist()
        return ForecastResult(point_forecast=[float(v) for v in values], samples=samples)
