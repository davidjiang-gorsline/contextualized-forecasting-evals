from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np

from cfevals.models.base import ForecastRequest, ForecastResult, Model


@dataclass
class ChronosModel(Model):
    model_name: str = "amazon/chronos-t5-small"

    def __post_init__(self) -> None:
        if importlib.util.find_spec("chronos") is None:
            raise RuntimeError("chronos-forecasting is not installed")
        from chronos import ChronosPipeline  # noqa: PLC0415

        self.pipeline = ChronosPipeline.from_pretrained(self.model_name)

    def predict(self, request: ForecastRequest) -> ForecastResult:
        history = np.asarray(request.history, dtype=float)
        forecast = self.pipeline.predict(history, prediction_length=request.horizon)
        values = forecast.mean(axis=0).tolist()
        samples = forecast.tolist()
        return ForecastResult(point_forecast=[float(v) for v in values], samples=samples)
