from __future__ import annotations

from dataclasses import dataclass

from cfevals.models.base import ForecastRequest, ForecastResult, Model


@dataclass
class LastValueModel(Model):
    fallback_value: float = 0.0

    def predict(self, request: ForecastRequest) -> ForecastResult:
        if request.history:
            value = float(request.history[-1])
            justification = "Naive baseline: repeat the last observed value across the forecast horizon."
        else:
            value = self.fallback_value
            justification = "Naive baseline fallback: no history available, using configured fallback value."
        return ForecastResult(
            point_forecast=[value for _ in range(request.horizon)],
            justification_text=justification,
        )
