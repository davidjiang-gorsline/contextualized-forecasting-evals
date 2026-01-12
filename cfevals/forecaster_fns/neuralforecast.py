from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cfevals.forecaster_fns.base import ForecastResult
from cfevals.prompts import ForecastPrompt


MODEL_REGISTRY = {
    "NHITS": "neuralforecast.models:NHITS",
    "PatchTST": "neuralforecast.models:PatchTST",
}


@dataclass
class NeuralForecastForecaster:
    model_name: str
    max_steps: int = 50
    freq: str = "D"

    def __post_init__(self) -> None:
        if importlib.util.find_spec("neuralforecast") is None:
            raise RuntimeError("neuralforecast is not installed")
        module_name, class_name = MODEL_REGISTRY[self.model_name].split(":")
        module = __import__(module_name, fromlist=[class_name])
        self.model_cls = getattr(module, class_name)

    def __call__(self, prompt: ForecastPrompt) -> ForecastResult:
        history = np.asarray(prompt.history, dtype=float)
        df = pd.DataFrame({"unique_id": "series", "ds": range(len(history)), "y": history})
        model = self.model_cls(h=prompt.horizon, max_steps=self.max_steps)
        from neuralforecast import NeuralForecast  # noqa: PLC0415

        nf = NeuralForecast(models=[model], freq=self.freq)
        nf.fit(df)
        forecast_df = nf.predict()
        values = forecast_df[self.model_name].tolist()
        return ForecastResult(point_forecast=[float(v) for v in values])
