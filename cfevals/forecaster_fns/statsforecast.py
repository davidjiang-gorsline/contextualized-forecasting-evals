from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cfevals.forecaster_fns.base import ForecastResult
from cfevals.prompts import ForecastPrompt


MODEL_REGISTRY = {
    "Naive": "statsforecast.models:Naive",
    "SeasonalNaive": "statsforecast.models:SeasonalNaive",
    "AutoARIMA": "statsforecast.models:AutoARIMA",
    "AutoETS": "statsforecast.models:AutoETS",
    "AutoTheta": "statsforecast.models:AutoTheta",
}


@dataclass
class StatsForecastForecaster:
    model_name: str
    season_length: int = 12

    def __post_init__(self) -> None:
        if importlib.util.find_spec("statsforecast") is None:
            raise RuntimeError("statsforecast is not installed")
        module_name, class_name = MODEL_REGISTRY[self.model_name].split(":")
        module = __import__(module_name, fromlist=[class_name])
        self.model_cls = getattr(module, class_name)

    def __call__(self, prompt: ForecastPrompt) -> ForecastResult:
        history = np.asarray(prompt.history, dtype=float)
        df = pd.DataFrame({"unique_id": "series", "ds": range(len(history)), "y": history})
        model = self.model_cls(season_length=self.season_length) if self.model_name != "Naive" else self.model_cls()
        from statsforecast import StatsForecast  # noqa: PLC0415

        sf = StatsForecast(models=[model], freq="D")
        sf.fit(df)
        forecast_df = sf.predict(h=prompt.horizon)
        values = forecast_df[self.model_name].tolist()
        return ForecastResult(point_forecast=[float(v) for v in values])
