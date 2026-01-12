from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from cfevals.eval import Eval, EvalResult
from cfevals.metrics.point import mae, mase, rmse, smape
from cfevals.prompts import ForecastPrompt
from cfevals.record import RecorderBase


@dataclass
class FredUnrateEval(Eval):
    forecaster: object
    horizons: list[int]
    train_window: int
    step: int = 1
    use_covariate: bool = False
    seed: int = 0
    max_samples: int | None = None

    def __post_init__(self) -> None:
        super().__init__(seed=self.seed, max_samples=self.max_samples)

    def build_samples(self, series: list[float], covariate: list[float] | None = None) -> list[dict]:
        samples = []
        total = len(series)
        max_lag = 6 if self.use_covariate else 0
        start_index = self.train_window + max_lag
        for start in range(start_index, total - max(self.horizons), self.step):
            history = series[start - self.train_window : start]
            for horizon in self.horizons:
                target = series[start : start + horizon]
                sample_id = f"unrate-{start}-h{horizon}"
                covariates = None
                if self.use_covariate and covariate is not None:
                    covariates = {}
                    for lag in range(1, 7):
                        lagged = covariate[start - self.train_window - lag : start - lag]
                        covariates[f"epu_lag{lag}"] = lagged
                samples.append(
                    {
                        "sample_id": sample_id,
                        "history": history,
                        "target": target,
                        "horizon": horizon,
                        "covariates": covariates,
                    }
                )
        return samples

    def eval_sample(self, sample: dict, *, recorder: RecorderBase) -> EvalResult:
        prompt = ForecastPrompt(
            history=sample["history"],
            horizon=sample["horizon"],
            covariates=sample.get("covariates"),
        )
        result = self.forecaster(prompt)
        forecast = result.point_forecast
        metrics = {
            "mae": mae(sample["target"], forecast),
            "rmse": rmse(sample["target"], forecast),
            "smape": smape(sample["target"], forecast),
            "mase": mase(sample["target"], forecast, sample["history"]),
            "forecast": forecast,
            "actual": sample["target"],
        }
        recorder.record_event(
            "forecast_sampling",
            {"forecast": forecast, "sample_id": sample["sample_id"]},
            sample_id=sample["sample_id"],
        )
        recorder.record_event(
            "forecast_metrics",
            {"metrics": metrics, "sample_id": sample["sample_id"]},
            sample_id=sample["sample_id"],
        )
        return EvalResult(sample_id=sample["sample_id"], metrics=metrics)
