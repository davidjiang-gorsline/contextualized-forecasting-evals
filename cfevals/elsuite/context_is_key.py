from __future__ import annotations

from dataclasses import dataclass

from cfevals.eval import Eval, EvalResult
from cfevals.metrics.probabilistic import rcrps
from cfevals.prompts import ForecastPrompt
from cfevals.record import RecorderBase


@dataclass
class ContextIsKeyEval(Eval):
    forecaster: object
    penalty_weight: float = 1.0
    seed: int = 0
    max_samples: int | None = None

    def __post_init__(self) -> None:
        super().__init__(seed=self.seed, max_samples=self.max_samples)

    def eval_sample(self, sample: dict, *, recorder: RecorderBase) -> EvalResult:
        prompt = ForecastPrompt(
            history=sample["history"],
            horizon=len(sample["future"]),
            context_text=sample.get("context_text"),
        )
        result = self.forecaster(prompt)
        samples = result.samples
        if samples is None and result.quantiles:
            quantiles = [result.quantiles[key] for key in sorted(result.quantiles.keys())]
            samples = [list(vals) for vals in zip(*quantiles)]
        if samples is None:
            samples = [[val] for val in result.point_forecast]
        elif samples and isinstance(samples[0], list):
            samples = [list(vals) for vals in zip(*samples)]
        roi = None
        if sample.get("roi"):
            roi = tuple(sample["roi"])
        scores = [
            rcrps(sample_set, target, roi=roi, penalty_weight=self.penalty_weight)
            for sample_set, target in zip(samples, sample["future"])
        ]
        metric_value = float(sum(scores) / len(scores))
        metrics = {"rcrps": metric_value}
        recorder.record_event(
            "forecast_sampling",
            {"forecast": result.point_forecast, "sample_id": sample["sample_id"]},
            sample_id=sample["sample_id"],
        )
        recorder.record_event(
            "forecast_metrics",
            {"metrics": metrics, "sample_id": sample["sample_id"]},
            sample_id=sample["sample_id"],
        )
        return EvalResult(sample_id=sample["sample_id"], metrics=metrics)
