from __future__ import annotations

from dataclasses import dataclass

from cfevals.benchmarks.base import ScenarioSample
from cfevals.engine.validation import normalize_samples, validate_forecast_result
from cfevals.metrics.probabilistic import rcrps
from cfevals.models.base import ForecastRequest, ForecastResult, Model
from cfevals.record import RecorderBase


@dataclass(frozen=True)
class ScenarioResult:
    sample_id: str
    metric: float


class ScenarioEvaluator:
    def run(self, samples: list[ScenarioSample], model: Model, *, recorder: RecorderBase) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        model.reset()
        for sample in samples:
            request = ForecastRequest(
                history=sample.history,
                horizon=len(sample.future),
                context_text=sample.context_text,
                metadata=sample.metadata,
            )
            result = model.predict(request)
            context = f"scenario sample {sample.sample_id}"
            validate_forecast_result(result, len(sample.future), context=context)
            samples_matrix = _expand_samples(result, len(sample.future), context=context)
            if not samples_matrix:
                raise ValueError(f"{context}: no samples available for RCRPS scoring")
            for idx, sample_set in enumerate(samples_matrix):
                if not sample_set:
                    raise ValueError(f"{context}: empty sample set at horizon index {idx}")
            roi = sample.roi
            scores = [
                rcrps(sample_set, target, roi=roi, penalty_weight=1.0)
                for sample_set, target in zip(samples_matrix, sample.future)
            ]
            metric_value = float(sum(scores) / len(scores))
            recorder.record_event(
                "scenario_result",
                {"sample_id": sample.sample_id, "rcrps": metric_value},
                sample_id=sample.sample_id,
            )
            results.append(ScenarioResult(sample_id=sample.sample_id, metric=metric_value))
        return results


def _expand_samples(result: ForecastResult, horizon: int, *, context: str) -> list[list[float]]:
    if result.samples is None and result.quantiles:
        keys = _sorted_quantile_keys(result.quantiles)
        quantiles = [result.quantiles[key] for key in keys]
        return [list(map(float, vals)) for vals in zip(*quantiles)]
    if result.samples is None:
        return [[float(val)] for val in result.point_forecast]
    samples = normalize_samples(result, horizon, context=context)
    if samples is None:
        return [[float(val)] for val in result.point_forecast]
    return [list(map(float, vals)) for vals in zip(*samples)]


def _sorted_quantile_keys(quantiles: dict[str, list[float]]) -> list[str]:
    def sort_key(key: str) -> tuple[int, float | str]:
        try:
            return (0, float(key))
        except ValueError:
            return (1, key)

    return sorted(quantiles.keys(), key=sort_key)
