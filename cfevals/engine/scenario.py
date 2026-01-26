from __future__ import annotations

from dataclasses import dataclass

from cfevals.benchmarks.base import ScenarioSample
from cfevals.metrics.probabilistic import rcrps
from cfevals.models.base import ForecastRequest, Model
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
            samples_matrix = _expand_samples(result)
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


def _expand_samples(result) -> list[list[float]]:
    if result.samples is None and result.quantiles:
        quantiles = [result.quantiles[key] for key in sorted(result.quantiles.keys())]
        return [list(vals) for vals in zip(*quantiles)]
    if result.samples is None:
        return [[val] for val in result.point_forecast]
    if result.samples and isinstance(result.samples[0], list):
        return [list(vals) for vals in zip(*result.samples)]
    return result.samples or []
