from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from cfevals.benchmarks.base import ScenarioBenchmark, TimeSeriesBenchmark
from cfevals.engine.backtest import WalkForwardBacktester, WalkForwardConfig
from cfevals.engine.scenario import ScenarioEvaluator
from cfevals.models.base import Model
from cfevals.record import LocalRecorder


@dataclass(frozen=True)
class RunOutput:
    benchmark_id: str
    model_id: str
    metrics: dict[str, float]
    num_samples: int


class Runner:
    def run(
        self,
        *,
        benchmark_id: str,
        benchmark: TimeSeriesBenchmark | ScenarioBenchmark,
        model_id: str,
        model: Model,
        output_dir: Path,
        backtest_config: WalkForwardConfig | None = None,
    ) -> RunOutput:
        output_dir.mkdir(parents=True, exist_ok=True)
        recorder = LocalRecorder(str(output_dir / "events.jsonl"))

        if isinstance(benchmark, TimeSeriesBenchmark):
            dataset = benchmark.load()
            config = backtest_config or WalkForwardConfig(horizon=1)
            results = WalkForwardBacktester().run(dataset, model, config, recorder=recorder)
            metrics = _aggregate_metrics([result.metrics for result in results])
            payload = {
                "benchmark_id": benchmark_id,
                "model_id": model_id,
                "metrics": metrics,
                "num_samples": len(results),
            }
        else:
            samples = benchmark.load()
            results = ScenarioEvaluator().run(samples, model, recorder=recorder)
            metrics = {"rcrps": sum(r.metric for r in results) / len(results) if results else 0.0}
            payload = {
                "benchmark_id": benchmark_id,
                "model_id": model_id,
                "metrics": metrics,
                "num_samples": len(results),
            }

        recorder.close()
        (output_dir / "results.json").write_text(json.dumps(payload, indent=2))
        (output_dir / "results.md").write_text(_render_markdown(payload))
        return RunOutput(**payload)


def _aggregate_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    totals: dict[str, list[float]] = {}
    for metrics in metrics_list:
        for key, value in metrics.items():
            totals.setdefault(key, []).append(float(value))
    return {key: sum(values) / len(values) for key, values in totals.items() if values}


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['benchmark_id']} ({payload['model_id']})",
        "",
        "## Metrics",
    ]
    for key, value in payload["metrics"].items():
        lines.append(f"- **{key}**: {value:.4f}")
    return "\n".join(lines)


def default_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")
