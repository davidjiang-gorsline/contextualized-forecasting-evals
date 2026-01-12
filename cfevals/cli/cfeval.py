from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from cfevals.data import load_dataset_from_spec
from cfevals.registry import Registry
from cfevals.record import LocalRecorder


def load_class(path: str):
    module_name, class_name = path.split(":")
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def build_forecaster(spec: dict[str, Any]):
    forecaster_cls = load_class(spec["class"])
    return forecaster_cls(**spec.get("args", {}))


def run_eval(eval_id: str, forecaster_set_id: str | None, run_id: str | None) -> None:
    registry = Registry().load()
    eval_spec = registry.get_eval(eval_id)
    dataset_spec = registry.get_dataset(eval_spec["dataset"])

    dataset = load_dataset_from_spec(dataset_spec)
    eval_cls = load_class(eval_spec["class"])

    forecaster_set_name = forecaster_set_id or eval_spec.get("forecaster_set")
    if not forecaster_set_name:
        raise ValueError("forecaster set not specified")

    forecaster_set = registry.get_forecaster_set(forecaster_set_name)
    forecaster_ids = forecaster_set["members"]

    timestamp = run_id or datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    for forecaster_id in forecaster_ids:
        forecaster_spec = registry.get_forecaster(forecaster_id)
        output_dir = Path("outputs") / eval_id / timestamp / forecaster_id
        output_dir.mkdir(parents=True, exist_ok=True)
        recorder = LocalRecorder(str(output_dir / "events.jsonl"))

        try:
            forecaster = build_forecaster(forecaster_spec)
        except Exception as exc:  # noqa: BLE001
            result = {"status": "skipped", "reason": str(exc)}
            (output_dir / "results.json").write_text(json.dumps(result, indent=2))
            recorder.close()
            continue

        eval_args = eval_spec.get("args", {})
        eval_obj = eval_cls(forecaster=forecaster, **eval_args)

        if hasattr(eval_obj, "build_samples"):
            series = [row["value"] for row in dataset]
            covariate = [row.get("covariate") for row in dataset]
            samples = eval_obj.build_samples(series, covariate if eval_args.get("use_covariate") else None)
        else:
            samples = dataset

        results = eval_obj.run(samples, recorder=recorder)
        recorder.close()

        aggregated = aggregate_metrics(results)
        output = {
            "eval_id": eval_id,
            "forecaster_id": forecaster_id,
            "metrics": aggregated,
            "num_samples": len(results),
        }
        (output_dir / "results.json").write_text(json.dumps(output, indent=2))
        (output_dir / "results.md").write_text(render_markdown(output))
        plot_path = output_dir / "plot.png"
        plot_forecast(results, plot_path)


def aggregate_metrics(results: list) -> dict[str, float]:
    totals: dict[str, list[float]] = {}
    for result in results:
        for key, value in result.metrics.items():
            if isinstance(value, (int, float)):
                totals.setdefault(key, []).append(float(value))
    return {key: sum(values) / len(values) for key, values in totals.items() if values}


def render_markdown(output: dict[str, Any]) -> str:
    lines = [f"# {output['eval_id']} ({output['forecaster_id']})", "", "## Metrics"]
    for key, value in output["metrics"].items():
        lines.append(f"- **{key}**: {value:.4f}")
    return "\n".join(lines)


def plot_forecast(results: list, path: Path) -> None:
    for result in reversed(results):
        if "forecast" in result.metrics and "actual" in result.metrics:
            forecast = result.metrics["forecast"]
            actual = result.metrics["actual"]
            plt.figure(figsize=(6, 3))
            plt.plot(range(len(actual)), actual, label="actual")
            plt.plot(range(len(forecast)), forecast, label="forecast")
            plt.legend()
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a contextualized forecasting eval")
    parser.add_argument("eval_id")
    parser.add_argument("--forecasters", dest="forecasters", default=None)
    parser.add_argument("--run-id", dest="run_id", default=None)
    args = parser.parse_args()

    run_eval(args.eval_id, args.forecasters, args.run_id)


if __name__ == "__main__":
    main()
