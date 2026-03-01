from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cfevals.engine import Runner, WalkForwardConfig
from cfevals.registry import Registry


def load_class(path: str):
    if ":" not in path:
        raise ValueError(
            f"Invalid registry class path {path!r}. Expected format 'package.module:ClassName'."
        )
    module_name, class_name = path.split(":", maxsplit=1)
    try:
        module = __import__(module_name, fromlist=[class_name])
    except Exception as exc:  # noqa: BLE001
        raise ImportError(f"Failed to import module {module_name!r} for class path {path!r}") from exc
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Class {class_name!r} not found in module {module_name!r} for registry path {path!r}"
        ) from exc


def build_benchmark(spec: dict[str, Any]):
    benchmark_cls = load_class(spec["class"])
    return benchmark_cls(**spec.get("args", {}))


def build_model(spec: dict[str, Any]):
    model_cls = load_class(spec["class"])
    return model_cls(**spec.get("args", {}))


def build_backtest_config(spec: dict[str, Any], overrides: dict[str, Any]) -> WalkForwardConfig | None:
    if spec.get("kind") == "scenario":
        return None
    base = spec.get("backtest", {})
    payload = {**base, **{k: v for k, v in overrides.items() if v is not None}}
    if not payload:
        return WalkForwardConfig(horizon=1)
    return WalkForwardConfig(**payload)


def run_eval(args: argparse.Namespace) -> None:
    registry = Registry().load()
    benchmark_spec = registry.get_benchmark(args.benchmark_id)
    model_spec = registry.get_model(args.model_id)

    benchmark = build_benchmark(benchmark_spec)
    model = build_model(model_spec)

    overrides = {
        "horizon": args.horizon,
        "step": args.step,
        "min_train_size": args.min_train_size,
        "max_train_size": args.max_train_size,
        "allow_retrain": args.allow_retrain,
        "retrain_frequency": args.retrain_frequency,
        "max_windows": args.max_windows,
    }
    backtest_config = build_backtest_config(benchmark_spec, overrides)

    output_dir = Path("outputs") / args.benchmark_id / args.run_id / args.model_id
    try:
        Runner().run(
            benchmark_id=args.benchmark_id,
            benchmark=benchmark,
            model_id=args.model_id,
            model=model,
            output_dir=output_dir,
            backtest_config=backtest_config,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Evaluation failed for benchmark={args.benchmark_id!r} model={args.model_id!r}: {exc}"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a time-series benchmark")
    parser.add_argument("benchmark_id")
    parser.add_argument("--model", dest="model_id", required=True)
    parser.add_argument("--run-id", dest="run_id", default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--min-train-size", type=int, default=None)
    parser.add_argument("--max-train-size", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--allow-retrain", action="store_true")
    parser.add_argument("--no-retrain", dest="allow_retrain", action="store_false")
    parser.set_defaults(allow_retrain=None)
    parser.add_argument("--retrain-frequency", type=int, default=None)
    args = parser.parse_args()

    if args.run_id is None:
        from cfevals.engine.runner import default_run_id

        args.run_id = default_run_id()

    run_eval(args)


if __name__ == "__main__":
    main()
