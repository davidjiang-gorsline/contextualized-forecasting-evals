from __future__ import annotations

import argparse
from pathlib import Path

from cfevals.cli.cfeval import build_backtest_config, build_benchmark, build_model
from cfevals.engine import Runner
from cfevals.engine.runner import default_run_id
from cfevals.registry import Registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a benchmark set")
    parser.add_argument("benchmark_set_id")
    parser.add_argument("--model", dest="model_id", required=True)
    parser.add_argument("--run-id", dest="run_id", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    registry = Registry().load()
    benchmark_set = registry.get_benchmark_set(args.benchmark_set_id)
    model_spec = registry.get_model(args.model_id)
    model = build_model(model_spec)

    run_id = args.run_id or default_run_id()

    for benchmark_id in benchmark_set["benchmarks"]:
        output_dir = Path("outputs") / benchmark_id / run_id / args.model_id
        if args.resume and output_dir.exists():
            continue
        benchmark_spec = registry.get_benchmark(benchmark_id)
        benchmark = build_benchmark(benchmark_spec)
        backtest_config = build_backtest_config(benchmark_spec, {})
        Runner().run(
            benchmark_id=benchmark_id,
            benchmark=benchmark,
            model_id=args.model_id,
            model=model,
            output_dir=output_dir,
            backtest_config=backtest_config,
        )


if __name__ == "__main__":
    main()
