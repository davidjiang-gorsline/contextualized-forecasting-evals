# Codex Development Guide

## Mission and Scope
- Build a registry-first evaluation harness for contextualized time-series forecasting (OpenAI eval style).
- Provide a uniform interface for contextualized models (LLMs, foundation time-series models, regressors) and benchmarks.
- Enforce strict as-of evaluation: walk-forward backtesting, no leakage, deterministic ordering.
- Support both scenario benchmarks (context text + RCRPS) and time-series benchmarks (walk-forward metrics).
- Roadmap: expand to MoTime, FNSPID, TGTSF, MoAT, Time-MMD, TimeText, News-Signal Library; add time-series CV utilities, detrend/deseasonalize helpers, event forecasting, and more model adapters (TimeGPT, Chronos variants, ARIMA/Moirai baselines).

## Current Capabilities
- Benchmarks: `benchmark.cik.v1` (Context Is Key), `benchmark.fred.unrate.v1` (FRED UNRATE walk-forward).
- Models: `model.naive.last.v1`, `model.chronos.t5.small.v1`, `model.openai.gpt4o-mini.v1` (optional extras).
- CLI: `cfeval` and `cfevalset`.
- Outputs: `outputs/<benchmark_id>/<run_id>/<model_id>/{events.jsonl,results.json,results.md}`.

## Architecture Map
- `cfevals/benchmarks/`: Benchmark loaders that return `TimeSeriesDataset` or `ScenarioSample`.
- `cfevals/models/`: Model adapters implementing `Model.predict` and `ForecastResult`.
- `cfevals/engine/`: `WalkForwardBacktester`, `ScenarioEvaluator`, and `Runner`.
- `cfevals/metrics/`: Point and probabilistic metrics (MAE/RMSE/SMAPE/MASE, RCRPS).
- `cfevals/registry/`: YAML registry for benchmarks, models, and sets.
- `cfevals/cli/`: CLI entry points.
- `tests/`: Fast, deterministic tests; no network access.

## Core Contracts (Do Not Break)
- `ForecastRequest` is the only model input. Use `history`, `horizon`, optional `timestamps`, `features`, `context_text`, `metadata`.
- `ForecastResult.point_forecast` length must equal `horizon`. If present, `samples` or `quantiles` must align to `horizon`.
- `TimeSeriesDataset.walk_forward_windows` defines the no-leakage boundary; history and features must be as-of the window cutoff.
- Scenario benchmarks use `ScenarioSample` and are scored with RCRPS; time-series benchmarks use walk-forward metrics.
- Treat `cfevals/benchmarks/base.py` and `cfevals/models/base.py` as stable contracts.

## Evaluation Invariants
- Determinism: stable ordering, no hidden randomness, no implicit shuffling.
- No leakage: never pass future values or features into `history`.
- Backtests respect `WalkForwardConfig` and retraining policy; use `fit` only for as-of training.

## Benchmark and Model Extension Rules
- Benchmarks must be self-contained and deterministic; download data from Hugging Face or source APIs with local caching.
- Use `CFEVALS_CACHE` (default `~/.cfevals/cache`) or the upstream cache; avoid global side effects.
- Model adapters should be thin, deterministic, and guard optional deps with clear import/runtime errors.
- Add utilities (detrending, deseasonalizing, baselines) as reusable helpers, not benchmark-specific hacks.

## Registry Workflow
- Registry is the source of truth. IDs must be descriptive and versioned (`benchmark.*.v1`, `model.*.v1`).
- Do not edit existing versioned entries for behavior changes; add a new version instead.
- Keep benchmark configs in YAML, not code.
- Registry paths are loaded from `cfevals/registry/` and `~/.cfevals` (local overrides).

## Dependencies and Testing
- Environment is managed by `uv` with Python 3.13; heavy deps stay behind optional extras.
- Run tests with `uv sync --extra dev` and `uv run pytest`.
- Tests must be fast, deterministic, and offline (use monkeypatching and caches).
