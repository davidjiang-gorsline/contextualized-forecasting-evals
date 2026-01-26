# Codex Development Guide

## Project Summary
- This repository is an OpenAI Eval style registry-first evaluation harness for contextualized time-series forecasting.
- Core goals: strict walk-forward backtesting, no leakage, modular benchmarks/models, minimal dependencies, and extensibility.
- Outputs are written under `outputs/<benchmark_id>/<run_id>/<model_id>/` with `events.jsonl`, `results.json`, and `results.md`.

## Architecture Map
- `cfevals/benchmarks/`: Benchmark definitions that load datasets into `TimeSeriesDataset` or `ScenarioSample`.
- `cfevals/models/`: Model adapters implementing `Model.predict` with `ForecastRequest`/`ForecastResult`.
- `cfevals/engine/`: Evaluation engines (`WalkForwardBacktester`, `ScenarioEvaluator`) and `Runner`.
- `cfevals/registry/`: YAML registry for benchmarks, models, and benchmark sets.
- `cfevals/metrics/`: Point and probabilistic metric implementations.
- `cfevals/cli/`: CLI entry points (`cfeval`, `cfevalset`).
- `tests/`: Fast, deterministic tests; no network access.

## Core Contracts and Invariants
- `ForecastRequest` is the single interface into models. Use `history`, `horizon`, and optional `timestamps`, `features`, `context_text`, `metadata`.
- `ForecastResult.point_forecast` must match `horizon` length. `samples` or `quantiles` are optional for probabilistic scoring.
- `TimeSeriesDataset.walk_forward_windows` defines the no-leakage boundary; history/feature values must be as-of the window cutoff.
- Scenario benchmarks use `ScenarioSample` and are scored with RCRPS; time-series benchmarks use walk-forward metrics.
- Registry entries are versioned and must be kept stable for reproducibility (`benchmark.*.v1`, `model.*.v1`).

## Code-As-Documentation Standard
- Prefer clear types, names, and small functions over comments.
- When comments or docstrings are needed, document why (rationale, invariants, tradeoffs, safety) rather than how.
- Keep public interfaces simple and stable; treat `cfevals/benchmarks/base.py` and `cfevals/models/base.py` as contracts.

## Efficiency and Robustness Requirements
- Determinism matters: keep evaluation ordering stable and avoid hidden randomness.
- Guard optional dependencies (Chronos/OpenAI) with explicit import/runtime errors and minimal side effects.
- Avoid heavy work in tests; no network calls in tests; use caching or monkeypatching.
- Validate shape assumptions early (horizon alignment, sample lengths) and fail fast with clear errors.
- Prefer streaming or incremental processing for large datasets; avoid unnecessary copies.

## Extensibility Guidelines
- New benchmarks: implement `TimeSeriesBenchmark` or `ScenarioBenchmark`, then add a registry YAML entry.
- New models: implement `Model.predict` and honor `ForecastRequest`; keep adapters thin and deterministic.
- Keep the harness modular: avoid hard-coding benchmark/model IDs in code paths.
- Favor adding new benchmark versions over editing existing ones to preserve reproducibility.

## Dependencies and Environment
- Environment is managed by `uv` with Python 3.13.
- Keep base dependencies minimal; add new heavy deps only as optional extras in `pyproject.toml`.
- Prefer standard library and existing deps (`numpy`, `pandas`, `datasets`, `pyyaml`) before adding new ones.

## Testing
- Run tests with `uv sync --extra dev` and `uv run pytest`.
- Add tests for new functionality, especially around leakage, determinism, and registry wiring.

## Registry Workflow
- Registry paths are loaded from `cfevals/registry/` and `~/.cfevals`.
- Keep IDs descriptive and versioned; document benchmark config in YAML, not code.
