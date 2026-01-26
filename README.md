# contextualized-forecasting-evals (cfevals)

Registry-first evaluation harness for contextualized time-series forecasting with strict as-of evaluation.

## What this repo provides
- Uniform interfaces for contextualized models (LLMs, foundation time-series models, regressors).
- Scenario benchmarks (context + RCRPS) and time-series benchmarks (walk-forward metrics).
- Deterministic walk-forward backtesting with no leakage and versioned registry entries.

## Quickstart (uv + Python 3.13)

```bash
uv venv --python 3.13
source .venv/bin/activate
uv sync --extra dev
```

## Run benchmarks

```bash
cfeval benchmark.cik.v1 --model model.naive.last.v1
cfeval benchmark.fred.unrate.v1 --model model.naive.last.v1
```

Run the starter set:

```bash
cfevalset benchmark_set.starter.v1 --model model.naive.last.v1
```

## Outputs

Results are written under `outputs/<benchmark_id>/<run_id>/<model_id>/` with
`events.jsonl`, `results.json`, and `results.md`.

## Optional model dependencies

```bash
uv sync --extra chronos
uv sync --extra openai
```

## Environment and caching

- `OPENAI_API_KEY` is required for `model.openai.gpt4o-mini.v1`.
- `CFEVALS_CACHE` (default `~/.cfevals/cache`) is used for dataset caching where supported.

## Add a benchmark or model (short version)

- Benchmarks: implement `TimeSeriesBenchmark` or `ScenarioBenchmark`, then add a
  versioned registry entry in `cfevals/registry/benchmarks/`.
- Models: implement `Model.predict` and add a versioned entry in `cfevals/registry/models/`.

## Development

```bash
uv run pytest
```
