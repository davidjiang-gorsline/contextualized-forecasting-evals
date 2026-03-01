# contextualized-forecasting-evals (cfevals)

Registry-first evaluation harness for contextualized time-series forecasting with strict as-of evaluation.

## What this repo provides
- Uniform interfaces for contextualized models (LLMs, foundation time-series models, regressors).
- Scenario benchmarks (context + RCRPS) and time-series benchmarks (walk-forward metrics).
- Deterministic walk-forward backtesting with no leakage and versioned registry entries.
- Frequency-aware evaluation via hourly/daily/weekly/monthly synthetic benchmarks.
- Timestamped contextual events carried into model requests as-of each backtest cutoff.
- Forecast outputs can include optional natural-language `justification_text`.
- Context is native and mandatory in model requests for all forecasting runs.
- Contract violations fail loudly with explicit errors (missing context, invalid forecast shapes, non-finite values).

## Reliability guardrails
- Mandatory context: time-series and scenario evaluators fail if no context is available as-of each sample.
- Deterministic ordering: dataset points and context events are sorted before windowing/evaluation.
- Strict validation: forecast horizon/shape/numeric finiteness are validated before scoring.
- Explicit failures: registry/CLI/benchmark loading failures include benchmark/model IDs and actionable messages.
- FNSPID ingestion uses deterministic parquet-shard scanning (no open-ended streaming loop behavior).

## Quickstart (uv + Python 3.13)

```bash
uv venv --python 3.13
source .venv/bin/activate
uv sync --extra dev
```

## Run benchmarks

```bash
cfeval benchmark.synthetic.daily.v1 --model model.naive.last.v1
cfeval benchmark.cik.v1 --model model.naive.last.v1
cfeval benchmark.fnspid.news_volume.primary.v1 --model model.naive.last.v1
```

Notes:
- FNSPID benchmarks require network access on first run unless cached.
- Context is mandatory for all forecasting runs; loaders must provide timestamped context events.

Run the starter set:

```bash
cfevalset benchmark_set.starter.v1 --model model.naive.last.v1
cfevalset benchmark_set.multifreq.v1 --model model.naive.last.v1
cfevalset benchmark_set.hedge_fund.v1 --model model.naive.last.v1
```

## Outputs

Results are written under `outputs/<benchmark_id>/<run_id>/<model_id>/` with
`events.jsonl`, `results.json`, and `results.md`.
- `events.jsonl` includes `context_event_count` and (if provided by model) `justification_text`.

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
