# cfevals

Registry-first evaluation harness for time-series benchmarks with strict walk-forward backtesting.

## Quickstart (uv + Python 3.13)

```bash
uv venv --python 3.13
source .venv/bin/activate
uv sync
```

## Run CiK (Context Is Key)

```bash
cfeval benchmark.cik.v1 --model model.naive.last.v1
```

## Run FRED (walk-forward backtest)

```bash
cfeval benchmark.fred.unrate.v1 --model model.naive.last.v1
```

Outputs are written under `outputs/<benchmark_id>/<run_id>/<model_id>/`.

## Optional model dependencies

To enable Chronos or OpenAI adapters, install the optional extras:

```bash
uv sync --extra chronos
uv sync --extra openai
```
