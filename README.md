# cfevals

Minimal, registry-first evaluation harness for contextualized forecasting.

## Quickstart

```bash
pip install -e .
cfeval fred_unrate.test.v1 --forecasters forecaster_set.fred.v1
cfeval context_is_key.test.v1 --forecasters forecaster_set.cik.v1
cfevalset starter_set.v1
```

Outputs are written under `outputs/<eval_id>/<run_id>/<forecaster_id>/`.
