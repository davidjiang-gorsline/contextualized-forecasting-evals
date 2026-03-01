from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from cfevals.benchmarks.base import TimeSeriesDataset, WalkForwardWindow
from cfevals.context import ContextEvent
from cfevals.engine.validation import validate_forecast_result
from cfevals.metrics.point import mae, mase, rmse, smape
from cfevals.models.base import ForecastRequest, ForecastResult, Model
from cfevals.record import RecorderBase


@dataclass(frozen=True)
class WalkForwardConfig:
    horizon: int
    step: int = 1
    min_train_size: int = 24
    max_train_size: int | None = None
    allow_retrain: bool = True
    retrain_frequency: int = 1
    max_windows: int | None = None


@dataclass(frozen=True)
class BacktestResult:
    sample_id: str
    as_of: str
    forecast: list[float]
    actual: list[float]
    metrics: dict[str, float]
    justification_text: str | None = None


class WalkForwardBacktester:
    def run(
        self,
        dataset: TimeSeriesDataset,
        model: Model,
        config: WalkForwardConfig,
        *,
        recorder: RecorderBase,
    ) -> list[BacktestResult]:
        _validate_config(config)
        results: list[BacktestResult] = []
        _validate_dataset_context(dataset)
        model.reset()
        trained_once = False
        seasonal_period = _seasonal_period(dataset)

        for window in _windows(dataset, config):
            request = _build_request(window, config.horizon)
            if _should_retrain(window, config, trained_once):
                model.fit(request)
                trained_once = True

            forecast_result = model.predict(request)
            validate_forecast_result(
                forecast_result,
                config.horizon,
                context=f"backtest window {window.window_index}",
            )
            metrics = _compute_metrics(window, forecast_result, seasonal_period=seasonal_period)
            sample_id = f"{window.window_index:05d}-{window.as_of.date()}"
            result = BacktestResult(
                sample_id=sample_id,
                as_of=window.as_of.isoformat(),
                forecast=forecast_result.point_forecast,
                actual=window.future,
                metrics=metrics,
                justification_text=forecast_result.justification_text,
            )
            recorder.record_event(
                "walk_forward_window",
                {
                    "sample_id": sample_id,
                    "as_of": result.as_of,
                    "forecast": result.forecast,
                    "actual": result.actual,
                    "metrics": result.metrics,
                    "context_event_count": len(request.context_events or []),
                    "justification_text": result.justification_text,
                },
                sample_id=sample_id,
            )
            results.append(result)
        return results


def _windows(dataset: TimeSeriesDataset, config: WalkForwardConfig) -> Iterable[WalkForwardWindow]:
    return dataset.walk_forward_windows(
        horizon=config.horizon,
        step=config.step,
        min_train_size=config.min_train_size,
        max_train_size=config.max_train_size,
        max_windows=config.max_windows,
    )


def _should_retrain(window: WalkForwardWindow, config: WalkForwardConfig, trained_once: bool) -> bool:
    if not config.allow_retrain:
        return not trained_once
    return window.window_index % max(config.retrain_frequency, 1) == 0


def _build_request(
    window: WalkForwardWindow,
    horizon: int,
) -> ForecastRequest:
    if not window.context_events:
        raise ValueError(
            f"backtest window {window.window_index}: context_events is empty at as_of={window.as_of.isoformat()}"
        )
    context_text = _context_summary(window.context_events)
    return ForecastRequest(
        history=window.history,
        horizon=horizon,
        timestamps=window.history_timestamps,
        features=window.history_features,
        context_text=context_text,
        context_events=window.context_events,
    )


def _compute_metrics(
    window: WalkForwardWindow,
    result: ForecastResult,
    *,
    seasonal_period: int,
) -> dict[str, float]:
    metrics = {
        "mae": mae(window.future, result.point_forecast),
        "rmse": rmse(window.future, result.point_forecast),
        "smape": smape(window.future, result.point_forecast),
        "mase": mase(
            window.future,
            result.point_forecast,
            window.history,
            seasonal_period=seasonal_period,
        ),
    }
    return metrics


def _validate_config(config: WalkForwardConfig) -> None:
    if config.horizon <= 0:
        raise ValueError("horizon must be greater than zero")
    if config.step <= 0:
        raise ValueError("step must be greater than zero")
    if config.min_train_size <= 0:
        raise ValueError("min_train_size must be greater than zero")
    if config.max_train_size is not None and config.max_train_size <= 0:
        raise ValueError("max_train_size must be greater than zero when provided")
    if config.max_train_size is not None and config.max_train_size < config.min_train_size:
        raise ValueError("max_train_size must be >= min_train_size")
    if config.retrain_frequency <= 0:
        raise ValueError("retrain_frequency must be greater than zero")
    if config.max_windows is not None and config.max_windows <= 0:
        raise ValueError("max_windows must be greater than zero when provided")


def _seasonal_period(dataset: TimeSeriesDataset) -> int:
    if dataset.metadata and dataset.metadata.get("seasonal_period") is not None:
        candidate = dataset.metadata.get("seasonal_period")
        if isinstance(candidate, int) and candidate > 0:
            return candidate
    if not dataset.frequency:
        return 1
    by_freq = {
        "H": 24,
        "D": 7,
        "B": 5,
        "W": 52,
        "M": 12,
        "Q": 4,
        "Y": 1,
        "A": 1,
    }
    return by_freq.get(dataset.frequency.upper()[0], 1)


def _validate_dataset_context(dataset: TimeSeriesDataset) -> None:
    if not dataset.context_events:
        raise ValueError(
            "TimeSeriesDataset.context_events is empty. Context is mandatory in this contextualized harness."
        )


def _context_summary(events: list[ContextEvent]) -> str:
    first = events[0]
    last = events[-1]
    return (
        "Timestamped context is attached via context_events. "
        f"events={len(events)}, first_available_at={first.as_of_time.isoformat()}, "
        f"last_available_at={last.as_of_time.isoformat()}."
    )


# TODO(P0): Add purged/embargoed walk-forward variants and explicit as-of validation
# hooks for feature/context availability to prove zero leakage in event-driven datasets.
# TODO(P1): Add configurable context retrieval policies (recency, source filters, and
# relevance scoring) before model requests are built.
# TODO(P2): Add vectorized window materialization for large panel datasets to reduce
# Python-loop overhead at scale.
