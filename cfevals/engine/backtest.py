from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from cfevals.benchmarks.base import TimeSeriesDataset, WalkForwardWindow
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


class WalkForwardBacktester:
    def run(
        self,
        dataset: TimeSeriesDataset,
        model: Model,
        config: WalkForwardConfig,
        *,
        recorder: RecorderBase,
    ) -> list[BacktestResult]:
        results: list[BacktestResult] = []
        model.reset()
        trained_once = False

        for window in _windows(dataset, config):
            if _should_retrain(window, config, trained_once):
                train_request = _build_request(window, config.horizon)
                model.fit(train_request)
                trained_once = True

            request = _build_request(window, config.horizon)
            forecast_result = model.predict(request)
            validate_forecast_result(
                forecast_result,
                config.horizon,
                context=f"backtest window {window.window_index}",
            )
            metrics = _compute_metrics(window, forecast_result)
            sample_id = f"{window.window_index:05d}-{window.as_of.date()}"
            result = BacktestResult(
                sample_id=sample_id,
                as_of=window.as_of.isoformat(),
                forecast=forecast_result.point_forecast,
                actual=window.future,
                metrics=metrics,
            )
            recorder.record_event(
                "walk_forward_window",
                {
                    "sample_id": sample_id,
                    "as_of": result.as_of,
                    "forecast": result.forecast,
                    "actual": result.actual,
                    "metrics": result.metrics,
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


def _build_request(window: WalkForwardWindow, horizon: int) -> ForecastRequest:
    return ForecastRequest(
        history=window.history,
        horizon=horizon,
        timestamps=window.history_timestamps,
        features=window.history_features,
    )


def _compute_metrics(window: WalkForwardWindow, result: ForecastResult) -> dict[str, float]:
    metrics = {
        "mae": mae(window.future, result.point_forecast),
        "rmse": rmse(window.future, result.point_forecast),
        "smape": smape(window.future, result.point_forecast),
        "mase": mase(window.future, result.point_forecast, window.history),
    }
    return metrics
