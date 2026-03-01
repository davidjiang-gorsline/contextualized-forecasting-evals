from datetime import datetime, timedelta

import pytest

from cfevals.benchmarks.base import ScenarioSample, TimeSeriesDataset, TimeSeriesPoint
from cfevals.engine.backtest import WalkForwardBacktester, WalkForwardConfig
from cfevals.engine.scenario import ScenarioEvaluator
from cfevals.models.base import ForecastRequest, ForecastResult, Model
from cfevals.record import NullRecorder


class BadLengthModel(Model):
    def predict(self, request: ForecastRequest) -> ForecastResult:
        return ForecastResult(point_forecast=[0.0])


class BadQuantileModel(Model):
    def predict(self, request: ForecastRequest) -> ForecastResult:
        return ForecastResult(point_forecast=[0.0] * request.horizon, quantiles={"0.5": [0.0]})


class GoodPointModel(Model):
    def predict(self, request: ForecastRequest) -> ForecastResult:
        return ForecastResult(point_forecast=[0.0] * request.horizon)


def test_backtest_validates_forecast_length():
    start = datetime(2024, 1, 1)
    points = [TimeSeriesPoint(timestamp=start + timedelta(days=i), value=float(i)) for i in range(8)]
    dataset = TimeSeriesDataset(points=points)
    config = WalkForwardConfig(horizon=2, min_train_size=4, step=1, max_windows=1)
    with pytest.raises(ValueError, match="point_forecast length"):
        WalkForwardBacktester().run(dataset, BadLengthModel(), config, recorder=NullRecorder())


def test_scenario_validates_quantiles():
    samples = [
        ScenarioSample(sample_id="s1", history=[1.0, 2.0], future=[3.0, 4.0], context_text="ctx"),
    ]
    with pytest.raises(ValueError, match="quantile"):
        ScenarioEvaluator().run(samples, BadQuantileModel(), recorder=NullRecorder())


def test_scenario_skips_empty_future():
    samples = [
        ScenarioSample(sample_id="empty", history=[1.0], future=[], context_text="ctx"),
        ScenarioSample(sample_id="ok", history=[1.0], future=[2.0], context_text="ctx"),
    ]
    results = ScenarioEvaluator().run(samples, GoodPointModel(), recorder=NullRecorder())
    assert len(results) == 1
    assert results[0].sample_id == "ok"
