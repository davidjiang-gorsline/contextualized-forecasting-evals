from datetime import datetime, timedelta

from cfevals.benchmarks.base import TimeSeriesDataset, TimeSeriesPoint
from cfevals.context import ContextEvent
from cfevals.engine.backtest import WalkForwardBacktester, WalkForwardConfig
from cfevals.models.base import ForecastRequest, ForecastResult, Model
from cfevals.record import NullRecorder


class CaptureRequestModel(Model):
    def __init__(self) -> None:
        self.requests: list[ForecastRequest] = []

    def predict(self, request: ForecastRequest) -> ForecastResult:
        self.requests.append(request)
        value = float(request.history[-1]) if request.history else 0.0
        return ForecastResult(point_forecast=[value] * request.horizon)


def _dataset_with_context() -> TimeSeriesDataset:
    start = datetime(2024, 1, 1)
    points = [TimeSeriesPoint(timestamp=start + timedelta(days=i), value=float(i)) for i in range(8)]
    context_events = [
        ContextEvent(
            event_time=start + timedelta(days=1),
            available_at=start + timedelta(days=1),
            text="visible-early",
        ),
        ContextEvent(
            event_time=start + timedelta(days=2),
            available_at=start + timedelta(days=4),
            text="not-yet-available",
        ),
    ]
    return TimeSeriesDataset(points=points, context_events=context_events, frequency="D")


def test_backtest_context_is_as_of_filtered():
    model = CaptureRequestModel()
    config = WalkForwardConfig(horizon=1, min_train_size=3, step=1, max_windows=1)
    WalkForwardBacktester().run(_dataset_with_context(), model, config, recorder=NullRecorder())
    request = model.requests[0]
    assert request.context_events is not None
    assert len(request.context_events) == 1
    assert request.context_events[0].text == "visible-early"
    assert request.context_text is not None
    assert "events=1" in request.context_text
