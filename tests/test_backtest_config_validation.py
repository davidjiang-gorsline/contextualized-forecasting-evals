from datetime import datetime, timedelta

import pytest

from cfevals.benchmarks.base import TimeSeriesDataset, TimeSeriesPoint
from cfevals.engine.backtest import WalkForwardBacktester, WalkForwardConfig
from cfevals.models.naive import LastValueModel
from cfevals.record import NullRecorder


def _dataset() -> TimeSeriesDataset:
    start = datetime(2024, 1, 1)
    points = [TimeSeriesPoint(timestamp=start + timedelta(days=i), value=float(i)) for i in range(20)]
    return TimeSeriesDataset(points=points, frequency="D")


def test_backtest_rejects_invalid_step():
    with pytest.raises(ValueError, match="step"):
        WalkForwardBacktester().run(
            _dataset(),
            LastValueModel(),
            WalkForwardConfig(horizon=2, step=0, min_train_size=5),
            recorder=NullRecorder(),
        )
