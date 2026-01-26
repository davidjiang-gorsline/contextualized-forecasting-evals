from datetime import datetime, timedelta

from cfevals.benchmarks.base import TimeSeriesDataset, TimeSeriesPoint


def test_no_leak_lags():
    start = datetime(2020, 1, 1)
    points = [
        TimeSeriesPoint(timestamp=start + timedelta(days=i), value=float(i), features={"f": 100.0 + i})
        for i in range(20)
    ]
    dataset = TimeSeriesDataset(points=points)
    window = next(
        dataset.walk_forward_windows(
            horizon=2,
            step=1,
            min_train_size=5,
        )
    )
    assert window.history[-1] == 4.0
    assert window.history_features["f"][-1] == 104.0
    assert window.future[0] == 5.0
    assert window.future_features["f"][0] == 105.0
