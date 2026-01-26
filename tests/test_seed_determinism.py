from datetime import datetime, timedelta

from cfevals.benchmarks.base import TimeSeriesDataset, TimeSeriesPoint
from cfevals.engine.backtest import WalkForwardBacktester, WalkForwardConfig
from cfevals.models.naive import LastValueModel
from cfevals.record import NullRecorder


def test_seed_determinism():
    start = datetime(2023, 1, 1)
    points = [TimeSeriesPoint(timestamp=start + timedelta(days=i), value=float(i)) for i in range(15)]
    dataset = TimeSeriesDataset(points=points)
    config = WalkForwardConfig(horizon=2, min_train_size=5, step=1)

    results_first = WalkForwardBacktester().run(dataset, LastValueModel(), config, recorder=NullRecorder())
    results_second = WalkForwardBacktester().run(dataset, LastValueModel(), config, recorder=NullRecorder())
    order_first = [res.sample_id for res in results_first]
    order_second = [res.sample_id for res in results_second]
    assert order_first == order_second
