from cfevals.benchmarks.synthetic import SyntheticSeasonalBenchmark


def test_synthetic_loader_smoke():
    benchmark = SyntheticSeasonalBenchmark(
        frequency="H",
        periods=32,
        start="2024-01-01",
    )
    dataset = benchmark.load()
    assert len(dataset.points) == 32
    assert dataset.frequency == "H"
    assert dataset.metadata["seasonal_period"] == 24
    assert dataset.metadata["context_event_count"] > 0
    assert len(dataset.context_events) > 0
    assert "known_covariate" in dataset.points[0].features
