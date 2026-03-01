from cfevals.registry import Registry


def test_registry_loads():
    registry = Registry().load()
    assert "benchmark.cik.v1" in registry.benchmarks
    assert "benchmark.synthetic.daily.v1" in registry.benchmarks
    assert "benchmark.synthetic.hourly.v1" in registry.benchmarks
    assert "benchmark.synthetic.weekly.v1" in registry.benchmarks
    assert "benchmark.synthetic.monthly.v1" in registry.benchmarks
    assert "model.naive.last.v1" in registry.models
    assert "benchmark_set.starter.v1" in registry.benchmark_sets
    assert "benchmark_set.multifreq.v1" in registry.benchmark_sets
