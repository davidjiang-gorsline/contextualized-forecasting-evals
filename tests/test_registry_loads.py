from cfevals.registry import Registry


def test_registry_loads():
    registry = Registry().load()
    assert "benchmark.fred.unrate.v1" in registry.benchmarks
    assert "benchmark.cik.v1" in registry.benchmarks
    assert "model.naive.last.v1" in registry.models
    assert "benchmark_set.starter.v1" in registry.benchmark_sets
