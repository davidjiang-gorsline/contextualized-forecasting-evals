from cfevals.registry import Registry


def test_registry_loads():
    registry = Registry().load()
    assert "fred_unrate.test.v1" in registry.evals
    assert "dataset.fred.unrate.v1" in registry.datasets
    assert "forecaster.stats.naive.v1" in registry.forecasters
    assert "starter_set.v1" in registry.eval_sets
