import pytest

from cfevals.registry import Registry


def test_registry_unknown_ids_raise_helpful_errors():
    registry = Registry().load()
    with pytest.raises(KeyError, match="Unknown benchmark id"):
        registry.get_benchmark("benchmark.does.not.exist")
    with pytest.raises(KeyError, match="Unknown model id"):
        registry.get_model("model.does.not.exist")
    with pytest.raises(KeyError, match="Unknown benchmark set id"):
        registry.get_benchmark_set("benchmark_set.does.not.exist")
