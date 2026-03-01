from cfevals.benchmarks.context_is_key import ContextIsKeyBenchmark


def test_cik_loader_smoke(monkeypatch):
    class DummyDataset:
        def __len__(self):
            return 1

        def __iter__(self):
            return iter(
                [
                    {
                        "sample_id": "cik-1",
                        "history": [1.0, 2.0],
                        "future": [3.0],
                        "context": "context",
                        "roi": [0.0, 10.0],
                    }
                ]
            )

    def fake_load_dataset(name, split, cache_dir=None, streaming=False):
        return DummyDataset()

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    benchmark = ContextIsKeyBenchmark(max_samples=1)
    samples = benchmark.load()
    assert samples[0].sample_id == "cik-1"


def test_cik_loader_parses_current_schema(monkeypatch):
    class DummyDataset:
        def __iter__(self):
            return iter(
                [
                    {
                        "name": "ExampleScenario",
                        "seed": 7,
                        "background": "Background text",
                        "scenario": "Scenario text",
                        "constraints": "",
                        "past_time": '{"0":{"2020-01-01T00:00:00.000":1.0,"2020-01-02T00:00:00.000":2.0}}',
                        "future_time": '{"0":{"2020-01-03T00:00:00.000":3.0,"2020-01-04T00:00:00.000":4.0}}',
                        "constraint_min": 0.0,
                        "constraint_max": 10.0,
                        "region_of_interest": [0, 1],
                    }
                ]
            )

    def fake_load_dataset(name, split, cache_dir=None, streaming=False):
        return DummyDataset()

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    benchmark = ContextIsKeyBenchmark(max_samples=1)
    samples = benchmark.load()
    assert len(samples) == 1
    assert samples[0].history == [1.0, 2.0]
    assert samples[0].future == [3.0, 4.0]
    assert samples[0].roi == (0.0, 10.0)
