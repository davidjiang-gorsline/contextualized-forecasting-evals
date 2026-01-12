from cfevals.data import ContextIsKeyLoader


def test_cik_loader_smoke(monkeypatch):
    class DummyDataset:
        def __len__(self):
            return 1

        def __iter__(self):
            return iter([
                {
                    "sample_id": "cik-1",
                    "history": [1.0, 2.0],
                    "future": [3.0],
                    "context": "context",
                    "roi": [0.0, 10.0],
                }
            ])

    def fake_load_dataset(name, split):
        return DummyDataset()

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    loader = ContextIsKeyLoader(max_samples=1)
    samples = loader.load()
    assert samples[0]["sample_id"] == "cik-1"
