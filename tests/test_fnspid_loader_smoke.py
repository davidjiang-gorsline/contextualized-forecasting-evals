from cfevals.benchmarks.fnspid import FNSPIDNewsVolumeBenchmark


def test_fnspid_loader_smoke(monkeypatch):
    rows = [
        {
            "Date": "2024-01-01 10:00:00 UTC",
            "Stock_symbol": "AAPL",
            "Article_title": "Apple demand rises",
            "Lsa_summary": "Strong iPhone cycle",
            "Publisher": "Reuters",
        },
        {
            "Date": "2024-01-01 14:00:00 UTC",
            "Stock_symbol": "AAPL",
            "Article_title": "Analyst upgrades Apple",
            "Lsa_summary": "Target price raised",
            "Publisher": "Bloomberg",
        },
        {
            "Date": "2024-01-02 09:00:00 UTC",
            "Stock_symbol": "AAPL",
            "Article_title": "Supply chain improves",
            "Lsa_summary": "Lead times normalize",
            "Publisher": "CNBC",
        },
    ]

    monkeypatch.setattr(
        "cfevals.benchmarks.fnspid._list_parquet_files",
        lambda dataset_name, split: ["data/train-00000-of-00001.parquet"],
    )
    monkeypatch.setattr(
        "cfevals.benchmarks.fnspid._download_parquet_file",
        lambda dataset_name, path, cache_dir: "/tmp/fake.parquet",
    )
    monkeypatch.setattr(
        "cfevals.benchmarks.fnspid._iter_parquet_rows",
        lambda path: iter(rows),
    )
    benchmark = FNSPIDNewsVolumeBenchmark(
        symbol="AAPL",
        max_rows=100,
        min_points=2,
        streaming=True,
    )
    dataset = benchmark.load()
    assert dataset.frequency == "D"
    assert len(dataset.points) == 2
    assert dataset.points[0].value == 2.0
    assert dataset.points[1].value == 1.0
    assert len(dataset.context_events) == 3
