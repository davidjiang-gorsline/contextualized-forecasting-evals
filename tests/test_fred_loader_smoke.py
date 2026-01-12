import json
import os

from cfevals.data import FredUnrateLoader, default_cache_dir


def test_fred_loader_smoke(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("CFEVALS_CACHE", str(cache_dir))
    payload = {
        "index": ["2020-01-01", "2020-02-01"],
        "target": [3.5, 3.6],
        "covariate": [100.0, 101.0],
    }
    cache_path = cache_dir / "fred_UNRATE.json"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload))
    loader = FredUnrateLoader(target_series="UNRATE", covariate_series="USEPUNEWSINDXM")
    samples = loader.load()
    assert samples[0]["value"] == 3.5
    assert samples[0]["covariate"] == 100.0
