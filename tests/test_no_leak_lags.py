from cfevals.elsuite.fred_unrate import FredUnrateEval


def test_no_leak_lags():
    series = list(range(200))
    covariate = [1000 + i for i in range(200)]
    eval_obj = FredUnrateEval(forecaster=lambda _: None, horizons=[1], train_window=10, use_covariate=True)
    samples = eval_obj.build_samples(series, covariate)
    sample = samples[0]
    history_start = series[6:16]
    assert sample["history"] == history_start
    for lag in range(1, 7):
        cov = sample["covariates"][f"epu_lag{lag}"]
        expected = covariate[16 - 10 - lag : 16 - lag]
        assert cov == expected
