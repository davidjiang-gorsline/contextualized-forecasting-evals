from cfevals.metrics.probabilistic import rcrps


def test_rcrps_smoke():
    samples = [1.0, 2.0, 3.0]
    score = rcrps(samples, target=2.0, roi=(0.0, 4.0))
    assert score >= 0.0
