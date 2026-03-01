import pytest

from cfevals.metrics.point import mae, mase, rmse


def test_point_metrics_reject_length_mismatch():
    with pytest.raises(ValueError, match="equal length"):
        mae([1.0, 2.0], [1.0])


def test_point_metrics_reject_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        rmse([1.0, float("nan")], [1.0, 2.0])


def test_mase_requires_insample_points():
    with pytest.raises(ValueError, match="at least two insample points"):
        mase([1.0], [1.0], insample=[1.0], seasonal_period=1)


def test_mase_falls_back_to_nonseasonal_scale_when_needed():
    value = mase([2.0], [1.0], insample=[0.0, 1.0], seasonal_period=2)
    assert value == pytest.approx(1.0)
