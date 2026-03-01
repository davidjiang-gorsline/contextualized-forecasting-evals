from __future__ import annotations

import numpy as np


def mae(y_true: list[float], y_pred: list[float]) -> float:
    arr_true, arr_pred = _validate_pair(y_true, y_pred, metric_name="MAE")
    value = float(np.mean(np.abs(arr_true - arr_pred)))
    _ensure_finite(value, metric_name="MAE")
    return value


def rmse(y_true: list[float], y_pred: list[float]) -> float:
    arr_true, arr_pred = _validate_pair(y_true, y_pred, metric_name="RMSE")
    value = float(np.sqrt(np.mean((arr_true - arr_pred) ** 2)))
    _ensure_finite(value, metric_name="RMSE")
    return value


def smape(y_true: list[float], y_pred: list[float]) -> float:
    arr_true, arr_pred = _validate_pair(y_true, y_pred, metric_name="SMAPE")
    denom = (np.abs(arr_true) + np.abs(arr_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    value = float(np.mean(np.abs(arr_true - arr_pred) / denom))
    _ensure_finite(value, metric_name="SMAPE")
    return value


def mase(
    y_true: list[float],
    y_pred: list[float],
    insample: list[float],
    seasonal_period: int = 1,
) -> float:
    arr_true, arr_pred = _validate_pair(y_true, y_pred, metric_name="MASE")
    insample_arr = _as_float_vector(insample, name="insample")
    seasonality = max(int(seasonal_period), 1)
    if len(insample_arr) <= 1:
        raise ValueError("MASE requires at least two insample points")
    if len(insample_arr) <= seasonality:
        seasonality = 1
    scale = np.mean(np.abs(insample_arr[seasonality:] - insample_arr[:-seasonality]))
    scale = scale if scale != 0 else 1.0
    value = float(np.mean(np.abs(arr_true - arr_pred)) / scale)
    _ensure_finite(value, metric_name="MASE")
    return value


def _validate_pair(
    y_true: list[float],
    y_pred: list[float],
    *,
    metric_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    arr_true = _as_float_vector(y_true, name="y_true")
    arr_pred = _as_float_vector(y_pred, name="y_pred")
    if arr_true.shape != arr_pred.shape:
        raise ValueError(
            f"{metric_name} requires y_true and y_pred to have equal length, "
            f"got {arr_true.size} and {arr_pred.size}"
        )
    return arr_true, arr_pred


def _as_float_vector(values: list[float], *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D sequence")
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite numeric values")
    return arr


def _ensure_finite(value: float, *, metric_name: str) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{metric_name} produced a non-finite value: {value}")
