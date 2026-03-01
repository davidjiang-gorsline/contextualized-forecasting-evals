from __future__ import annotations

import numpy as np


def mae(y_true: list[float], y_pred: list[float]) -> float:
    arr_true = np.asarray(y_true)
    arr_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(arr_true - arr_pred)))


def rmse(y_true: list[float], y_pred: list[float]) -> float:
    arr_true = np.asarray(y_true)
    arr_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((arr_true - arr_pred) ** 2)))


def smape(y_true: list[float], y_pred: list[float]) -> float:
    arr_true = np.asarray(y_true)
    arr_pred = np.asarray(y_pred)
    denom = (np.abs(arr_true) + np.abs(arr_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(np.abs(arr_true - arr_pred) / denom))


def mase(
    y_true: list[float],
    y_pred: list[float],
    insample: list[float],
    seasonal_period: int = 1,
) -> float:
    arr_true = np.asarray(y_true)
    arr_pred = np.asarray(y_pred)
    insample_arr = np.asarray(insample)
    seasonality = max(int(seasonal_period), 1)
    if len(insample_arr) <= seasonality:
        return float("nan")
    scale = np.mean(np.abs(insample_arr[seasonality:] - insample_arr[:-seasonality]))
    scale = scale if scale != 0 else 1.0
    return float(np.mean(np.abs(arr_true - arr_pred)) / scale)
