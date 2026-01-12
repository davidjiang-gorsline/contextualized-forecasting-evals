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


def mase(y_true: list[float], y_pred: list[float], insample: list[float]) -> float:
    arr_true = np.asarray(y_true)
    arr_pred = np.asarray(y_pred)
    insample_arr = np.asarray(insample)
    if len(insample_arr) < 2:
        return float("nan")
    scale = np.mean(np.abs(np.diff(insample_arr)))
    scale = scale if scale != 0 else 1.0
    return float(np.mean(np.abs(arr_true - arr_pred)) / scale)
