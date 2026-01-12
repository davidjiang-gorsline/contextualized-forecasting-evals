from __future__ import annotations

import numpy as np


def crps(samples: list[float], target: float) -> float:
    arr = np.asarray(samples, dtype=float)
    target_val = float(target)
    term1 = np.mean(np.abs(arr - target_val))
    term2 = 0.5 * np.mean(np.abs(arr[:, None] - arr[None, :]))
    return float(term1 - term2)


def rcrps(
    samples: list[float],
    target: float,
    roi: tuple[float, float] | None = None,
    penalty_weight: float = 1.0,
) -> float:
    # Minimal adaptation of the CiK (Context is Key) RCRPS definition.
    base = crps(samples, target)
    if roi is None:
        return base
    lower, upper = roi
    penalty = 0.0
    if target < lower:
        penalty = lower - target
    elif target > upper:
        penalty = target - upper
    return float(base + penalty_weight * penalty)
