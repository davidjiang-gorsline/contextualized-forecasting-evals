from __future__ import annotations

import math
from typing import Iterable

from cfevals.models.base import ForecastResult


def validate_forecast_result(result: ForecastResult, horizon: int, *, context: str) -> None:
    if horizon <= 0:
        raise ValueError(f"{context}: horizon must be > 0")
    if len(result.point_forecast) != horizon:
        raise ValueError(
            f"{context}: point_forecast length {len(result.point_forecast)} does not match horizon {horizon}"
        )
    _validate_numeric_sequence(result.point_forecast, context=f"{context}: point_forecast")
    if result.samples is not None:
        _validate_samples(result.samples, horizon, context=context)
    if result.quantiles is not None:
        _validate_quantiles(result.quantiles, horizon, context=context)


def normalize_samples(
    result: ForecastResult,
    horizon: int,
    *,
    context: str,
) -> list[list[float]] | None:
    if result.samples is None:
        return None
    samples = result.samples
    if not isinstance(samples, list):
        raise TypeError(f"{context}: samples must be a list")
    if not samples:
        raise ValueError(f"{context}: samples is empty")
    if isinstance(samples[0], list):
        for idx, sample in enumerate(samples):
            if len(sample) != horizon:
                raise ValueError(
                    f"{context}: sample[{idx}] length {len(sample)} does not match horizon {horizon}"
                )
            _validate_numeric_sequence(sample, context=f"{context}: sample[{idx}]")
        return [[float(v) for v in sample] for sample in samples]
    if len(samples) != horizon:
        raise ValueError(f"{context}: samples length {len(samples)} does not match horizon {horizon}")
    _validate_numeric_sequence(samples, context=f"{context}: samples")
    return [[float(v) for v in samples]]


def _validate_samples(samples: list[float] | list[list[float]], horizon: int, *, context: str) -> None:
    if not isinstance(samples, list):
        raise TypeError(f"{context}: samples must be a list")
    if not samples:
        return
    if isinstance(samples[0], list):
        for idx, sample in enumerate(samples):
            if len(sample) != horizon:
                raise ValueError(
                    f"{context}: sample[{idx}] length {len(sample)} does not match horizon {horizon}"
                )
            _validate_numeric_sequence(sample, context=f"{context}: sample[{idx}]")
    elif len(samples) != horizon:
        raise ValueError(f"{context}: samples length {len(samples)} does not match horizon {horizon}")
    else:
        _validate_numeric_sequence(samples, context=f"{context}: samples")


def _validate_quantiles(quantiles: dict[str, list[float]], horizon: int, *, context: str) -> None:
    if not isinstance(quantiles, dict):
        raise TypeError(f"{context}: quantiles must be a dict")
    if not quantiles:
        raise ValueError(f"{context}: quantiles is empty")
    for key, series in quantiles.items():
        if not isinstance(series, Iterable):
            raise TypeError(f"{context}: quantile {key} is not a sequence")
        if len(series) != horizon:
            raise ValueError(
                f"{context}: quantile {key} length {len(series)} does not match horizon {horizon}"
            )
        _validate_numeric_sequence(series, context=f"{context}: quantile[{key}]")


def _validate_numeric_sequence(values: Iterable[float], *, context: str) -> None:
    for idx, value in enumerate(values):
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{context}[{idx}] is not numeric: {value!r}") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"{context}[{idx}] must be finite, got {value!r}")
