from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from cfevals.benchmarks.base import ScenarioBenchmark, ScenarioSample
from cfevals.context import ContextEvent, render_context_text


@dataclass
class ContextIsKeyBenchmark(ScenarioBenchmark):
    dataset_name: str = "ServiceNow/context-is-key"
    split: str = "test"
    max_samples: int | None = None
    cache_dir: str | None = None
    allow_fallback: bool = False
    streaming: bool = False

    def load(self) -> list[ScenarioSample]:
        from datasets import load_dataset  # noqa: PLC0415

        _validate_config(self)
        cache_dir = self.cache_dir or os.environ.get("CFEVALS_CACHE")
        try:
            ds = load_dataset(
                self.dataset_name,
                split=self.split,
                cache_dir=cache_dir,
                streaming=self.streaming,
            )
        except Exception as exc:  # noqa: BLE001
            if self.allow_fallback:
                return _fallback_samples()
            raise RuntimeError(
                f"Failed to load CiK dataset {self.dataset_name!r}. "
                "Check network access or set CFEVALS_CACHE for offline use."
            ) from exc

        samples: list[ScenarioSample] = []
        limit = self.max_samples
        for idx, row in enumerate(ds):
            if limit is not None and idx >= limit:
                break
            sample = _build_sample(row, idx, dataset_name=self.dataset_name)
            if sample is None:
                continue
            samples.append(sample)
        if not samples and self.allow_fallback:
            return _fallback_samples()
        if not samples:
            raise ValueError(
                f"CiK benchmark {self.dataset_name!r} produced zero valid samples. "
                "Expected numeric history/future and at least one context event per sample."
            )
        return samples


def _build_sample(row: dict[str, Any], idx: int, *, dataset_name: str) -> ScenarioSample | None:
    history = _parse_numeric_series(row.get("history"))
    future = _parse_numeric_series(row.get("future"))
    if not history and not future:
        history = _parse_numeric_series(row.get("past_time"))
        future = _parse_numeric_series(row.get("future_time"))
    if not future:
        return None

    context_events = _build_context_events(row, idx=idx)
    if not context_events:
        return None
    context = render_context_text(context_events)

    roi = _parse_roi(row.get("roi"))
    if roi is None:
        roi = _parse_constraint_roi(row)

    sample_id = row.get("sample_id")
    if not sample_id:
        name = row.get("name")
        seed = row.get("seed")
        if name is not None and seed is not None:
            sample_id = f"{name}-{seed}-{idx}"
        else:
            sample_id = f"cik-{idx}"

    metadata = {
        "dataset": dataset_name,
        "name": row.get("name"),
        "seed": row.get("seed"),
        "region_of_interest": row.get("region_of_interest"),
        "metric_scaling": row.get("metric_scaling"),
    }
    return ScenarioSample(
        sample_id=sample_id,
        history=history,
        future=future,
        context_text=context,
        context_events=context_events,
        roi=roi,
        metadata=metadata,
    )


def _build_context_events(row: dict[str, Any], *, idx: int) -> list[ContextEvent]:
    ref_time = _infer_reference_time(row, idx=idx)
    events: list[ContextEvent] = []
    background = row.get("background")
    if background:
        events.append(
            ContextEvent(
                event_time=ref_time - timedelta(minutes=2),
                available_at=ref_time - timedelta(minutes=2),
                text=f"Background: {background}",
                modality="text",
                source="context_is_key.background",
            )
        )
    context = row.get("context")
    if context:
        events.append(
            ContextEvent(
                event_time=ref_time - timedelta(minutes=1),
                available_at=ref_time - timedelta(minutes=1),
                text=str(context),
                modality="text",
                source="context_is_key.context",
            )
        )
    scenario = row.get("scenario")
    if scenario:
        events.append(
            ContextEvent(
                event_time=ref_time,
                available_at=ref_time,
                text=f"Scenario: {scenario}",
                modality="text",
                source="context_is_key.scenario",
            )
        )
    constraints = row.get("constraints")
    if constraints:
        events.append(
            ContextEvent(
                event_time=ref_time,
                available_at=ref_time,
                text=f"Constraints: {constraints}",
                modality="text",
                source="context_is_key.constraints",
            )
        )
    return events


def _parse_numeric_series(value: Any) -> list[float]:
    parsed = _maybe_json(value)
    return _extract_numeric_values(parsed)


def _maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return []
    if stripped[0] not in ("[", "{"):
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _extract_numeric_values(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        number = _to_finite_float(value)
        return [number] if number is not None else []
    if isinstance(value, str):
        number = _to_finite_float(value)
        return [number] if number is not None else []
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(_extract_numeric_values(item))
        return out
    if isinstance(value, dict):
        if not value:
            return []
        nested_values = list(value.values())
        if any(isinstance(item, (dict, list)) for item in nested_values):
            # Prefer deterministic key ordering for nested structures.
            for key in sorted(value.keys(), key=lambda item: str(item)):
                out = _extract_numeric_values(value[key])
                if out:
                    return out
            return []
        items = sorted(value.items(), key=lambda item: str(item[0]))
        return [val for _, raw in items if (val := _to_finite_float(raw)) is not None]
    return []


def _to_finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(out):
        return out
    return None


def _parse_roi(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    low = _to_finite_float(value[0])
    high = _to_finite_float(value[1])
    if low is None or high is None:
        return None
    if low > high:
        return None
    return (low, high)


def _parse_constraint_roi(row: dict[str, Any]) -> tuple[float, float] | None:
    low = _to_finite_float(row.get("constraint_min"))
    high = _to_finite_float(row.get("constraint_max"))
    if low is None or high is None:
        return None
    if low > high:
        return None
    return (low, high)


def _infer_reference_time(row: dict[str, Any], *, idx: int) -> datetime:
    past_time = _maybe_json(row.get("past_time"))
    if isinstance(past_time, dict) and past_time:
        parsed = _parse_datetime_keys(past_time)
        if parsed:
            return parsed[-1]
    return datetime(1970, 1, 1) + timedelta(seconds=idx)


def _parse_datetime_keys(payload: Any) -> list[datetime]:
    times: list[datetime] = []
    if isinstance(payload, dict):
        values = payload.values()
        if any(isinstance(v, dict) for v in values):
            for value in values:
                times.extend(_parse_datetime_keys(value))
            return sorted(times)
        for key in payload.keys():
            dt = _parse_datetime(key)
            if dt is not None:
                times.append(dt)
    return sorted(times)


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        return parsed.astimezone(datetime.UTC).replace(tzinfo=None)
    return parsed


def _fallback_samples() -> list[ScenarioSample]:
    event = ContextEvent(
        event_time=datetime(1970, 1, 1),
        available_at=datetime(1970, 1, 1),
        text="Synthetic fallback sample.",
        modality="text",
        source="context_is_key.fallback",
    )
    return [
        ScenarioSample(
            sample_id="cik-fallback-0",
            history=[1.0, 2.0, 3.0],
            future=[3.5, 3.7],
            context_text=render_context_text([event]),
            context_events=[event],
            roi=(0.0, 10.0),
        )
    ]


def _validate_config(config: ContextIsKeyBenchmark) -> None:
    if not config.dataset_name:
        raise ValueError("dataset_name must be set for ContextIsKeyBenchmark")
    if not config.split:
        raise ValueError("split must be set for ContextIsKeyBenchmark")
    if config.max_samples is not None and config.max_samples <= 0:
        raise ValueError("max_samples must be > 0 when provided")
