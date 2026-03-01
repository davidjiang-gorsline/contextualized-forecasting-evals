from __future__ import annotations

import importlib.util
import json
import os
import re
from dataclasses import dataclass
from typing import Any

from cfevals.models.base import ForecastRequest, ForecastResult, Model


def parse_json_response(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


@dataclass
class OpenAIModel(Model):
    model: str = "gpt-4o-mini"
    max_retries: int = 2
    max_context_events_in_prompt: int = 64

    def __post_init__(self) -> None:
        if importlib.util.find_spec("openai") is None:
            raise RuntimeError("openai is not installed")
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        from openai import OpenAI  # noqa: PLC0415

        self.client = OpenAI()

    def predict(self, request: ForecastRequest) -> ForecastResult:
        messages = _build_messages(request, max_context_events_in_prompt=self.max_context_events_in_prompt)
        last_error = None
        for _ in range(self.max_retries + 1):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            try:
                payload = parse_json_response(content)
                point = payload.get("point_forecast") or payload.get("point")
                samples = payload.get("samples")
                quantiles = payload.get("quantiles")
                justification = payload.get("justification") or payload.get("rationale")
                if point is None:
                    raise ValueError("missing point_forecast in response")
                return ForecastResult(
                    point_forecast=[float(v) for v in point],
                    samples=samples,
                    quantiles=quantiles,
                    justification_text=str(justification) if justification is not None else None,
                    metadata={"raw": payload},
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                messages = messages + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": "Return ONLY valid JSON with point_forecast."},
                ]
        raise RuntimeError(f"LLM response parsing failed: {last_error}")


def _build_messages(
    request: ForecastRequest,
    *,
    max_context_events_in_prompt: int,
) -> list[dict[str, str]]:
    lines = [
        "You are a forecasting model.",
        f"History: {request.history}",
        f"Horizon: {request.horizon}",
    ]
    if request.features:
        lines.append(f"Features: {request.features}")
    if request.context_text:
        lines.append(f"Context: {request.context_text}")
    if request.context_events:
        lines.append(f"Context events available: {len(request.context_events)}")
        preview = _render_context_preview(
            request.context_events,
            max_events=max_context_events_in_prompt,
        )
        if preview:
            lines.append("Context preview (retrieved from full context_events):")
            lines.append(preview)
    if request.metadata:
        lines.append(f"Metadata: {request.metadata}")
    prompt = "\n".join(lines)
    system = {"role": "system", "content": "You are a forecasting assistant that returns JSON only."}
    user = {
        "role": "user",
        "content": prompt
        + "\nReturn JSON with keys: point_forecast (list), quantiles (dict) or samples (list of lists), "
        "and optional justification (string).",
    }
    return [system, user]


def _render_context_preview(events, *, max_events: int) -> str:
    if not events:
        return ""
    selected = events[-max(max_events, 1) :]
    lines: list[str] = []
    for event in selected:
        lines.append(
            f"- [{event.as_of_time.isoformat()}] ({event.modality}) {event.text}"
        )
    return "\n".join(lines)
