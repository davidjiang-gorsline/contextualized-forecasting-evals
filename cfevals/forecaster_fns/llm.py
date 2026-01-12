from __future__ import annotations

import importlib.util
import json
import os
import re
from dataclasses import dataclass
from typing import Any

from cfevals.forecaster_fns.base import ForecastResult
from cfevals.prompts import ForecastPrompt


def parse_json_response(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


@dataclass
class OpenAIForecaster:
    model: str = "gpt-4o-mini"
    max_retries: int = 2

    def __post_init__(self) -> None:
        if importlib.util.find_spec("openai") is None:
            raise RuntimeError("openai is not installed")
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        from openai import OpenAI  # noqa: PLC0415

        self.client = OpenAI()

    def __call__(self, prompt: ForecastPrompt) -> ForecastResult:
        messages = prompt.to_chat_messages()
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
                if point is None:
                    raise ValueError("missing point_forecast in response")
                return ForecastResult(
                    point_forecast=[float(v) for v in point],
                    samples=samples,
                    quantiles=quantiles,
                    metadata={"raw": payload},
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                messages = messages + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": "Return ONLY valid JSON with point_forecast."},
                ]
        raise RuntimeError(f"LLM response parsing failed: {last_error}")
