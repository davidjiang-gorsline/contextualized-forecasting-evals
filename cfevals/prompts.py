from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ForecastPrompt:
    history: list[float]
    horizon: int
    context_text: str | None = None
    covariates: dict[str, list[float]] | None = None
    metadata: dict[str, Any] | None = None

    def to_text(self) -> str:
        lines = ["You are a forecasting model.", f"History: {self.history}", f"Horizon: {self.horizon}"]
        if self.covariates:
            lines.append(f"Covariates: {self.covariates}")
        if self.context_text:
            lines.append(f"Context: {self.context_text}")
        return "\n".join(lines)

    def to_chat_messages(self) -> list[dict[str, str]]:
        system = {
            "role": "system",
            "content": "You are a forecasting assistant that returns JSON only.",
        }
        user = {
            "role": "user",
            "content": self.to_text()
            + "\nReturn JSON with keys: point_forecast (list), quantiles (dict) or samples (list of lists).",
        }
        return [system, user]
