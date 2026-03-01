from cfevals.models.base import ForecastRequest
from cfevals.models.naive import LastValueModel


def test_naive_model_returns_justification_text():
    result = LastValueModel().predict(ForecastRequest(history=[1.0, 2.0], horizon=2))
    assert result.justification_text is not None
    assert "Naive baseline" in result.justification_text
