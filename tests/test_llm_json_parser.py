from cfevals.models.llm import parse_json_response


def test_llm_json_parser():
    text = "Here is the forecast: {\"point_forecast\": [1, 2]}"
    payload = parse_json_response(text)
    assert payload["point_forecast"] == [1, 2]
