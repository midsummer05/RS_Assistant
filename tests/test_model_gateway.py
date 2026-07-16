import json

from pydantic import BaseModel

from rs_agent.models import gateway
from rs_agent.models.gateway import OpenAICompatibleChatModel


class DemoResponse(BaseModel):
    answer: str


class FakeHTTPResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(
            {"choices": [{"message": {"content": '{"answer":"ok"}'}}]}
        ).encode()


def test_deepseek_json_object_payload(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        return FakeHTTPResponse()

    monkeypatch.setattr(gateway, "urlopen", fake_urlopen)
    model = OpenAICompatibleChatModel(
        api_key="test-key",
        model_name="deepseek-v4-pro",
        base_url="https://api.deepseek.com",
        response_format="json_object",
    )

    result = model.generate_json("Return JSON.", "test", DemoResponse)

    assert result.answer == "ok"
    assert captured["payload"]["response_format"] == {"type": "json_object"}
