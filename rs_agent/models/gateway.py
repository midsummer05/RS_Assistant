from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel


class ModelGatewayError(RuntimeError):
    pass


class ChatModel(Protocol):
    model_name: str

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
    ) -> BaseModel:
        ...


@dataclass
class OpenAICompatibleChatModel:
    api_key: str
    model_name: str
    base_url: str = "https://api.openai.com/v1"
    timeout_sec: float = 120.0
    response_format: str = "json_schema"
    max_retries: int = 2

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
    ) -> BaseModel:
        schema = response_model.model_json_schema()
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"{system_prompt}\n"
                        "必须输出 JSON，且必须满足以下 JSON Schema：\n"
                        f"{json.dumps(schema, ensure_ascii=False)}"
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 8192,
        }
        if self.response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        else:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": schema,
                },
            }

        last_error: Exception | None = None
        for _ in range(self.max_retries + 1):
            request = Request(
                f"{self.base_url.rstrip('/')}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout_sec) as response:
                    body = json.loads(response.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "".join(
                        item.get("text", "") for item in content if item.get("type") == "text"
                    )
                if not content or not content.strip():
                    raise ValueError("LLM returned empty JSON content.")
                return response_model.model_validate_json(content)
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise ModelGatewayError(f"LLM HTTP {exc.code}: {detail}") from exc
            except URLError as exc:
                raise ModelGatewayError(f"LLM connection failed: {exc.reason}") from exc
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = exc
        raise ModelGatewayError(
            f"Invalid structured LLM response after retries: {last_error}"
        ) from last_error


def build_chat_model_from_env() -> ChatModel | None:
    api_key = os.getenv("RS_AGENT_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("RS_AGENT_LLM_MODEL")
    if not api_key or not model_name:
        return None
    base_url = os.getenv("RS_AGENT_LLM_BASE_URL", "https://api.openai.com/v1")
    response_format = os.getenv("RS_AGENT_LLM_RESPONSE_FORMAT")
    if not response_format:
        response_format = "json_object" if "api.deepseek.com" in base_url else "json_schema"
    return OpenAICompatibleChatModel(
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
        timeout_sec=float(os.getenv("RS_AGENT_LLM_TIMEOUT_SEC", "120")),
        response_format=response_format,
        max_retries=int(os.getenv("RS_AGENT_LLM_MAX_RETRIES", "2")),
    )
