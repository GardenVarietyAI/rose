import json
import pathlib
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class LlamaError(Exception):
    pass


class LlamaUnavailableError(LlamaError):
    pass


class LlamaStatusError(LlamaError):
    def __init__(self, status_code: int, text: str) -> None:
        super().__init__(f"Upstream returned HTTP {status_code}")
        self.status_code = status_code
        self.text = text


class LlamaInvalidJSONError(LlamaError):
    pass


class LlamaInvalidResponseError(LlamaError):
    pass


class LlamaMissingChoicesError(LlamaError):
    pass


class CompletionMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    content: Any


class CompletionChoice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: CompletionMessage
    finish_reason: Any = None


class CompletionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    choices: list[CompletionChoice] = Field(default_factory=list)
    model: str | None = None
    usage: dict[str, Any] | None = None


def normalize_model_name(model: str) -> tuple[str, str | None]:
    model = model.strip()
    if not model or model == "default":
        return ("default", None)

    if ("/" in model or "\\" in model) and model.lower().endswith(".gguf"):
        name = pathlib.PurePath(model).name
        return (name or model, model)

    return (model, None)


def serialize_message_content(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content)
    except (TypeError, ValueError):
        return str(content)


async def request_chat_completion_json(
    client: httpx.AsyncClient,
    payload: dict[str, Any],
) -> dict[str, Any]:
    try:
        response = await client.post("chat/completions", json=payload)
    except httpx.RequestError as e:
        raise LlamaUnavailableError(str(e)) from e

    if response.status_code >= 400:
        raise LlamaStatusError(response.status_code, response.text)

    try:
        data = response.json()
    except ValueError as e:
        raise LlamaInvalidJSONError(response.text) from e

    if not isinstance(data, dict):
        raise LlamaInvalidResponseError("Expected object JSON response")

    return data


def parse_completion_response(data: dict[str, Any]) -> CompletionResponse:
    try:
        completion = CompletionResponse.model_validate(data)
    except ValidationError as e:
        raise LlamaInvalidResponseError("Invalid completion response shape") from e

    if not completion.choices:
        raise LlamaMissingChoicesError("Completion missing choices")

    return completion
