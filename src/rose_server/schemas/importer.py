from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator


class ImportMessage(BaseModel):
    thread_id: str
    role: str
    content: str | None
    model: str | None
    created_at: int
    import_external_id: str
    meta: dict[str, Any] | None

    model_config = ConfigDict(extra="ignore")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got '{v}'")
        return v

    @field_validator("created_at")
    @classmethod
    def validate_timestamp(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"created_at must be non-negative, got {v}")
        return v


class ImportRequest(BaseModel):
    import_source: str
    messages: list[ImportMessage]

    model_config = ConfigDict(extra="ignore")

    @field_validator("import_source")
    @classmethod
    def validate_import_source(cls, value: str) -> str:
        if not value:
            raise ValueError("import_source cannot be empty")
        return value
