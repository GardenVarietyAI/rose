from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from rose_server.models.messages import Message


class LensMeta(BaseModel):
    model_config = ConfigDict(frozen=True)

    object: Literal["lens"] = "lens"
    at_name: str
    label: str
    root_message_id: str | None = None
    parent_message_id: str | None = None


class LensMessage(BaseModel):
    message: Message

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: Message) -> Message:
        meta = value.meta
        if meta is None:
            raise ValueError("Lens message missing meta")
        try:
            LensMeta.model_validate(meta)
        except ValidationError as e:
            raise ValueError("Lens message invalid meta") from e
        if value.role != "system":
            raise ValueError("Lens message role must be 'system'")
        return value

    @property
    def lens_id(self) -> str:
        if self.message.meta and self.message.meta.get("root_message_id"):
            return str(self.message.meta["root_message_id"])
        return self.message.uuid

    @property
    def at_name(self) -> str:
        if self.message.meta is None:
            raise ValueError("Lens message missing meta")
        return str(self.message.meta["at_name"])

    @property
    def label(self) -> str:
        if self.message.meta is None:
            raise ValueError("Lens message missing meta")
        return str(self.message.meta["label"])


class UserMessage(BaseModel):
    message: Message

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: Message) -> Message:
        if value.role != "user":
            raise ValueError("User message role must be 'user'")
        if value.content is None or not str(value.content).strip():
            raise ValueError("User message missing content")
        return value


class AssistantMessage(BaseModel):
    message: Message

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: Message) -> Message:
        if value.role != "assistant":
            raise ValueError("Assistant message role must be 'assistant'")
        if value.content is None or not str(value.content).strip():
            raise ValueError("Assistant message missing content")
        return value
