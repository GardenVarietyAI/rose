from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from rose_server.models.messages import Message


class LensMeta(BaseModel):
    model_config = ConfigDict(frozen=True)

    object: Literal["lens"] = "lens"
    lens_id: str
    at_name: str
    label: str


class JobMeta(BaseModel):
    model_config = ConfigDict(frozen=False)

    object: Literal["job"] = "job"
    job_name: str
    status: str
    user_message_uuid: str
    attempt: int = 0
    lens_id: str | None = None
    lens_at_name: str | None = None
    assistant_message_uuid: str | None = None
    error: str | None = None


class LensMessage(BaseModel):
    message: Message

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: Message) -> Message:
        meta = value.meta
        if meta is None:
            raise ValueError("Lens message missing meta")
        try:
            lens_meta = LensMeta.model_validate(meta)
        except ValidationError as e:
            raise ValueError("Lens message invalid meta") from e
        if value.role != "system":
            raise ValueError("Lens message role must be 'system'")
        if lens_meta.lens_id != value.uuid:
            raise ValueError(f"Lens meta lens_id {lens_meta.lens_id} does not match message uuid {value.uuid}")
        return value

    @property
    def lens_id(self) -> str:
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


class JobMessage(BaseModel):
    message: Message

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: Message) -> Message:
        meta = value.meta
        if meta is None:
            raise ValueError("Job message missing meta")
        try:
            JobMeta.model_validate(meta)
        except ValidationError as e:
            raise ValueError("Job message invalid meta") from e
        if value.role != "system":
            raise ValueError("Job message role must be 'system'")
        return value

    @property
    def status(self) -> str:
        if self.message.meta is None:
            raise ValueError("Job message missing meta")
        return str(self.message.meta["status"])

    @property
    def job_name(self) -> str:
        if self.message.meta is None:
            raise ValueError("Job message missing meta")
        return str(self.message.meta["job_name"])

    @property
    def user_message_uuid(self) -> str:
        if self.message.meta is None:
            raise ValueError("Job message missing meta")
        return str(self.message.meta["user_message_uuid"])


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
