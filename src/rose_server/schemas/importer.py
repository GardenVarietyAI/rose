from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    role: str
    content: str


class Conversation(BaseModel):
    messages: list[ChatMessage]


class ImportRequest(BaseModel):
    import_source: str = Field(min_length=1)
    conversations: list[Conversation] = Field(min_length=1)

    @field_validator("import_source")
    @classmethod
    def validate_import_source(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("import_source cannot be empty")
        return value


class ImportResponse(BaseModel):
    imported_count: int
    conversations_count: int
