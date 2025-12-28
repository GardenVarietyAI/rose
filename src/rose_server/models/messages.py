import time
import uuid as uuid_module
from typing import Any, Dict

from sqlalchemy import JSON, Column, Computed, Index, String
from sqlmodel import Field, SQLModel


class Message(SQLModel, table=True):
    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_completion_id", "completion_id"),
        Index("ix_messages_lens_id", "lens_id"),
        Index("ix_messages_at_name", "at_name"),
        Index("ix_messages_object", "object"),
        Index("ix_messages_root_message_id", "root_message_id"),
        Index("ix_messages_parent_message_id", "parent_message_id"),
        Index("ix_messages_import_source", "import_source"),
    )

    id: int | None = Field(default=None, primary_key=True)
    uuid: str = Field(default_factory=lambda: str(uuid_module.uuid4()), index=True, unique=True)
    thread_id: str | None = Field(default=None, index=True)
    role: str
    content: str | None = Field(default=None)
    reasoning: str | None = Field(default=None)
    model: str | None = Field(default=None)
    meta: Dict[str, Any] | None = Field(default=None, sa_column=Column(JSON(none_as_null=True), nullable=True))
    created_at: int = Field(default_factory=lambda: int(time.time()))
    accepted_at: int | None = Field(default=None, index=True)
    deleted_at: int | None = Field(default=None, index=True)
    import_batch_id: str | None = Field(default=None, index=True)
    import_external_id: str | None = Field(default=None)

    completion_id: str | None = Field(
        default=None,
        sa_column=Column(String, Computed("json_extract(meta, '$.completion_id')"), index=False),
    )

    import_source: str | None = Field(
        default=None,
        sa_column=Column(String, Computed("json_extract(meta, '$.import_source')"), index=False),
    )

    lens_id: str | None = Field(
        default=None,
        sa_column=Column(String, Computed("json_extract(meta, '$.lens_id')"), index=False),
    )

    at_name: str | None = Field(
        default=None,
        sa_column=Column(String, Computed("json_extract(meta, '$.at_name')"), index=False),
    )

    object: str | None = Field(
        default=None,
        sa_column=Column(String, Computed("json_extract(meta, '$.object')"), index=False),
    )

    root_message_id: str | None = Field(
        default=None,
        sa_column=Column(String, Computed("json_extract(meta, '$.root_message_id')"), index=False),
    )

    parent_message_id: str | None = Field(
        default=None,
        sa_column=Column(String, Computed("json_extract(meta, '$.parent_message_id')"), index=False),
    )

    @property
    def display_role(self) -> str:
        if self.role == "assistant":
            if self.meta:
                at_name = str(self.meta.get("lens_at_name", "")).strip()
                if at_name:
                    return f"@{at_name}"
        return self.role
