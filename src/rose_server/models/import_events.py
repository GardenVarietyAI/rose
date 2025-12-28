import time
import uuid as uuid_module

from sqlmodel import Field, SQLModel


class ImportEvent(SQLModel, table=True):
    __tablename__ = "import_events"

    id: int | None = Field(default=None, primary_key=True)
    uuid: str = Field(default_factory=lambda: str(uuid_module.uuid4()), index=True, unique=True)
    batch_id: str = Field(default_factory=lambda: str(uuid_module.uuid4()), index=True)
    import_source: str = Field(index=True)
    imported_count: int
    skipped_duplicates: int
    total_records: int
    created_at: int = Field(default_factory=lambda: int(time.time()), index=True)
