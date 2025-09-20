from sqlmodel import Field, SQLModel


class Thread(SQLModel, table=True):
    __tablename__ = "threads"

    id: str = Field(primary_key=True)
    data: str = Field()


class ThreadItemEntity(SQLModel, table=True):
    __tablename__ = "thread_items"

    id: str = Field(primary_key=True)
    thread_id: str = Field(foreign_key="threads.id", index=True)
    data: str = Field()


class AttachmentEntity(SQLModel, table=True):
    __tablename__ = "attachments"

    id: str = Field(primary_key=True)
    data: str = Field()
