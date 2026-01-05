from pydantic import BaseModel, Field

from rose_server.schemas.query import QueryRequest


class SearchAppData(QueryRequest):
    lens_map: dict[str, str] = Field(default_factory=dict)
    factsheet_map: dict[str, str] = Field(default_factory=dict)
    factsheet_title_map: dict[str, str] = Field(default_factory=dict)


class ThreadsAppData(BaseModel):
    currentThreadId: str | None = None
    deleting: bool = False


class AppData(BaseModel):
    search: SearchAppData = Field(default_factory=SearchAppData)
    threads: ThreadsAppData = Field(default_factory=ThreadsAppData)
