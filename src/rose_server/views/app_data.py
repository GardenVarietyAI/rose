from pydantic import BaseModel, Field


class SearchAppData(BaseModel):
    lens_map: dict[str, str] = Field(default_factory=dict)
    factsheet_map: dict[str, str] = Field(default_factory=dict)
    factsheet_title_map: dict[str, str] = Field(default_factory=dict)
    factsheet_ids: list[str] = Field(default_factory=list)
    content: str = ""
    lens_ids: list[str] = Field(default_factory=list)
    limit: int = 10
    exact: bool = False


class ThreadsAppData(BaseModel):
    currentThreadId: str | None = None
    deleting: bool = False


class AppData(BaseModel):
    search: SearchAppData = Field(default_factory=SearchAppData)
    threads: ThreadsAppData = Field(default_factory=ThreadsAppData)
