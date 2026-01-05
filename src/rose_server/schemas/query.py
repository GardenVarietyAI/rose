from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = ""
    lens_ids: list[str] = Field(default_factory=list)
    factsheet_ids: list[str] = Field(default_factory=list)
    exact: bool = False
    limit: int = Field(default=10, ge=1, le=100)
