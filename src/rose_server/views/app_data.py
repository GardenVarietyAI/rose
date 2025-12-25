from pydantic import BaseModel, Field


class SearchAppData(BaseModel):
    lens_map: dict[str, str] = Field(default_factory=dict)
    query: str = ""
    lens_id: str = ""
    limit: int = 10
    exact: bool = False


class AppData(BaseModel):
    search: SearchAppData = Field(default_factory=SearchAppData)
