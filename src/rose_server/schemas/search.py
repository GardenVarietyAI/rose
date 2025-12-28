from pydantic import BaseModel


class SearchRequest(BaseModel):
    q: str = ""
    limit: int = 10
    exact: bool = False
    lens_id: str | None = None
