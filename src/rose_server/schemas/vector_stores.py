from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class Vector(BaseModel):
    id: str
    values: List[float] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VectorStoreCreate(BaseModel):
    name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    file_ids: Optional[List[str]] = None


class VectorStoreUpdate(BaseModel):
    name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class VectorStoreMetadata(BaseModel):
    object: str = "vector_store"
    id: str
    name: str
    dimensions: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: int


class VectorStoreList(BaseModel):
    object: str = "list"
    data: List[VectorStoreMetadata]


class VectorSearch(BaseModel):
    query: Union[str, List[float]]
    max_num_results: int = 10
    filters: Optional[Dict[str, Any]] = None
    include_metadata: bool = True
    include_values: bool = False


class VectorSearchResult(BaseModel):
    object: str = "list"
    data: List[Vector]
    usage: Dict[str, int]
