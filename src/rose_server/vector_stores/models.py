from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class Vector(BaseModel):
    id: str
    values: List[float] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VectorStoreCreate(BaseModel):
    name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    file_ids: Optional[List[str]] = None

    @field_validator("name")
    def validate_name(cls, v):
        if not v:
            raise ValueError("Name cannot be empty")
        return v


class VectorStoreUpdate(BaseModel):
    name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class VectorStore(BaseModel):
    id: str
    name: str
    dimensions: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


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


class VectorBatch(BaseModel):
    vectors: List[List[float]]
    ids: List[str]
    metadata: Optional[List[Dict[str, Any]]] = None

    @field_validator("ids")
    def validate_ids(cls, v, values):
        vectors = values.data.get("vectors", [])
        if len(v) != len(vectors):
            raise ValueError(f"Number of IDs ({len(v)}) must match number of vectors ({len(vectors)})")
        return v

    @field_validator("metadata")
    def validate_metadata(cls, v, values):
        if v is None:
            return [{}] * len(values.data.get("vectors", []))
        vectors = values.data.get("vectors", [])
        if len(v) != len(vectors):
            raise ValueError(f"Number of metadata entries ({len(v)}) must match number of vectors ({len(vectors)})")
        return v


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
