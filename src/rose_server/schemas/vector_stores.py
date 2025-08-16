from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class Vector(BaseModel):
    id: str
    values: List[float] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = Field(default=None, description="Similarity score (higher is more similar)")


class StaticChunkingConfig(BaseModel):
    max_chunk_size_tokens: int = Field(
        800,
        description="Maximum number of tokens in each chunk.",
        ge=50,
        le=4000,
    )
    chunk_overlap_tokens: int = Field(
        150,
        description="Number of tokens shared between consecutive chunks.",
        ge=0,
        le=4000,
    )


class ChunkingStrategy(BaseModel):
    type: Literal["static"] = Field("static", description="The chunking strategy type.")
    static: StaticChunkingConfig = Field(..., description="Chunking strategy configuration.")


class VectorStoreCreate(BaseModel):
    name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    file_ids: Optional[List[str]] = None
    chunking_strategy: Optional[ChunkingStrategy] = None


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
    max_num_results: int = Field(default=10, ge=1, le=100, description="Maximum number of results to return (1-100)")
    filters: Optional[Dict[str, Any]] = None
    include_metadata: bool = True
    include_values: bool = False


class VectorSearchChunk(BaseModel):
    file_id: str
    filename: str
    score: float
    attributes: Dict[str, Any] = Field(default_factory=dict)
    content: List[Dict[str, str]]


class VectorSearchUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class VectorSearchResult(BaseModel):
    object: str = "vector_store.search_results.page"
    search_query: str
    data: List[VectorSearchChunk]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool = False
    next_page: Optional[str] = None
    usage: VectorSearchUsage


class VectorStoreFileCreate(BaseModel):
    file_id: str
    chunking_strategy: Optional[ChunkingStrategy] = None


class VectorStoreFile(BaseModel):
    object: str = "vector_store.file"
    id: str
    vector_store_id: str
    status: str
    created_at: int
    last_error: Optional[Dict[str, Any]] = None
