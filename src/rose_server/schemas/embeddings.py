"""Embeddings API schemas."""

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """Request for generating embeddings."""

    input: Union[str, List[str]] = Field(description="Input text(s) to embed")
    model: str = Field(description="Model to use for embeddings")
    encoding_format: Optional[Literal["float", "base64"]] = Field(
        default="float", description="Format for the embeddings"
    )
    dimensions: Optional[int] = Field(default=None, description="Number of dimensions for the embeddings")
    user: Optional[str] = Field(default=None, description="User identifier")


class EmbeddingData(BaseModel):
    """Individual embedding data."""

    object: str = Field(default="embedding")
    embedding: List[float] = Field(description="The embedding vector")
    index: int = Field(description="Index of this embedding in the request")


class EmbeddingResponse(BaseModel):
    """Response for embeddings API."""

    object: str = Field(default="list")
    data: List[EmbeddingData] = Field(description="List of embeddings")
    model: str = Field(description="Model used for embeddings")
    usage: dict = Field(description="Usage statistics")
