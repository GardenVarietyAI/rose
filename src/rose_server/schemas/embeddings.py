"""Embeddings API schemas."""
from typing import Any, Dict, List, Union
from pydantic import BaseModel, Field

class EmbeddingsRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "text-embedding-ada-002"
    encoding_format: str = Field(
        default="float", description="The format of the returned embeddings: 'float' or 'base64'"
    )

class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    usage: Dict[str, int]