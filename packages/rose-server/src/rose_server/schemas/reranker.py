import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RerankRequest(BaseModel):
    model: str = Field(default="Qwen3-Reranker-0.6B-ONNX")
    query: str = Field(..., description="The search query")
    documents: List[str] = Field(..., description="Documents to rerank")
    top_n: Optional[int] = Field(default=None, description="Number of results to return")
    return_documents: bool = Field(default=True, description="Include documents in response")


class RerankResult(BaseModel):
    index: int = Field(..., description="Original document index")
    relevance_score: float = Field(..., description="Relevance score 0-1")
    document: Optional[str] = Field(default=None, description="Original document text")


class RerankResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"rerank-{int(time.time() * 1000)}")
    results: List[RerankResult] = Field(..., description="Reranked results")
    meta: Dict[str, Any] = Field(default_factory=dict)
