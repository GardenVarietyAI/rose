"""OpenAI-compatible evaluation schemas."""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class EvalCreateRequest(BaseModel):
    """Request to create an evaluation - OpenAI compatible."""

    name: Optional[str] = None
    data_source_config: Dict[str, Any]
    testing_criteria: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None


class EvalResponse(BaseModel):
    """Evaluation object - OpenAI compatible."""

    id: str
    object: Literal["eval"] = "eval"
    created_at: int
    name: Optional[str] = None
    data_source_config: Dict[str, Any]
    testing_criteria: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None


class EvalListResponse(BaseModel):
    """Paginated response for eval listing - OpenAI SDK compatible."""

    object: Literal["list"] = "list"
    data: List[EvalResponse]
    has_more: bool = False
    first_id: Optional[str] = None
    last_id: Optional[str] = None


class EvalDeleteResponse(BaseModel):
    """Response for eval deletion."""

    id: str
    object: Literal["eval"] = "eval"
    deleted: bool


class SamplingParams(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    max_completion_tokens: Optional[int] = Field(default=None, alias="max_tokens")
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    seed: Optional[int] = None


class DataSourceConfig(BaseModel):
    """Data source configuration for eval runs."""

    type: str
    model: str
    input_messages: Optional[Dict] = None
    sampling_params: Optional[Union[Dict, SamplingParams]] = None
    source: Optional[Dict] = None
    max_samples: Optional[int] = Field(None, description="Maximum number of samples to evaluate")


class EvalRunResponse(BaseModel):
    """Eval run response - OpenAI compatible."""

    id: str
    object: Literal["eval.run"] = "eval.run"
    eval_id: str
    name: str
    model: str
    status: str
    created_at: int
    started_at: Optional[int] = None
    completed_at: Optional[int] = None
    data_source: Optional[DataSourceConfig] = None
    result_counts: Optional[Dict[str, int]] = None
    report_url: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
