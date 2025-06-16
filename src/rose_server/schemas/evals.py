"""OpenAI-compatible evaluation schemas."""
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

class DataSourceSchema(BaseModel):
    """JSON schema for data source."""

    type: str = "object"
    properties: Dict[str, Any]
    required: List[str] = []

class StoredCompletionsDataSourceConfig(BaseModel):
    """Data source config for stored completions."""

    model_config = ConfigDict(populate_by_name=True)
    type: Literal["stored_completions"] = "stored_completions"
    metadata: Optional[Dict[str, str]] = None
    data_schema: Optional[DataSourceSchema] = Field(None, alias="schema")

class StringCheckGrader(BaseModel):
    """String check grader for exact/fuzzy matching."""

    name: str
    id: str
    type: Literal["string_check"] = "string_check"
    input: str
    reference: str
    operation: Literal["eq", "ne", "like", "ilike"] = "eq"

class TextSimilarityGrader(BaseModel):
    """Text similarity grader for BLEU/F1 scoring."""

    name: str
    id: str
    type: Literal["text_similarity"] = "text_similarity"
    input: str
    reference: str
    metric: Literal["bleu", "f1", "rouge"] = "f1"
    pass_threshold: float = 0.8

class EvalCreateRequest(BaseModel):
    """Request to create an evaluation - simplified."""

    name: str
    metadata: Optional[Dict[str, str]] = None

class EvalObject(BaseModel):
    """Evaluation object - OpenAI compatible."""

    object: Literal["eval"] = "eval"
    id: str
    data_source_config: StoredCompletionsDataSourceConfig
    testing_criteria: List[Union[StringCheckGrader, TextSimilarityGrader]]
    name: str
    created_at: int
    metadata: Optional[Dict[str, str]] = None

class EvalListResponse(BaseModel):
    """Paginated response for eval listing - OpenAI SDK compatible."""

    object: Literal["list"] = "list"
    data: List[EvalObject]
    has_more: bool = False
    first_id: Optional[str] = None
    last_id: Optional[str] = None

class EvalDeleteResponse(BaseModel):
    """Response for eval deletion."""

    id: str
    object: Literal["eval"] = "eval"
    deleted: bool

class SamplingParams(BaseModel):
    """OpenAI-compatible sampling parameters."""

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

class EvalRunCreateRequest(BaseModel):
    """Request to create an eval run - OpenAI compatible."""

    name: str
    data_source: DataSourceConfig

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
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None