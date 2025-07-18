"""Fine-tuning models and schemas."""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from rose_server.entities.fine_tuning import FineTuningEvent, FineTuningJob


class Error(BaseModel):
    """Error information for failed jobs."""

    code: str
    message: str
    param: Optional[str] = None


class Hyperparameters(BaseModel):
    """Hyperparameters for fine-tuning."""

    batch_size: Optional[int] = Field(
        default=4,  # Default batch size
        description="Number of examples in each batch",
    )
    learning_rate_multiplier: Optional[float] = Field(
        default=1.0,  # Default multiplier
        description="Scaling factor for the learning rate",
    )
    n_epochs: Optional[int] = Field(
        default=3,  # Default epochs
        description="Number of epochs to train for",
    )
    # Additional fields from normalized hyperparameters
    learning_rate: Optional[float] = None
    base_learning_rate: Optional[float] = None


class SupervisedHyperparameters(Hyperparameters):
    """Hyperparameters for supervised fine-tuning (SFT)."""

    # Inherits all fields and validators from Hyperparameters
    pass


class DpoHyperparameters(Hyperparameters):
    """Hyperparameters for Direct Preference Optimization (DPO)."""

    # Inherits all fields and validators from Hyperparameters
    # Add DPO-specific fields
    beta: Optional[Union[Literal["auto"], float]] = Field(
        0.1, description="The beta value for DPO - higher values increase penalty weight"
    )


class DpoConfig(BaseModel):
    """DPO configuration wrapper."""

    hyperparameters: Optional[DpoHyperparameters] = None


class SupervisedConfig(BaseModel):
    """Supervised configuration wrapper."""

    hyperparameters: Optional[SupervisedHyperparameters] = None


class SupervisedMethod(BaseModel):
    """Supervised fine-tuning method configuration."""

    type: Literal["supervised"] = "supervised"
    supervised: Optional[SupervisedConfig] = None


class DpoMethod(BaseModel):
    """Direct Preference Optimization method configuration."""

    type: Literal["dpo"] = "dpo"
    dpo: Optional[DpoConfig] = None


Method = Union[SupervisedMethod, DpoMethod]


class HyperparameterInput(BaseModel):
    """Input hyperparameters that can accept 'auto' strings."""

    batch_size: Optional[Union[Literal["auto"], int]] = None
    learning_rate_multiplier: Optional[Union[Literal["auto"], float]] = None
    n_epochs: Optional[Union[Literal["auto"], int]] = None
    # Direct learning rate (not multiplier)
    learning_rate: Optional[float] = None


class FineTuningJobCreateRequest(BaseModel):
    """Request body for creating a fine-tuning job."""

    model: str = Field(..., description="The name of the model to fine-tune")
    training_file: str = Field(..., description="The ID of the uploaded file for training")
    hyperparameters: Optional[Union[Dict[str, Any], HyperparameterInput]] = Field(
        default=None, description="Hyperparameters for training"
    )
    method: Optional[Method] = Field(
        default_factory=lambda: SupervisedMethod(), description="Fine-tuning method configuration"
    )
    suffix: Optional[str] = Field(default=None, description="Suffix for the fine-tuned model")
    validation_file: Optional[str] = Field(default=None, description="The ID of the uploaded file for validation")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Set of 16 key-value pairs for job metadata")
    # ROSE-specific extension
    trainer: Optional[Literal["huggingface", "pytorch"]] = Field(
        default=None, description="Training backend to use (ROSE-specific extension)"
    )


class FineTuningJobResponse(BaseModel):
    """Response model for fine-tuning jobs."""

    id: str
    created_at: int
    error: Optional[Error] = None
    fine_tuned_model: Optional[str] = None
    finished_at: Optional[int] = None
    hyperparameters: Hyperparameters
    model: str
    object: Literal["fine_tuning.job"] = "fine_tuning.job"
    organization_id: str
    result_files: List[str]
    seed: int
    status: Literal["validating_files", "queued", "running", "succeeded", "failed", "cancelled"]
    trained_tokens: Optional[int] = None
    training_file: str
    validation_file: Optional[str] = None
    estimated_finish: Optional[int] = None
    integrations: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    method: Optional[Dict[str, Any]] = None
    suffix: Optional[str] = None
    # ROSE-specific fields
    trainer: Optional[str] = None

    @classmethod
    def from_entity(cls, job: FineTuningJob) -> "FineTuningJobResponse":
        """Convert internal FineTuningJob entity."""
        data = job.model_dump()

        if "hyperparameters" not in data or data["hyperparameters"] is None:
            data["hyperparameters"] = {}

        data["metadata"] = data.pop("meta", None)

        # Remove internal fields
        internal_fields = ["started_at"]
        for field in internal_fields:
            data.pop(field, None)

        status_mapping = {"cancelling": "cancelled", "pausing": "queued"}
        if data["status"] in status_mapping:
            data["status"] = status_mapping[data["status"]]

        # Ensure hyperparameters is properly typed
        if data.get("hyperparameters"):
            data["hyperparameters"] = Hyperparameters(**data["hyperparameters"])

        # Ensure error is properly typed
        if data.get("error") and isinstance(data["error"], dict):
            data["error"] = Error(**data["error"])

        return cls(**data)


class FineTuningJobEventResponse(BaseModel):
    """Response model for fine-tuning job events."""

    id: str
    created_at: int
    level: str
    message: str
    object: Literal["fine_tuning.job.event"] = "fine_tuning.job.event"
    data: Optional[Dict[str, Any]] = None

    @classmethod
    def from_entity(cls, event: FineTuningEvent) -> "FineTuningJobEventResponse":
        """Convert internal FineTuningEvent entity."""
        return cls(**event.model_dump())
