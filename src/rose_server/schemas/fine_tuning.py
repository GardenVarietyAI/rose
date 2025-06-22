"""Fine-tuning models - reusing OpenAI SDK types."""

from typing import Literal, Optional, Union

from openai.types.fine_tuning import (
    FineTuningJob,
    FineTuningJobEvent,
    JobCreateParams,
)
from pydantic import BaseModel, Field

__all__ = [
    "FineTuningJob",
    "FineTuningJobEvent",
    "JobCreateParams",
    "FineTuningJobEventStore",
    "SupervisedHyperparameters",
    "DpoHyperparameters",
    "SupervisedConfig",
    "SupervisedMethod",
    "DpoConfig",
    "DpoMethod",
    "Method",
]


class SupervisedHyperparameters(BaseModel):
    """Hyperparameters for supervised fine-tuning (SFT)."""

    batch_size: Optional[Union[Literal["auto"], int]] = Field(None, description="Number of examples in each batch")
    learning_rate_multiplier: Optional[Union[Literal["auto"], float]] = Field(
        None, description="Scaling factor for the learning rate"
    )
    n_epochs: Optional[Union[Literal["auto"], int]] = Field(None, description="Number of epochs to train for")


class DpoHyperparameters(BaseModel):
    """Hyperparameters for Direct Preference Optimization (DPO)."""

    batch_size: Optional[Union[Literal["auto"], int]] = Field(None, description="Number of examples in each batch")
    beta: Optional[Union[Literal["auto"], float]] = Field(
        0.1, description="The beta value for DPO - higher values increase penalty weight"
    )
    learning_rate_multiplier: Optional[Union[Literal["auto"], float]] = Field(
        None, description="Scaling factor for the learning rate"
    )
    n_epochs: Optional[Union[Literal["auto"], int]] = Field(None, description="Number of epochs to train for")


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


class FineTuningJobEventStore(BaseModel):
    """Store for job events."""

    job_id: str
    events: list[FineTuningJobEvent] = Field(default_factory=list)
