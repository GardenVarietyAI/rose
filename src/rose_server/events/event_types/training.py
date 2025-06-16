"""Training-related events for fine-tuning pipeline integration."""
from typing import Any, Dict, Literal, Optional

from pydantic import Field

from .base import LLMEvent


class TrainingStarted(LLMEvent):
    """Fired when a training job begins."""

    job_id: str = Field(..., description="Fine-tuning job identifier")
    base_model: str = Field(..., description="Base model being fine-tuned")
    training_method: Literal["supervised", "dpo"] = Field(..., description="Training method")
    num_examples: int = Field(..., ge=1, description="Number of training examples")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Training hyperparameters")

class TrainingStepCompleted(LLMEvent):
    """Fired after each training step completes."""

    job_id: str = Field(..., description="Fine-tuning job identifier")
    step: int = Field(..., ge=0, description="Completed step number")
    loss: float = Field(..., description="Training loss for this step")
    learning_rate: Optional[float] = Field(None, description="Current learning rate")
    tokens_processed: int = Field(..., ge=0, description="Total tokens processed so far")
    estimated_finish_time: Optional[float] = Field(None, description="Estimated completion timestamp")

class CheckpointSaved(LLMEvent):
    """Fired when a training checkpoint is saved."""

    job_id: str = Field(..., description="Fine-tuning job identifier")
    checkpoint_path: str = Field(..., description="Path to saved checkpoint")
    step: int = Field(..., ge=0, description="Step number of this checkpoint")
    loss: float = Field(..., description="Loss at checkpoint time")

class TrainingCompleted(LLMEvent):
    """Fired when training job completes successfully."""

    job_id: str = Field(..., description="Fine-tuning job identifier")
    final_model_path: str = Field(..., description="Path to final trained model")
    total_steps: int = Field(..., ge=0, description="Total training steps completed")
    final_loss: float = Field(..., description="Final training loss")
    training_time: float = Field(..., ge=0, description="Total training time in seconds")
    model_registered: bool = Field(default=False, description="Whether model was registered for inference")

class TrainingError(LLMEvent):
    """Fired when training encounters an error."""

    job_id: str = Field(..., description="Fine-tuning job identifier")
    error_message: str = Field(..., description="Error description")
    error_type: str = Field(..., description="Type/category of error")
    step: Optional[int] = Field(None, description="Step where error occurred")
    recoverable: bool = Field(default=False, description="Whether training can be resumed")