"""PyTorch trainer implementation."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from datasets import Dataset
from peft.peft_model import PeftModel
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from rose_trainer.types.fine_tuning import Hyperparameters

logger = logging.getLogger(__name__)


class PyTorchTrainer:
    """Encapsulates PyTorch training session with state management.

    Expected batch format from DataLoader:
        - input_ids: Tensor of token IDs
        - attention_mask: Tensor for attention masking
        - labels: Tensor of target token IDs
        - real_lengths: Tensor of actual sequence lengths (before padding)
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, PeftModel],
        tokenizer: PreTrainedTokenizerBase,
        is_peft: bool,
        hyperparams: Hyperparameters,
        device: str,
        checkpoint_dir: Path,
        event_callback: Callable[[str, str, Optional[Dict[str, Any]]], None],
        check_cancel_callback: Callable[[], str],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.is_peft = is_peft
        self.hyperparams = hyperparams
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.event_callback = event_callback
        self.check_cancel_callback = check_cancel_callback

        # Training components (initialized in prepare_training)
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[Any] = None

        # Training state
        self.global_step: int = 0
        self.total_loss: float = 0.0
        self.total_batches: int = 0  # Track total batches for accurate loss averaging
        self.total_tokens: int = 0
        self.total_steps: int = 0
        self.start_time: Optional[float] = None
        self.best_loss: float = float("inf")

    def prepare_training(self, dataloader: DataLoader[Any], optimizer: Optional[Optimizer] = None) -> None:
        """Setup optimizer and scheduler.

        Args:
            dataloader: Training dataloader
            optimizer: Optional pre-configured optimizer. If None, creates AdamW.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        # Use provided optimizer or create default AdamW
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.hyperparams.learning_rate,
                weight_decay=self.hyperparams.weight_decay,
            )

        self.total_steps = len(dataloader) * self.hyperparams.n_epochs // self.hyperparams.gradient_accumulation_steps
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            int(self.total_steps * self.hyperparams.warmup_ratio),
            self.total_steps,
        )

        self.model.train()
        self.start_time = time.time()

    def _create_dataloader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader[Any]:
        """Create a DataLoader with consistent settings.

        Args:
            dataset: Dataset to load
            shuffle: Whether to shuffle the data

        Returns:
            Configured DataLoader
        """
        pad_to_multiple = 8 if self.hyperparams.batch_size > 1 else None
        return DataLoader(
            dataset,
            batch_size=self.hyperparams.batch_size,
            shuffle=shuffle,
            collate_fn=DataCollatorForLanguageModeling(self.tokenizer, mlm=False, pad_to_multiple_of=pad_to_multiple),
        )

    def load_checkpoint(self, checkpoint_path: Path) -> int:
        """Load training state from checkpoint.

        Returns:
            Starting epoch number
        """
        state_path = checkpoint_path / "training_state.pt"
        if not state_path.exists():
            raise ValueError(f"No training state found at {state_path}")

        state = torch.load(state_path, map_location=self.device)

        # Restore training state
        self.global_step = state["global_step"]
        self.best_loss = state["loss"]

        if self.optimizer and state["optimizer_state"]:
            self.optimizer.load_state_dict(state["optimizer_state"])
        if self.scheduler and state["scheduler_state"]:
            self.scheduler.load_state_dict(state["scheduler_state"])

        self.event_callback(
            "info", f"Resumed from checkpoint at epoch {state['epoch'] + 1}, step {self.global_step}", None
        )

        return state["epoch"] + 1

    def _save_checkpoint(self, epoch: int, loss: float) -> None:
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint-epoch-{epoch + 1}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model state
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)

        # Save training state
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "loss": loss,
                "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
                "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            },
            checkpoint_path / "training_state.pt",
        )

        self.event_callback("info", f"Saved checkpoint at epoch {epoch + 1} with loss {loss:.4f}", None)

    def _optimizer_step(self, max_grad_norm: float = 1.0) -> None:
        """Perform single optimizer step with gradient clipping."""
        if self.optimizer is None or self.scheduler is None:
            raise RuntimeError("Optimizer or scheduler not initialized")

        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        # Update global step counter
        self.global_step += 1

    def _process_batch(self, batch: Dict[str, Any]) -> float:
        """Process a single batch through forward and backward pass.

        Args:
            batch: Input batch dictionary

        Returns:
            Scaled loss value for tracking
        """
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Forward pass
        outputs = self.model(**batch)
        loss = outputs.loss

        # Check for NaN/inf
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(f"Loss is {loss.item()}. Training is unstable.")

        scaled_loss = loss / self.hyperparams.gradient_accumulation_steps
        scaled_loss.backward()

        # Track batch statistics
        self.total_batches += 1

        # Return scaled loss for consistent tracking
        return float(scaled_loss.item())

    def run_epoch(self, dataloader: DataLoader[Any], epoch: int) -> None:
        """Run one epoch of training."""
        epoch_loss = 0.0
        epoch_tokens = 0

        for i, batch in enumerate(dataloader):
            # Count tokens before moving to device
            epoch_tokens += batch["real_lengths"].sum().item()

            # Process batch through forward/backward pass
            batch_loss = self._process_batch(batch)
            epoch_loss += batch_loss

            # Update weights if gradient accumulation is complete
            # The or condition handles the final step if it doesn’t align with gradient_accumulation_steps
            if (i + 1) % self.hyperparams.gradient_accumulation_steps == 0 or (i + 1) == len(dataloader):
                self._optimizer_step()

                # Log progress every 10 steps
                if self.global_step % 10 == 0:
                    # Use total_loss for consistent average calculation
                    current_avg_loss = self.total_loss / self.total_batches
                    self.event_callback(
                        "info",
                        f"Step {self.global_step}/{self.total_steps}",
                        {
                            "loss": current_avg_loss,
                            "learning_rate": self.scheduler.get_last_lr()[0] if self.scheduler else 0.0,
                            "epoch": epoch + 1,
                            "global_step": self.global_step,
                            "tokens_seen": self.total_tokens + epoch_tokens,
                        },
                    )

        # Update total counters
        self.total_loss += epoch_loss
        self.total_tokens += epoch_tokens

        # Calculate average loss per batch for reporting
        avg_loss = epoch_loss / len(dataloader)

        # Save checkpoint if this is the best epoch
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self._save_checkpoint(epoch, avg_loss)

        # Log epoch completion
        self.event_callback(
            "info",
            (
                f"Epoch {epoch + 1}/{self.hyperparams.n_epochs} complete. "
                f"Loss: {avg_loss:.4f} - Tokens: {epoch_tokens:,}"
            ),
            None,
        )

    def train(self, dataset: Dataset) -> None:
        """Main training loop."""
        # Create dataloader
        dataloader = self._create_dataloader(dataset, shuffle=True)

        # Setup optimizer and scheduler
        self.prepare_training(dataloader)
        self.event_callback("info", f"Starting training for {self.hyperparams.n_epochs} epochs", None)

        # Training loop
        for epoch in range(self.hyperparams.n_epochs):
            if self.check_cancel_callback() in ["cancelled", "cancelling"]:
                return
            self.run_epoch(dataloader, epoch)

            # Clear GPU cache after each epoch
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

    def evaluate(self, eval_dataset: Dataset, metric_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Run evaluation on dataset.

        Args:
            eval_dataset: Dataset to evaluate on
            metric_names: List of metric names to compute (e.g., ["bleu", "rouge"])
                         If None, only computes loss and perplexity

        Returns:
            Dict with evaluation metrics
        """
        self.model.eval()

        # Create eval dataloader
        eval_dataloader = self._create_dataloader(eval_dataset, shuffle=False)

        total_loss = 0.0
        total_batches = 0

        # Compute loss
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                total_batches += 1

        avg_loss = total_loss / total_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        metrics = {
            "eval_loss": avg_loss,
            "eval_perplexity": perplexity,
        }

        # TODO: Add HF evaluate metrics computation if metric_names provided
        # This would require generating text and comparing to references

        self.model.train()
        return metrics

    def save(self, output_dir: Path, base_model_id: str) -> Dict[str, Any]:
        """Save model and return training metadata."""
        # Calculate final metrics
        final_loss = self.total_loss / self.total_batches if self.total_batches > 0 else 0
        training_time = time.time() - self.start_time if self.start_time else 0

        # Log training completion with time
        self.event_callback(
            "info", f"Training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)", None
        )

        # Save metadata
        metadata = {
            "base_model": base_model_id,
            "trainer": "pytorch",
            "hyperparameters": self.hyperparams.model_dump(),
            "training_time": training_time,
            "global_steps": self.global_step,
            "final_loss": final_loss,
        }

        with open(output_dir / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Calculate perplexity from loss
        final_perplexity = torch.exp(torch.tensor(final_loss)).item() if final_loss > 0 else None

        return {
            "success": True,
            "final_loss": final_loss,
            "final_perplexity": final_perplexity,
            "steps": self.global_step,
            "tokens_processed": self.total_tokens,
            "model_path": str(output_dir),
        }
