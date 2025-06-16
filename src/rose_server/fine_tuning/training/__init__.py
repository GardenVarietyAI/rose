"""Synchronous training implementations for fine-tuning jobs.
This submodule contains all sync code for actual training execution,
while the parent fine_tuning module provides the async OpenAI-compatible API layer.
"""
from .hf_trainer import HFTrainer

__all__ = [
    "HFTrainer",
]