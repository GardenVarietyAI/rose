"""Qwen3 model family configurations."""

from dataclasses import dataclass


@dataclass
class QwenModelConfig:
    """Configuration for a Qwen3 model variant."""

    # Architecture (required fields first)
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    intermediate_size: int

    # Optimal runtime settings
    max_context_length: int
    optimal_batch_size: int
    max_response_tokens: int

    # Generation parameters
    temperature: float
    top_p: float
    repetition_penalty: float

    # Tool-calling optimizations
    tool_temperature: float
    tool_repetition_penalty: float
    requires_tool_reinforcement: bool

    # Fields with defaults
    vocab_size: int = 151936
    max_position_embeddings: int = 32768


def get_qwen_config(model_id: str) -> QwenModelConfig:
    """Get configuration for a specific Qwen3 model."""

    # Strip suffixes for matching base model
    base_id = model_id.split("-GGUF")[0].split("-ft-")[0]

    match base_id:
        case "Qwen--Qwen3-0.6B" | "Qwen--Qwen3-0.6B-Base":
            return QwenModelConfig(
                # Architecture
                hidden_size=1024,
                num_attention_heads=16,
                num_hidden_layers=28,
                num_key_value_heads=8,
                intermediate_size=3072,
                # Runtime
                max_context_length=2048,
                optimal_batch_size=1,
                max_response_tokens=512,
                # Generation
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.05,
                # Tools
                tool_temperature=0.1,
                tool_repetition_penalty=1.02,
                requires_tool_reinforcement=True,
            )

        case "Qwen--Qwen3-1.7B" | "Qwen--Qwen3-1.7B-Base" | "Qwen--Qwen3-1.5B":
            return QwenModelConfig(
                # Architecture
                hidden_size=2048,
                num_attention_heads=16,
                num_hidden_layers=28,
                num_key_value_heads=8,
                intermediate_size=6144,
                # Runtime
                max_context_length=4096,
                optimal_batch_size=2,
                max_response_tokens=1024,
                # Generation
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                # Tools
                tool_temperature=0.3,
                tool_repetition_penalty=1.05,
                requires_tool_reinforcement=True,
            )

        case "Qwen--Qwen3-4B":
            return QwenModelConfig(
                # Architecture
                hidden_size=2560,
                num_attention_heads=20,
                num_hidden_layers=40,
                num_key_value_heads=20,
                intermediate_size=6912,
                # Runtime
                max_context_length=8192,
                optimal_batch_size=4,
                max_response_tokens=2048,
                # Generation
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                # Tools
                tool_temperature=0.5,
                tool_repetition_penalty=1.1,
                requires_tool_reinforcement=False,
            )

        case "Qwen--Qwen3-7B" | "Qwen--Qwen3-8B":
            return QwenModelConfig(
                # Architecture
                hidden_size=3584,
                num_attention_heads=28,
                num_hidden_layers=28,
                num_key_value_heads=28,
                intermediate_size=18944,
                # Runtime
                max_context_length=16384,
                optimal_batch_size=8,
                max_response_tokens=4096,
                # Generation
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.15,
                # Tools
                tool_temperature=0.7,
                tool_repetition_penalty=1.1,
                requires_tool_reinforcement=False,
            )

        case _:
            # Default to 1.7B config for unknown models
            return get_qwen_config("Qwen--Qwen3-1.7B")


def should_use_tool_config(model_id: str, is_tool_request: bool) -> bool:
    """Determine if tool-specific config should be used."""
    if not is_tool_request:
        return False

    config = get_qwen_config(model_id)
    return config.requires_tool_reinforcement
