# rose_server/services/huggingface_llm.py
import logging
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from peft import PeftModel
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from rose_core.models import cleanup_model_memory, get_tokenizer, load_hf_model, load_peft_model
from rose_server.schemas.chat import ChatMessage

logger = logging.getLogger(__name__)


class HuggingFaceLLM:
    """
    Thin wrapper that
     - loads / off-loads a Hugging Face model + tokenizer
     - converts a list[ChatMessage] to prompt string (chat-template aware)
     - exposes helper utilities (max token split, registry loading)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config
        self.model_name: str = config.get("model_name", "huggingface_llm")
        self.model_path: Optional[str] = config.get("model_path")
        self.device_map: Union[str, Dict[str, Union[int, str]]] = config.get("device_map", "auto")

        # ensure model_name is always present in config (needed by off-load dir)
        self.config.setdefault("model_name", self.model_name)

        self._model: Optional[Union[PreTrainedModel, PeftModel]] = None
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None

        self._load_model()

    @property
    def model(self) -> Optional[Union[PreTrainedModel, PeftModel]]:
        return self._model

    @property
    def tokenizer(self) -> Optional[PreTrainedTokenizerBase]:
        return self._tokenizer

    def _load_model(self) -> bool:
        """Load model/tokenizer once and cache them."""
        try:
            # torch_dtype "auto" means let the model decide
            torch_dtype = None if self.config.get("torch_dtype") == "auto" else self.config.get("torch_dtype")

            if self.model_path:
                # Check if it's a PEFT adapter or a full model
                adapter_config_path = Path(self.model_path) / "adapter_config.json"
                if adapter_config_path.exists():
                    # This is a PEFT/LoRA adapter
                    self._model = load_peft_model(
                        model_id=self.model_name, model_path=self.model_path, torch_dtype=torch_dtype
                    )
                else:
                    # This is a full model saved locally
                    self._model = load_hf_model(model_id=self.model_path, torch_dtype=torch_dtype)
                # Load tokenizer from the local path
                self._tokenizer = get_tokenizer(self.model_path)
            else:
                # This is a base model from HuggingFace hub
                self._model = load_hf_model(model_id=self.model_name, torch_dtype=torch_dtype)
                # Load tokenizer from HuggingFace hub
                self._tokenizer = get_tokenizer(self.model_name)
            return True
        except Exception as e:
            logger.error("Error loading %s: %s", self.model_name, e)
            logger.error("Model path: %s", os.path.abspath(self.model_path or "undefined"))
            logger.error("Config: %s", self.config)
            traceback.print_exc()
            return False

    def format_messages(self, messages: List[ChatMessage]) -> str:
        """
        Convert a sequence of ChatMessage objects into a single string prompt.

        If the tokenizer ships with a chat template we delegate completely to
        tokenizer.apply_chat_template(...), because it will take care of
        role/stop token handling.  Otherwise we fall back to a simple
        OpenAI-style plain-text format.

        Unsupported / non-text content (images, audio, tool calls, etc.) are
        silently skipped; you can choose to raise instead if that matters for
        your use-case.
        """
        if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            tmpl_msgs: List[Dict[str, str]] = []
            for msg in messages:
                text = self._extract_text(msg)
                # Ignore empty content (e.g. tool call placeholders)
                if text is None:
                    continue
                tmpl_msgs.append({"role": msg.role, "content": text})
            # When tokenize=False, apply_chat_template returns a string
            result = self.tokenizer.apply_chat_template(tmpl_msgs, tokenize=False, add_generation_prompt=True)
            assert isinstance(result, str), f"Expected string from apply_chat_template, got {type(result)}"
            return result

        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded")

        prompt_parts: List[str] = []
        role_map = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            # Treat function/tool/developer messages as plain system notes.
            "tool": "System",
            "function": "System",
            "developer": "System",
        }

        for msg in messages:
            text = self._extract_text(msg)
            if text is None:
                continue
            prompt_parts.append(f"{role_map.get(msg.role, 'User')}: {text}\n\n")

        prompt_parts.append("Assistant: ")
        return "".join(prompt_parts)

    @staticmethod
    def _extract_text(msg: ChatMessage) -> Optional[str]:
        """
        Pull plain text out of msg.content.
        - content is None then return None
        - content is str then return as-is
        - content is list[dict], 'text' or 'input_text' then grab first item with type
        - anything else then return None
        """
        if msg.content is None:
            return None
        if isinstance(msg.content, str):
            return msg.content

        # content is list[dict[str, Any]]
        for item in msg.content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type in ("text", "input_text") and "text" in item:
                    return str(item["text"])
                else:
                    logger.debug(f"Skipping non-text content part: type={item_type}")
        return None

    def get_max_tokens(self) -> Tuple[int, int]:
        """Compute how many tokens are left for the prompt vs. the completion."""
        max_response = self.config.get("max_response_tokens", 512)
        ctx_window = self.config.get("n_ctx", 2048)
        return ctx_window - max_response, max_response

    @classmethod
    async def load_model(cls, model_name: str, registry) -> "HuggingFaceLLM":
        """
        Factory that resolves model_name through the model registry,
        loads the model, and returns a ready-to-use instance.
        """
        available_models = await registry.list_models()
        if model_name not in available_models:
            raise ValueError(f"Model '{model_name}' not found in registry")

        config = await registry.get_model_config(model_name)
        if not config:
            raise ValueError(f"No config entry for model '{model_name}'")

        try:
            return cls(config)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}': {e}") from e

    def cleanup(self) -> None:
        """Clean up model resources and memory."""
        # Clean up GPU/memory with proper PEFT handling
        cleanup_model_memory(self._model)
        # Clear references to allow garbage collection
        self._model = None
        self._tokenizer = None
