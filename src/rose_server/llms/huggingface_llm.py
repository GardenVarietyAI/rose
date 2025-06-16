import logging
import os
import traceback
from typing import Any, Dict, List, Tuple

from rose_server.config import ServiceConfig
from rose_server.hf.loading import load_model_and_tokenizer
from rose_server.schemas.chat import ChatMessage

logger = logging.getLogger(__name__)


class HuggingFaceLLM:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("model_name", "huggingface_llm")
        self.model_path = config.get("model_path")
        if "model_name" not in self.config:
            self.config["model_name"] = self.model_name
        self.device_map = config.get("device_map", "auto")
        self._model = None
        self._tokenizer = None
        self._load_model()

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    def _load_model(self) -> bool:
        try:
            offload_dir = os.path.join(ServiceConfig.MODEL_OFFLOAD_DIR, self.model_name)
            self._model, self._tokenizer = load_model_and_tokenizer(
                model_id=self.model_name,
                model_path=self.model_path,
                device="auto",
                torch_dtype="auto",
                offload_dir=offload_dir,
                device_map=self.device_map,
            )
            return True
        except Exception as e:
            logger.error(f"Error loading {self.model_name} model: {str(e)}")
            logger.error(f"Model path was: {os.path.abspath(self.config.get('model_path', 'undefined'))}")
            logger.error(f"Configuration: {self.config}")
            traceback.print_exc()
            return False

    def format_messages(self, messages: List[ChatMessage]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            chat_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
            return self.tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
        prompt = ""
        for msg in messages:
            if msg.role == "system":
                prompt += f"System: {msg.content}\n\n"
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n\n"
        prompt += "Assistant: "
        return prompt

    def get_max_tokens(self) -> Tuple[int, int]:
        max_response_tokens = self.config.get("max_response_tokens", 512)
        n_ctx = self.config.get("n_ctx", 2048)
        max_prompt_tokens = n_ctx - max_response_tokens
        return max_prompt_tokens, max_response_tokens
