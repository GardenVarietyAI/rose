"""Model runner for evaluations - handles model inference via API."""

import logging
from typing import Any, Dict, Optional

from rose_worker.client import get_client

logger = logging.getLogger(__name__)


class ModelRunner:
    """Handles model inference for evaluations via API calls."""

    def __init__(self, model_name: str):
        """Initialize with a specific model."""
        self.model_name = model_name

    def generate(self, input_text: str, sampling_params: Optional[Dict[str, Any]] = None) -> str:
        """Generate model output for input via API.

        Args:
            input_text: Input prompt text
            sampling_params: Optional sampling parameters
                - temperature: float (0-2)
                - max_completion_tokens: int
                - top_p: float (0-1)
                - seed: int

        Returns:
            Generated text response
        """
        # Build request
        params = sampling_params or {}
        request_data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": input_text}],
            "temperature": params.get("temperature", 1.0),
            "max_tokens": params.get("max_completion_tokens", params.get("max_tokens", 2048)),
            "top_p": params.get("top_p", 1.0),
        }

        if "seed" in params:
            request_data["seed"] = params["seed"]

        logger.debug(f"Generating with model {self.model_name}, params: {params}")

        # Make API call via global client
        result = get_client().create_chat_completion(
            model=self.model_name,
            messages=[{"role": "user", "content": input_text}],
            temperature=params.get("temperature", 1.0),
            max_tokens=params.get("max_completion_tokens", params.get("max_tokens", 2048)),
            top_p=params.get("top_p", 1.0),
            seed=params.get("seed"),
            timeout=300.0,  # 5 minute timeout for long generations
        )

        return result["choices"][0]["message"]["content"]  # type: ignore[no-any-return]
