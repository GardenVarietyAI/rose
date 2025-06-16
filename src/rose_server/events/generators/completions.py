"""Completions event generator for text completion API."""

import time
from typing import List, Optional, Union

from ...schemas.chat import ChatMessage
from .. import TokenGenerated
from .base import BaseEventGenerator


class CompletionsGenerator(BaseEventGenerator):
    """Generate events for text completions (non-chat format).

    This handles the legacy completions API that takes raw prompts
    instead of chat messages.
    """

    async def generate_prompt_events(
        self,
        prompt: Union[str, List[str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        echo: bool = False,
        **kwargs,
    ):
        """Generate events from a raw prompt.
        Args:
            prompt: Text prompt or list of prompts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            echo: Whether to include prompt in response
            **kwargs: Additional generation parameters
        """
        if isinstance(prompt, list):
            if not prompt:
                prompt = ""
            else:
                prompt = prompt[0]
        self._echo_prompt = prompt if echo else None
        messages = [ChatMessage(role="user", content=prompt)]
        if echo:
            yield TokenGenerated(token=prompt, timestamp=time.time(), model_name=self.llm.model_name)
        async for event in self.generate_events(
            messages, temperature=temperature, max_tokens=max_tokens, enable_tools=False, **kwargs
        ):
            yield event

    async def prepare_prompt(
        self, messages: List[ChatMessage], enable_tools: bool = False, tools: Optional[List] = None
    ) -> str:
        """Prepare prompt for completions.
        Completions API uses a simpler format without roles.
        """
        return self.llm.format_messages(messages)
