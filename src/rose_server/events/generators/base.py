import asyncio
import json
import logging
import threading
import uuid
from contextlib import contextmanager
from typing import AsyncGenerator, List, Optional
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from ...llms.huggingface_llm import HuggingFaceLLM
from ...schemas.chat import ChatMessage
from ...tools import StreamingXMLDetector
from .. import (
    ResponseCompleted,
    ResponseStarted,
    TokenGenerated,
    ToolCallCompleted,
    ToolCallStarted,
)
logger = logging.getLogger(__name__)

class ForceStopAfterN(StoppingCriteria):
    """Hard-stop after N *new* tokens."""

    def __init__(self, max_new: int, prompt_len: int) -> None:
        self._max_new = max_new
        self._prompt_len = prompt_len

    def __call__(self, input_ids, scores, **_) -> bool:
        return len(input_ids[0]) - self._prompt_len >= self._max_new

class StopOnSpecialTokens(StoppingCriteria):
    """Abort when *any* special stop-token is seen."""

    def __init__(self, tokenizer) -> None:
        self._stop_ids = {tid for tid in (tokenizer.eos_token_id, tokenizer.pad_token_id) if tid is not None}
        logger.info("Stop-token IDs: %s", sorted(self._stop_ids))

    def __call__(self, input_ids, scores, **_) -> bool:
        return input_ids[0][-1].item() in self._stop_ids
@contextmanager

def background_generation(model, **gen_kwargs):
    """Run ``model.generate`` in a daemon thread and ensure cleanup."""
    t = threading.Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
    t.start()
    try:
        yield
    finally:
        t.join(timeout=1.0)
        if t.is_alive():
            logger.warning("Generation thread still alive - continuing anyway.")

class BaseEventGenerator:
    """Base class for event generators with common streaming logic."""

    def __init__(self, llm: HuggingFaceLLM):
        """Initialize with a HuggingFaceLLM instance."""
        self.llm = llm
        self.model_name = llm.model_name
        self.config = llm.config

    async def generate_events(
        self,
        messages: List[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_tools: bool = False,
        tools: Optional[List] = None,
        tool_choice: str | None = "auto",
        **kwargs,
    ) -> AsyncGenerator[
        ResponseStarted | TokenGenerated | ToolCallStarted | ToolCallCompleted | ResponseCompleted,
        None,
    ]:
        """Generate events from messages. Main entry point."""
        if not self.llm.model or not self.llm.tokenizer:
            yield ResponseCompleted(
                model_name=self.model_name,
                response_id=f"resp_error_{uuid.uuid4().hex[:8]}",
                total_tokens=0,
                finish_reason="error",
                output_tokens=0,
            )
            return
        prompt = await self.prepare_prompt(messages, enable_tools=enable_tools, tools=tools)
        inputs = self.llm.tokenizer(prompt, return_tensors="pt", truncation=True)
        if hasattr(self.llm.model, "device"):
            inputs = {k: v.to(self.llm.model.device) for k, v in inputs.items()}
        response_id = f"resp_{uuid.uuid4().hex[:16]}"
        max_new = max_tokens or self.config.get("max_response_tokens", 512)
        temp = temperature or self.config.get("temperature", 0.7)
        yield ResponseStarted(
            model_name=self.model_name,
            response_id=response_id,
            input_tokens=len(inputs["input_ids"][0]),
            max_tokens=max_new,
            temperature=temp,
        )
        async for event in self._stream_generation(inputs, response_id, max_new, temp, enable_tools):
            yield event

    async def prepare_prompt(
        self, messages: List[ChatMessage], enable_tools: bool = False, tools: Optional[List] = None
    ) -> str:
        """Prepare prompt from messages. Override in subclasses."""
        return self.llm.format_messages(messages)

    async def _stream_generation(
        self,
        inputs: dict,
        response_id: str,
        max_new: int,
        temperature: float,
        enable_tools: bool,
    ) -> AsyncGenerator:
        """Common streaming logic."""
        streamer = TextIteratorStreamer(self.llm.tokenizer, skip_prompt=True, skip_special_tokens=True)
        stop_list = StoppingCriteriaList(
            [
                ForceStopAfterN(max_new, len(inputs["input_ids"][0])),
                StopOnSpecialTokens(self.llm.tokenizer),
            ]
        )
        gen_kwargs = dict(
            inputs=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_new,
            temperature=temperature,
            top_p=self.config.get("top_p", 0.9),
            do_sample=True,
            pad_token_id=self.llm.tokenizer.pad_token_id or self.llm.tokenizer.eos_token_id,
            eos_token_id=self.llm.tokenizer.eos_token_id,
            repetition_penalty=1.1,
            length_penalty=1.0,
            streamer=streamer,
            stopping_criteria=stop_list,
        )
        detector = StreamingXMLDetector() if enable_tools else None
        accumulated = ""
        position = 0
        total_tokens = 0
        with background_generation(self.llm.model, **gen_kwargs):
            for token in streamer:
                total_tokens += 1
                if detector:
                    plain, call = detector.process_token(token)
                    if call:
                        call_id = f"call_{uuid.uuid4().hex[:16]}"
                        yield ToolCallStarted(
                            model_name=self.model_name,
                            function_name=call["tool"],
                            call_id=call_id,
                            arguments_so_far="",
                        )
                        yield ToolCallCompleted(
                            model_name=self.model_name,
                            function_name=call["tool"],
                            call_id=call_id,
                            arguments=json.dumps(call["arguments"]),
                        )
                    if plain:
                        accumulated += plain
                        yield TokenGenerated(
                            model_name=self.model_name,
                            token=plain,
                            token_id=total_tokens,
                            position=position,
                        )
                        position += 1
                else:
                    accumulated += token
                    yield TokenGenerated(
                        model_name=self.model_name,
                        token=token,
                        token_id=total_tokens,
                        position=position,
                    )
                    position += 1
                await asyncio.sleep(0)
        if detector:
            leftover = detector.flush()
            if leftover:
                accumulated += leftover
                total_tokens += 1
                yield TokenGenerated(
                    model_name=self.model_name,
                    token=leftover,
                    token_id=total_tokens,
                    position=position,
                )
        yield ResponseCompleted(
            model_name=self.model_name,
            response_id=response_id,
            total_tokens=total_tokens,
            finish_reason="stop",
            output_tokens=total_tokens,
        )