import asyncio
import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers.generation.streamers import TextIteratorStreamer

from rose_server.events import (
    ResponseCompleted,
    ResponseStarted,
    TokenGenerated,
    ToolCallCompleted,
    ToolCallStarted,
)
from rose_server.llms.huggingface_llm import HuggingFaceLLM
from rose_server.schemas.chat import ChatMessage
from rose_server.tools import StreamingXMLDetector

logger = logging.getLogger(__name__)


class ForceStopAfterN(StoppingCriteria):
    """Hard-stop after N *new* tokens."""

    def __init__(self, max_new: int, prompt_len: int) -> None:
        self._max_new = max_new
        self._prompt_len = prompt_len

    def __call__(self, input_ids: Any, scores: Any, **_: Any) -> bool:
        return len(input_ids[0]) - self._prompt_len >= self._max_new


class StopOnSpecialTokens(StoppingCriteria):
    """Abort when *any* special stop-token is seen."""

    def __init__(self, tokenizer: Any) -> None:
        self._stop_ids = {tid for tid in (tokenizer.eos_token_id, tokenizer.pad_token_id) if tid is not None}
        logger.info("Stop-token IDs: %s", sorted(self._stop_ids))

    def __call__(self, input_ids: Any, scores: Any, **_: Any) -> bool:
        return input_ids[0][-1].item() in self._stop_ids


@contextmanager
def background_generation(model: Any, **gen_kwargs: Any) -> Any:
    """Run model.generate in a daemon thread and ensure cleanup."""
    t = threading.Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
    t.start()
    try:
        yield
    finally:
        t.join(timeout=5.0)
        if t.is_alive():
            logger.warning("Generation thread still alive after 5s, continuing anyway...")


class BaseEventGenerator:
    def __init__(self, llm: HuggingFaceLLM) -> None:
        self.llm = llm
        self.model_name: str = llm.model_name
        self.config: Dict[str, Any] = llm.config

    async def generate_events(
        self,
        messages: List[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_tools: bool = False,
        tools: Optional[List[Any]] = None,
        tool_choice: Optional[str] = "auto",
        **kwargs: Any,
    ) -> AsyncGenerator[
        Union[ResponseStarted, TokenGenerated, ToolCallStarted, ToolCallCompleted, ResponseCompleted],
        None,
    ]:
        """Main entrypoint: yield events for an LLM run."""
        if not self.llm.model or not self.llm.tokenizer:
            yield self._response_completed_zero()
            return

        prompt = await self.prepare_prompt(messages, enable_tools=enable_tools, tools=tools)
        inputs = self._encode_prompt(prompt)
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
        self, messages: List[ChatMessage], enable_tools: bool = False, tools: Optional[List[Any]] = None
    ) -> str:
        """Override to customize prompt construction."""
        formatted: Any = self.llm.format_messages(messages)
        return str(formatted)

    def _encode_prompt(self, prompt: str) -> Dict[str, Any]:
        inputs: Dict[str, Any] = self.llm.tokenizer(prompt, return_tensors="pt", truncation=True)
        if hasattr(self.llm.model, "device"):
            device = self.llm.model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        return inputs

    async def _stream_generation(
        self,
        inputs: Dict[str, Any],
        response_id: str,
        max_new: int,
        temperature: float,
        enable_tools: bool,
    ) -> AsyncGenerator[
        Union[ResponseStarted, TokenGenerated, ToolCallStarted, ToolCallCompleted, ResponseCompleted], None
    ]:
        start_time = time.time()
        streamer = TextIteratorStreamer(self.llm.tokenizer, skip_prompt=True, skip_special_tokens=True)
        stop_list = StoppingCriteriaList(
            [
                ForceStopAfterN(max_new, len(inputs["input_ids"][0])),
                StopOnSpecialTokens(self.llm.tokenizer),
            ]
        )
        gen_kwargs = self._make_gen_kwargs(inputs, max_new, temperature, streamer, stop_list)
        detector = StreamingXMLDetector() if enable_tools else None

        async for event in self._yield_tokens(streamer, gen_kwargs, detector, response_id, start_time):
            yield event

    def _make_gen_kwargs(
        self, inputs: Dict[str, Any], max_new: int, temperature: float, streamer: Any, stop_list: Any
    ) -> Dict[str, Any]:
        return dict(
            inputs=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_new,
            temperature=temperature,
            top_p=self.config.get("top_p", 0.9),
            do_sample=True,
            pad_token_id=self.llm.tokenizer.pad_token_id or self.llm.tokenizer.eos_token_id,
            eos_token_id=self.llm.tokenizer.eos_token_id,
            repetition_penalty=self.config.get("repetition_penalty", 1.1),
            length_penalty=self.config.get("length_penalty", 1.0),
            streamer=streamer,
            stopping_criteria=stop_list,
        )

    async def _yield_tokens(
        self, streamer: Any, gen_kwargs: Dict[str, Any], detector: Any, response_id: str, start_time: float
    ) -> AsyncGenerator[Union[TokenGenerated, ToolCallStarted, ToolCallCompleted, ResponseCompleted], None]:
        accumulated = ""
        position = 0
        total_tokens = 0

        with background_generation(self.llm.model, **gen_kwargs):
            for token in streamer:
                total_tokens += 1
                if detector:
                    tool_events, plain = self._handle_tool_streaming(token, detector, total_tokens)
                    for event in tool_events:
                        yield event
                    if plain:
                        accumulated += plain
                        yield TokenGenerated(
                            model_name=self.model_name,
                            token=plain,
                            token_id=total_tokens,
                            position=position,
                            logprob=None,
                        )
                        position += 1
                else:
                    accumulated += token
                    yield TokenGenerated(
                        model_name=self.model_name,
                        token=token,
                        token_id=total_tokens,
                        position=position,
                        logprob=None,
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
                    logprob=None,
                )

        completion_time = time.time() - start_time
        yield ResponseCompleted(
            model_name=self.model_name,
            response_id=response_id,
            total_tokens=total_tokens,
            finish_reason="stop",
            output_tokens=total_tokens,
            completion_time=completion_time,
        )

    def _handle_tool_streaming(
        self, token: str, detector: Any, total_tokens: int
    ) -> Tuple[List[Union[ToolCallStarted, ToolCallCompleted]], Optional[str]]:
        """Process token for tool calls, returns (events_list, plain_text)"""
        events: List[Union[ToolCallStarted, ToolCallCompleted]] = []
        plain, call = detector.process_token(token)
        if call:
            call_id = f"call_{uuid.uuid4().hex[:16]}"
            events.append(
                ToolCallStarted(
                    model_name=self.model_name,
                    function_name=call["tool"],
                    call_id=call_id,
                    arguments_so_far="",
                )
            )
            events.append(
                ToolCallCompleted(
                    model_name=self.model_name,
                    function_name=call["tool"],
                    call_id=call_id,
                    arguments=json.dumps(call["arguments"]),
                )
            )
        return events, plain

    def _response_completed_zero(self) -> ResponseCompleted:
        return ResponseCompleted(
            model_name=self.model_name,
            response_id=f"resp_error_{uuid.uuid4().hex[:8]}",
            total_tokens=0,
            finish_reason="stop",
            output_tokens=0,
            completion_time=0.0,
        )
