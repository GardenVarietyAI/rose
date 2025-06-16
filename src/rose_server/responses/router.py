"""OpenAI Responses API endpoint with streaming and tool detection."""
import json
import logging
import time
import traceback
import uuid
from typing import Optional

from fastapi import APIRouter, Body, Header, Request
from fastapi.responses import StreamingResponse

from rose_server.database import run_in_session
from rose_server.entities.threads import Message
from rose_server.events.formatters import ResponsesFormatter
from rose_server.events.generators import ResponsesGenerator
from rose_server.llms.huggingface_llm import HuggingFaceLLM
from rose_server.schemas.chat import ChatMessage
from rose_server.schemas.responses import ResponsesRequest
from rose_server.services import get_model_registry
from rose_server.tools import format_function_output
from rose_server.utils import extract_user_content

logger = logging.getLogger(__name__)
router = APIRouter(tags=["responses"])
@router.get("/v1/responses/{response_id}", response_model=None)

async def retrieve_response(response_id: str):
    """Retrieve a stored response by ID."""
    try:

        async def get_response_operation(session):
            response_msg = await session.get(Message, response_id)
            return response_msg
        response_msg = await run_in_session(get_response_operation, read_only=True)
        if not response_msg:
            return {"error": {"message": f"Response {response_id} not found", "type": "not_found", "code": None}}
        text_content = ""
        if isinstance(response_msg.content, list):
            for content_item in response_msg.content:
                if isinstance(content_item, dict) and content_item.get("type") == "text":
                    text_content = content_item.get("text", "")
                    break
        else:
            logger.warning(f"Unexpected content format for response {response_id}: {type(response_msg.content)}")
            text_content = str(response_msg.content) if response_msg.content else ""
        model_name = response_msg.meta.get("model", "unknown") if response_msg.meta else "unknown"
        return {
            "id": response_msg.id,
            "object": "response",
            "created": response_msg.created_at,
            "status": "completed",
            "output": [
                {
                    "id": f"msg_{response_msg.id.split('_')[1]}",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": text_content}],
                }
            ],
            "model": model_name,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": response_msg.meta.get("token_count", 0) if response_msg.meta else 0,
            },
        }
    except Exception as e:
        logger.error(f"Error retrieving response: {str(e)}")
        return {"error": {"message": str(e), "type": "server_error", "code": None}}
@router.post("/v1/responses", response_model=None)

async def create_response(
    request: ResponsesRequest = Body(...),
    http_request: Request = None,
    authorization: Optional[str] = Header(None),
    originator: Optional[str] = Header(None),
    user_agent: Optional[str] = Header(None, alias="User-Agent"),
    x_session_id: Optional[str] = Header(None, alias="X-Session-Id"),
):
    """Create a response using the model."""
    logger.info("=== RESPONSES API REQUEST ===")
    logger.info(f"Model: {request.model}")
    logger.info(f"Stream: {request.stream}")
    logger.info(f"Store: {request.store}")
    logger.info(f"Previous response ID: {request.previous_response_id}")
    logger.info(f"Input type: {type(request.input)}")
    logger.info(f"Headers - Originator: {originator}, User-Agent: {user_agent}, Session-ID: {x_session_id}")
    if http_request:
        logger.info(f"All headers: {dict(http_request.headers)}")
    if isinstance(request.input, list):
        logger.info(f"Input items: {len(request.input)}")
        for i, item in enumerate(request.input):
            if isinstance(item, dict):
                logger.info(f"  Item {i}: type={item.get('type')}")
                if item.get("type") == "function_call_output":
                    logger.info("    Found function_call_output!")
    try:
        messages = []
        base_instructions = (
            "You are a helpful assistant. Always provide substantive, well-formatted responses. "
            "Use clean markdown formatting with appropriate headers, simple bullet points, and code "
            "formatting for file/directory names. Avoid excessive nesting or overly complex formatting. "
            "Never respond with just whitespace or empty messages."
        )
        if request.instructions:
            messages.append(ChatMessage(role="system", content=f"{request.instructions}\n\n{base_instructions}"))
        else:
            messages.append(ChatMessage(role="system", content=base_instructions))
        if isinstance(request.input, str):
            messages.append(ChatMessage(role="user", content=request.input))
        elif isinstance(request.input, list):
            for item in request.input:
                if isinstance(item, dict):
                    item_type = item.get("type", "message")
                    if item_type == "message":
                        role = item.get("role", "user")
                        content = extract_user_content(item.get("content", ""))
                        if content:
                            messages.append(ChatMessage(role=role, content=content))
                    elif item_type == "function_call_output":
                        output = item.get("output", "")
                        try:
                            output_data = json.loads(output)
                            stdout = output_data.get("output", "")
                            exit_code = output_data.get("metadata", {}).get("exit_code", 0)
                            formatted_output = format_function_output(stdout, exit_code, request.model)
                            messages.append(ChatMessage(role="user", content=formatted_output))
                            logger.info(f"Added function output as user message: {formatted_output[:100]}...")
                        except Exception:
                            messages.append(ChatMessage(role="user", content=f"Tool output: {output}"))
        has_tool_output = False
        if isinstance(request.input, list):
            has_tool_output = any(
                item.get("type") == "function_call_output" for item in request.input if isinstance(item, dict)
            )
        enable_functions = bool(request.tools)
        logger.info("Multi-turn tool execution enabled")
        logger.info(f"Tools provided: {request.tools}")
        logger.info(f"Has tool output in input: {has_tool_output}")
        logger.info(f"Enable functions: {enable_functions}")
        if enable_functions:
            logger.info(f"Tools will be formatted by EventGeneratingLLM: {len(request.tools)} tools")
        elif has_tool_output and messages:
            no_tools_prompt = (
                "\n\nCRITICAL CONTEXT: You just executed a tool and received its output. The user is waiting "
                "for your analysis of this output. DO NOT execute more tools. DO NOT say you need more information. "
                "The tool output contains everything you need. Provide a complete, well-formatted response based on "
                "the tool output. Use clean markdown formatting with headers and simple bullet points. Avoid excessive "
                "nesting or complex indentation. Focus on explaining the PURPOSE and ORGANIZATION of what you found, "
                "not just listing items."
            )
            messages[0].content += no_tools_prompt
            logger.info("Added no-tools directive to system message for tool output response")
        logger.info(f"Final messages to send to model ({len(messages)} total):")
        for i, msg in enumerate(messages):
            logger.info(f"  Message {i}: role={msg.role}, content={msg.content[:100]}...")
        if request.stream:
            logger.info("Using event-based streaming")
            return await create_event_streaming_response(request, messages, enable_functions)
        return await create_event_complete_response(request, messages, enable_functions)
    except Exception as e:
        logger.error(f"Error in responses endpoint: {str(e)}")
        traceback.print_exc()
        return {"error": {"message": str(e), "type": "server_error", "code": None}}

async def create_event_streaming_response(
    request: ResponsesRequest, messages: list[ChatMessage], enable_functions: bool
) -> StreamingResponse:
    """Create streaming response using events and ResponsesFormatter."""

    async def generate():
        """Generate SSE events from LLM events."""
        try:
            registry = get_model_registry()
            if request.model not in registry.list_models():
                error_event = {
                    "type": "response.error",
                    "error": f"Model '{request.model}' not found",
                }
                yield f"data: {json.dumps(error_event)}\n\n"
                yield "data: [DONE]\n\n"
                return
            config = registry.get_model_config(request.model)
            if not config:
                error_event = {
                    "type": "response.error",
                    "error": f"Model '{request.model}' is not available.",
                }
                yield f"data: {json.dumps(error_event)}\n\n"
                yield "data: [DONE]\n\n"
                return
            try:
                base_llm = HuggingFaceLLM(config)
            except Exception as e:
                error_event = {
                    "type": "response.error",
                    "error": f"Failed to load model '{request.model}': {str(e)}",
                }
                yield f"data: {json.dumps(error_event)}\n\n"
                yield "data: [DONE]\n\n"
                return
            generator = ResponsesGenerator(base_llm)
            formatter = ResponsesFormatter()
            async for event in generator.generate_events(
                messages, enable_tools=enable_functions, tools=request.tools if enable_functions else None
            ):
                formatted = formatter.format_event(event)
                if formatted:
                    yield f"data: {json.dumps(formatted)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Event streaming error: {str(e)}")
            error_event = {
                "type": "response.error",
                "error": str(e),
            }
            yield f"data: {json.dumps(error_event)}\n\n"
            yield "data: [DONE]\n\n"
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )

async def create_event_complete_response(
    request: ResponsesRequest, messages: list[ChatMessage], enable_functions: bool
) -> dict:
    """Create complete (non-streaming) response from events."""
    try:
        registry = get_model_registry()
        if request.model not in registry.list_models():
            return {"error": {"message": f"Model '{request.model}' not found", "type": "model_error", "code": None}}
        config = registry.get_model_config(request.model)
        if not config:
            return {
                "error": {"message": f"Model '{request.model}' is not available", "type": "model_error", "code": None}
            }
        try:
            base_llm = HuggingFaceLLM(config)
        except Exception as e:
            return {
                "error": {
                    "message": f"Failed to load model '{request.model}': {str(e)}",
                    "type": "model_error",
                    "code": None,
                }
            }
        generator = ResponsesGenerator(base_llm)
        formatter = ResponsesFormatter()
        all_events = []
        async for event in generator.generate_events(
            messages, enable_tools=enable_functions, tools=request.tools if enable_functions else None
        ):
            all_events.append(event)
        complete_response = formatter.format_complete_response(all_events)
        complete_response["model"] = request.model
        if request.store:
            await store_response_messages(request, messages, complete_response)
        logger.info(f"Generated {len(all_events)} events for complete response")
        return complete_response
    except Exception as e:
        logger.error(f"Complete response error: {str(e)}")
        return {"error": {"message": f"Error generating response: {str(e)}", "type": "server_error", "code": None}}

async def store_response_messages(request: ResponsesRequest, messages: list[ChatMessage], response: dict):
    """Store request/response messages in database."""
    try:
        reply_text = ""
        for output_item in response.get("output", []):
            if output_item.get("type") == "message":
                content_list = output_item.get("content", [])
                for content_item in content_list:
                    if content_item.get("type") == "text":
                        reply_text = content_item.get("text", "")
                        break
        end_time = time.time()
        created_timestamp = response["created"]

        async def store_messages_operation(session):
            for msg in messages:
                if msg.role == "user":
                    user_message = Message(
                        id=f"msg_{uuid.uuid4().hex[:8]}",
                        thread_id=None,
                        role="user",
                        content=[{"type": "text", "text": msg.content}],
                        created_at=created_timestamp,
                        meta={"model": request.model},
                    )
                    session.add(user_message)
            assistant_message = Message(
                id=response["id"],
                thread_id=None,
                role="assistant",
                content=[{"type": "text", "text": reply_text}],
                created_at=created_timestamp,
                meta={
                    "model": request.model,
                    "token_count": response["usage"]["total_tokens"],
                    "response_time_ms": int((end_time - created_timestamp) * 1000),
                },
            )
            session.add(assistant_message)
            await session.commit()
            logger.info(f"Stored response {response['id']} to messages table")
        await run_in_session(store_messages_operation)
    except Exception as e:
        logger.error(f"Failed to store messages: {e}")