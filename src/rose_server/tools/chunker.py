"""Tool output chunking utilities for handling large outputs."""
import logging
from typing import Tuple
from rose_server.config import ResponseConfig
from rose_server.services import get_tokenizer_service
logger = logging.getLogger(__name__)
MAX_TOOL_OUTPUT_TOKENS = ResponseConfig.MAX_TOOL_OUTPUT_TOKENS
TRUNCATION_NOTICE_TOKENS = 100
MAX_OUTPUT_CHARS = ResponseConfig.MAX_OUTPUT_CHARS
LARGE_OUTPUT_CHARS = ResponseConfig.LARGE_OUTPUT_CHARS

def chunk_tool_output(output: str, max_tokens: int = MAX_TOOL_OUTPUT_TOKENS, model: str = "gpt-4") -> Tuple[str, bool]:
    """Chunk tool output to fit within token limits.

    Args:
        output: The raw tool output
        max_tokens: Maximum tokens allowed
        model: Model name for accurate token counting
    Returns:
        Tuple of (chunked_output, was_truncated)
    """
    output_len = len(output)
    if output_len < MAX_OUTPUT_CHARS:
        tokenizer_service = get_tokenizer_service()
        token_count = tokenizer_service.count_tokens(output, model)
        logger.debug(f"Tool output: {output_len} chars, {token_count} tokens")
        if token_count <= max_tokens:
            return output, False
    lines = output.split("\n")
    logger.info(f"Chunking large tool output: {output_len} chars, {len(lines)} lines")
    if len(lines) > 100:
        header_lines = lines[:30]
        footer_lines = lines[-20:]
        middle_notice = ["", f"... ({len(lines) - 50} lines omitted) ...", ""]
        truncated_lines = header_lines + middle_notice + footer_lines
        truncated_output = "\n".join(truncated_lines)
        truncation_notice = (
            f"\n\n[NOTE: Output truncated. Original had {len(lines)} lines "
            f"and approximately {len(output)} characters. "
            "Showing first 30 and last 20 lines.]"
        )
    else:
        truncated_output = ""
        current_tokens = 0
        for line in lines:
            line_tokens = tokenizer_service.count_tokens(line + "\n", model)
            if current_tokens + line_tokens > max_tokens - TRUNCATION_NOTICE_TOKENS:
                break
            truncated_output += line + "\n"
            current_tokens += line_tokens
        remaining_lines = len(lines) - len(truncated_output.split("\n"))
        truncation_notice = f"\n\n[NOTE: Output truncated at {current_tokens} tokens. {remaining_lines} lines omitted.]"
    return truncated_output + truncation_notice, True