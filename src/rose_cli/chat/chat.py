from typing import Optional, cast

import typer
from openai import OpenAI
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam

from rose_cli.utils import get_client


def _do_chat(client: OpenAI, model: str, prompt: str, system: Optional[str], stream: bool, name: Optional[str] = None):
    messages: list[ChatCompletionMessageParam] = []
    if system:
        system_msg: ChatCompletionMessageParam = {"role": "system", "content": system}
        messages.append(system_msg)

    # Build user message with optional name
    user_message: ChatCompletionMessageParam = {"role": "user", "content": prompt}
    if name:
        user_message["name"] = name  # type: ignore[typeddict-unknown-key]

    messages.append(user_message)

    try:
        if stream:
            stream_response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )
            for chunk in stream_response:
                chunk = cast(ChatCompletionChunk, chunk)
                if chunk.choices and chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print()
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}", file=typer.get_text_stream("stderr"))


def chat(
    ctx: typer.Context,
    prompt: str = typer.Argument(..., help="Message to send"),
    model: str = typer.Option("qwen-coder", "--model", "-m", help="Model to use"),
    system: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream response"),
):
    """Chat with models using a single message."""
    if ctx.invoked_subcommand is not None:
        return

    client = get_client()
    _do_chat(client, model, prompt, system, stream)
