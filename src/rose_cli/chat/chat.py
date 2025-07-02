from typing import Optional

import typer
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from rose_cli.utils import get_client


def _do_chat(
    client: OpenAI, model: str, prompt: str, system: Optional[str], stream: bool, name: Optional[str] = None
) -> None:
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
    prompt: Optional[str] = typer.Argument(None, help="Message to send (omit for interactive mode)"),
    model: str = typer.Option("qwen2.5-0.5b", "--model", "-m", help="Model to use"),
    system: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream response"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Start interactive session"),
) -> None:
    """Chat with models using a single message.

    For interactive sessions, use: rose chat interactive
    Or: rose chat --interactive
    """
    if ctx.invoked_subcommand is not None:
        return

    # Handle interactive mode
    if interactive or prompt is None:
        from .interactive import interactive_chat

        client = get_client()

        # Check connection
        try:
            models = client.models.list()
            model_names = [m.id for m in models.data]
            if model not in model_names:
                print(f"Error: Model '{model}' not found.", file=typer.get_text_stream("stderr"))
                print(f"Available models: {', '.join(model_names)}", file=typer.get_text_stream("stderr"))
                raise typer.Exit(1)
        except Exception as e:
            print(f"Error: Failed to connect to ROSE server: {e}", file=typer.get_text_stream("stderr"))
            print("Make sure the server is running on http://localhost:8004", file=typer.get_text_stream("stderr"))
            raise typer.Exit(1)

        interactive_chat(client, model, system, stream, markdown=True)
        return

    # Single message mode
    if not prompt:
        print("Error: Provide a message or use --interactive for chat mode", file=typer.get_text_stream("stderr"))
        raise typer.Exit(1)

    client = get_client()
    _do_chat(client, model, prompt, system, stream)
