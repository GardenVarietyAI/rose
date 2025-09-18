from typing import Any, Optional

import typer
from openai.types.chat import ChatCompletionMessageParam
from rich.console import Console

from rose_cli.chat.interactive import interactive as interactive_func
from rose_cli.utils import get_client

console = Console()


def chat(
    ctx: typer.Context,
    prompt: Optional[str] = typer.Argument(None, help="Message to send (omit for interactive mode)"),
    model: str = typer.Option("Qwen/Qwen2.5-1.5B-Instruct", "--model", "-m", help="Model to use"),
    system: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Start interactive session"),
    temperature: Optional[float] = typer.Option(None, "--temperature", "-t", help="Sampling temperature (0.0-2.0)"),
    seed: Optional[int] = typer.Option(None, "--seed", min=0, help="Seed for deterministic generation (non-negative)"),
) -> None:
    """Chat with models using a single message.

    For interactive sessions, use: rose chat interactive
    Or: rose chat --interactive
    """

    if ctx.invoked_subcommand is not None:
        return

    # Handle interactive mode
    if interactive or prompt is None:
        interactive_func(model=model, system=system)
        return

    # Single message mode
    messages: list[ChatCompletionMessageParam] = []
    if system:
        system_msg: ChatCompletionMessageParam = {"role": "system", "content": system}
        messages.append(system_msg)

    user_message: ChatCompletionMessageParam = {"role": "user", "content": prompt}
    messages.append(user_message)

    try:
        flat_model_name = model.replace("/", "--")
        kwargs: dict[str, Any] = {
            "model": flat_model_name,
            "messages": messages,
        }

        if temperature is not None:
            kwargs["temperature"] = temperature
        if seed is not None:
            kwargs["seed"] = seed

        client = get_client()
        response = client.chat.completions.create(**kwargs)

        # Print the message content
        console.print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}", file=typer.get_text_stream("stderr"))
