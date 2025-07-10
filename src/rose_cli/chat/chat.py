from typing import Any, Optional

import typer
from openai.types.chat import ChatCompletionMessageParam
from rich.console import Console
from rich.table import Table

from rose_cli.chat.interactive import interactive as interactive_func
from rose_cli.utils import get_client

console = Console()


def chat(
    ctx: typer.Context,
    prompt: Optional[str] = typer.Argument(None, help="Message to send (omit for interactive mode)"),
    model: str = typer.Option("Qwen--Qwen2.5-1.5B-Instruct", "--model", "-m", help="Model to use"),
    system: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Start interactive session"),
    logprobs: bool = typer.Option(False, "--logprobs", help="Return log probabilities of tokens"),
    top_logprobs: Optional[int] = typer.Option(None, "--top-logprobs", help="Number of top logprobs to return (0-5)"),
    temperature: Optional[float] = typer.Option(None, "--temperature", "-t", help="Sampling temperature (0.0-2.0)"),
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
        # Build kwargs for completion
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if logprobs:
            kwargs["logprobs"] = logprobs
            if top_logprobs is not None:
                kwargs["top_logprobs"] = top_logprobs

        client = get_client()
        response = client.chat.completions.create(**kwargs)

        # Print the message content
        console.print(response.choices[0].message.content)

        # Print logprobs if present
        if logprobs and response.choices[0].logprobs:
            console.print()

            # Create a simple table for token probabilities
            table = Table(show_header=True, header_style="bold")
            table.add_column("Token", style="cyan")
            table.add_column("Log Prob", justify="right")

            if top_logprobs:
                table.add_column("Top Alternative")

            for token_data in response.choices[0].logprobs.content or []:
                row = [repr(token_data.token), f"{token_data.logprob:.4f}"]

                # Add the best alternative if requested
                if top_logprobs and token_data.top_logprobs:
                    # Find the best alternative that isn't the selected token
                    best_alt = None
                    for alt in token_data.top_logprobs:
                        if alt.token != token_data.token:
                            best_alt = alt
                            break

                    if best_alt:
                        row.append(f"{repr(best_alt.token)}: {best_alt.logprob:.4f}")
                    else:
                        row.append("")

                table.add_row(*row)

            console.print(table)
    except Exception as e:
        print(f"Error: {e}", file=typer.get_text_stream("stderr"))
