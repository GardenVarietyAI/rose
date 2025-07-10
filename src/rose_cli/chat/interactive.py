"""Interactive chat mode for ROSE CLI."""

from typing import Optional

import typer
from openai.types.chat import ChatCompletionMessageParam
from rich.console import Console

from rose_cli.utils import get_client

console = Console()


def interactive(
    model: str = typer.Option("Qwen--Qwen2.5-1.5B-Instruct", "--model", "-m", help="Model to use"),
    system: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt"),
) -> None:
    """Start an interactive chat session with a model."""
    client = get_client()

    messages: list[ChatCompletionMessageParam] = []

    # Add system message if provided
    if system:
        messages.append({"role": "system", "content": system})
        console.print(f"[dim]System: {system}[/dim]")

    console.print(f"[bold green]ROSE Chat[/bold green] - Model: [cyan]{model}[/cyan]")
    console.print("[dim]Type 'exit' or 'quit' to end the session. Use Ctrl+C to interrupt.[/dim]\n")

    while True:
        try:
            # Get user input
            user_input = console.input("[bold blue]You:[/bold blue] ")

            # Check for exit commands
            if user_input.lower() in ["exit", "quit", "/exit", "/quit"]:
                console.print("\n[dim]Ending chat session...[/dim]")
                break

            # Skip empty inputs
            if not user_input.strip():
                continue

            # Add user message to history
            messages.append({"role": "user", "content": user_input})

            # Get assistant response
            console.print("\n[bold magenta]Assistant:[/bold magenta] ", end="")

            try:
                # Streaming response
                response_text = ""
                stream_response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                )

                for chunk in stream_response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        response_text += content
                        print(content, end="", flush=True)

                print("\n")  # New line after response

                # Add assistant response to history
                messages.append({"role": "assistant", "content": response_text})

            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")
                # Don't add failed response to history
                messages.pop()  # Remove the user message that caused the error

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Interrupted. Type 'exit' to quit or continue chatting.[/yellow]")
            continue
        except EOFError:
            # Handle Ctrl+D
            console.print("\n[dim]Ending chat session...[/dim]")
            break
