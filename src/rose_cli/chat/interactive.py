"""Interactive chat mode for ROSE CLI."""

from typing import Optional

import typer
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from rich.console import Console
from rich.markdown import Markdown

from rose_cli.utils import get_client

console = Console()


def interactive_chat(
    client: OpenAI,
    model: str,
    system: Optional[str] = None,
    stream: bool = True,
    markdown: bool = True,
) -> None:
    """Run an interactive chat session."""
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
                if stream:
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
                else:
                    # Non-streaming response
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                    )
                    response_text = response.choices[0].message.content or ""

                    if markdown:
                        console.print(Markdown(response_text))
                    else:
                        console.print(response_text)
                    print()  # Extra line for readability

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


def interactive(
    model: str = typer.Option("qwen2.5-0.5b", "--model", "-m", help="Model to use"),
    system: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream responses"),
    markdown: bool = typer.Option(True, "--markdown/--no-markdown", help="Render markdown in responses"),
) -> None:
    """Start an interactive chat session with a model."""
    client = get_client()

    try:
        # Test connection first
        models = client.models.list()
        model_names = [m.id for m in models.data]

        # Check if model exists
        if model not in model_names:
            console.print(f"[red]Model '{model}' not found.[/red]")
            console.print(f"Available models: {', '.join(model_names)}")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Failed to connect to ROSE server: {e}[/red]")
        console.print("[dim]Make sure the server is running on http://localhost:8004[/dim]")
        raise typer.Exit(1)

    interactive_chat(client, model, system, stream, markdown)
