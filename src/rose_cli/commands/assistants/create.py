from typing import Optional

import typer
from rich.console import Console

from ...utils import get_client

console = Console()


def create_assistant(
    name: str = typer.Argument(..., help="Assistant name"),
    model: str = typer.Option("gpt-4", help="Model to use"),
    instructions: Optional[str] = typer.Option(None, help="System instructions"),
    description: Optional[str] = typer.Option(None, help="Assistant description"),
    temperature: float = typer.Option(0.7, help="Temperature (0.0-2.0)"),
    code_interpreter: bool = typer.Option(False, help="Enable code interpreter"),
    file_search: bool = typer.Option(False, help="Enable file search"),
    base_url: Optional[str] = typer.Option(None, help="Override base URL"),
):
    """Create a new assistant."""
    client = get_client(base_url)
    tools = []
    if code_interpreter:
        tools.append({"type": "code_interpreter"})
    if file_search:
        tools.append({"type": "file_search"})
    try:
        assistant = client.beta.assistants.create(
            name=name,
            model=model,
            instructions=instructions,
            description=description,
            tools=tools,
            temperature=temperature,
        )
        console.print(f"Created assistant: [green]{assistant.id}[/green]")
        console.print(f"Name: {assistant.name}")
        console.print(f"Model: {assistant.model}")
        console.print(f"Temperature: {assistant.temperature}")
        if assistant.tools:
            console.print(f"Tools: {', '.join([t.type for t in assistant.tools])}")
    except Exception as e:
        console.print(f"Error: {e}", style="red")
