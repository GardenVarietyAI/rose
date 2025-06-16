"""Assistant management commands using OpenAI SDK."""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ..utils import get_client

app = typer.Typer()
console = Console()


@app.command("create")
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


@app.command("list")
def list_assistants(
    limit: int = typer.Option(20, help="Number of assistants to list"),
    base_url: Optional[str] = typer.Option(None, help="Override base URL"),
):
    """List assistants."""
    client = get_client(base_url)
    try:
        assistants = client.beta.assistants.list(limit=limit)
        table = Table(title="Assistants")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Model", style="yellow")
        table.add_column("Tools", style="blue")
        table.add_column("Temperature", style="magenta")
        for assistant in assistants.data:
            tools_str = ", ".join([t.type for t in assistant.tools]) if assistant.tools else "None"
            table.add_row(
                assistant.id, assistant.name or "Unnamed", assistant.model, tools_str, str(assistant.temperature)
            )
        console.print(table)
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@app.command("get")
def get_assistant(
    assistant_id: str = typer.Argument(..., help="Assistant ID to retrieve"),
    base_url: Optional[str] = typer.Option(None, help="Override base URL"),
):
    """Get a specific assistant."""
    client = get_client(base_url)
    try:
        assistant = client.beta.assistants.retrieve(assistant_id)
        console.print(f"Assistant: [cyan]{assistant.id}[/cyan]")
        console.print(f"Name: {assistant.name}")
        console.print(f"Description: {assistant.description or 'None'}")
        console.print(f"Model: {assistant.model}")
        console.print(f"Instructions: {assistant.instructions or 'None'}")
        console.print(f"Temperature: {assistant.temperature}")
        console.print(f"Top P: {assistant.top_p}")
        if assistant.tools:
            console.print("Tools:")
            for tool in assistant.tools:
                console.print(f"   - {tool.type}")
    except Exception as e:
        console.print(f"Error: {e}", style="red")


@app.command("update")
def update_assistant(
    assistant_id: str = typer.Argument(..., help="Assistant ID to update"),
    name: Optional[str] = typer.Option(None, help="New name"),
    model: Optional[str] = typer.Option(None, help="New model"),
    instructions: Optional[str] = typer.Option(None, help="New instructions"),
    temperature: Optional[float] = typer.Option(None, help="New temperature"),
    base_url: Optional[str] = typer.Option(None, help="Override base URL"),
):
    """Update an assistant."""
    client = get_client(base_url)
    update_data = {}
    if name is not None:
        update_data["name"] = name
    if model is not None:
        update_data["model"] = model
    if instructions is not None:
        update_data["instructions"] = instructions
    if temperature is not None:
        update_data["temperature"] = temperature
    try:
        assistant = client.beta.assistants.update(assistant_id=assistant_id, **update_data)
        console.print(f"Updated assistant: [green]{assistant.id}[/green]")
        console.print(f"Name: {assistant.name}")
        console.print(f"Model: {assistant.model}")
        console.print(f"Temperature: {assistant.temperature}")
    except Exception as e:
        console.print(f"Error: {e}", style="red")


@app.command("delete")
def delete_assistant(
    assistant_id: str = typer.Argument(..., help="Assistant ID to delete"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    base_url: Optional[str] = typer.Option(None, help="Override base URL"),
):
    """Delete an assistant."""
    if not confirm:
        confirm = typer.confirm(f"Are you sure you want to delete assistant {assistant_id}?")
        if not confirm:
            console.print("Cancelled")
            return
    client = get_client(base_url)
    try:
        response = client.beta.assistants.delete(assistant_id)
        console.print(f"Deleted assistant: [red]{assistant_id}[/red]")
    except Exception as e:
        console.print(f"Error: {e}", style="red")
