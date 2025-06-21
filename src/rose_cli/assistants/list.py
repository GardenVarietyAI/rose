import typer
from rich.console import Console
from rich.table import Table

from rose_cli.utils import get_client

console = Console()


def list_assistants(
    limit: int = typer.Option(20, help="Number of assistants to list"),
):
    """List assistants."""
    client = get_client()
    try:
        assistants = client.beta.assistants.list(limit=limit)
        if not assistants.data:
            console.print("No assistants found.")
            return

        table = Table(title="Assistants")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Model", style="yellow")
        table.add_column("Temperature", style="magenta")
        table.add_column("Tools", style="blue")

        for assistant in assistants.data:
            tools = []
            if assistant.tools:
                tools = [t.type for t in assistant.tools]
            tools_str = ", ".join(tools) if tools else "-"

            table.add_row(
                assistant.id,
                assistant.name or "-",
                assistant.model,
                str(assistant.temperature),
                tools_str,
            )

        console.print(table)
    except Exception as e:
        console.print(f"Error: {e}", style="red")
