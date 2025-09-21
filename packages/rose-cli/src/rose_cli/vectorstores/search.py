import typer
from rich.table import Table

from rose_cli.utils import console, get_client

app = typer.Typer()


@app.command()
def search_vectorstore(
    vector_store_id: str = typer.Argument(..., help="Vector store ID to search"),
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(5, "--limit", "-l", help="Maximum number of results"),
    show_content: bool = typer.Option(False, "--content", "-c", help="Show full content"),
) -> None:
    """Search a vector store for similar content."""
    client = get_client()
    try:
        # Perform search
        response = client.vector_stores.search(vector_store_id=vector_store_id, query=query, max_num_results=limit)

        # Display results
        if not response.data:
            console.print(f"[yellow]No results found for query: '{query}'[/yellow]")
            return

        console.print(f"[green]Found {len(response.data)} results for:[/green] '{query}'")
        # Display token usage if available
        if hasattr(response, "usage") and response.usage:
            if isinstance(response.usage, dict):
                prompt_tokens = response.usage.get("prompt_tokens", 0)
                total_tokens = response.usage.get("total_tokens", 0)
            else:
                prompt_tokens = response.usage.prompt_tokens
                total_tokens = response.usage.total_tokens
            console.print(f"[dim]Token usage: {prompt_tokens} prompt, {total_tokens} total[/dim]")
        console.print()

        table = Table(title="Search Results")
        table.add_column("Rank", style="cyan", width=4)
        table.add_column("File", style="green")
        table.add_column("Score", style="yellow", width=10)
        if show_content:
            table.add_column("Content", style="white", max_width=80)
        else:
            table.add_column("Preview", style="white", max_width=60)

        for i, result in enumerate(response.data, 1):
            score = f"{result.score:.3f}"
            filename = result.filename or result.file_id
            content_text = result.content[0].text if result.content and len(result.content) > 0 else ""

            if show_content:
                display_text = content_text
            else:
                display_text = content_text[:120] + "..." if len(content_text) > 120 else content_text

            table.add_row(str(i), filename, score, display_text)

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
