import httpx
import typer
from rich.console import Console
from rich.table import Table

from rose_cli.utils import BASE_URL

console = Console()


def test() -> None:
    """Run a quick test of the reranker with sample data."""
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital and largest city of France.",
        "Lyon is the third-largest city in France.",
        "The Eiffel Tower is located in Paris.",
        "France is a country in Western Europe.",
        "Marseille is a port city in southern France.",
    ]

    payload = {
        "query": query,
        "documents": documents,
        "top_n": 3,
        "return_documents": True,
    }

    try:
        console.print(f"[cyan]Testing reranker with query:[/cyan] {query}\n")

        with httpx.Client(timeout=30.0) as client:
            response = client.post(f"{BASE_URL}/rerank", json=payload)
            response.raise_for_status()

        data = response.json()

        table = Table(title="Test Results")
        table.add_column("Rank", style="cyan")
        table.add_column("Score", style="magenta")
        table.add_column("Document", style="white")

        for i, result in enumerate(data["results"], 1):
            score_str = f"{result['relevance_score']:.4f}"
            doc = result.get("document", f"[Index {result['index']}]")
            table.add_row(str(i), score_str, doc)

        console.print(table)
        console.print("\n[green]✓ Reranker test completed successfully[/green]")

    except httpx.HTTPStatusError as e:
        console.print(f"[red]error: HTTP {e.response.status_code}[/red]", err=True)
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]error: {e}[/red]", err=True)
        raise typer.Exit(1)
