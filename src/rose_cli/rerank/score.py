from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from rose_cli.utils import get_client

console = Console()


def score(
    query: str = typer.Argument(..., help="Search query"),
    documents: List[str] = typer.Argument(..., help="Documents to rerank"),
    top_n: Optional[int] = typer.Option(None, "--top-n", "-n", help="Number of results to return"),
    model: str = typer.Option("Qwen3-Reranker-0.6B-ONNX", "--model", "-m", help="Reranker model to use"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only output scores"),
) -> None:
    """Score and rerank documents based on relevance to query."""
    client = get_client()

    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": top_n,
        "return_documents": not quiet,
    }

    headers = {}
    if client.api_key:
        headers["Authorization"] = f"Bearer {client.api_key}"

    try:
        response = client._client.post(
            "/rerank",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

        if quiet:
            for result in data["results"]:
                print(f"{result['relevance_score']:.4f}")
        else:
            table = Table(title=f"Reranking: {query}")
            table.add_column("Rank", style="cyan")
            table.add_column("Score", style="magenta")
            table.add_column("Document", style="white")

            for i, result in enumerate(data["results"], 1):
                score_str = f"{result['relevance_score']:.4f}"
                doc = result.get("document", f"[Index {result['index']}]")
                if len(doc) > 60:
                    doc = doc[:57] + "..."
                table.add_row(str(i), score_str, doc)

            console.print(table)

    except Exception as e:
        if hasattr(e, "response"):
            console.print(f"[red]error: HTTP {e.response.status_code}[/red]", err=True)
        else:
            console.print(f"[red]error: {e}[/red]", err=True)
        raise typer.Exit(1)
