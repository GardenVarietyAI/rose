from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from rose_cli.utils import get_cohere_client

console = Console()


def score(
    query: str = typer.Argument(..., help="Search query"),
    documents: List[str] = typer.Argument(..., help="Documents to rerank"),
    top_n: Optional[int] = typer.Option(None, "--top-n", "-n", help="Number of results to return"),
    model: str = typer.Option("jina-reranker-v2-base-multilingual", "--model", "-m", help="Reranker model to use"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only output scores"),
) -> None:
    """Score and rerank documents based on relevance to query."""
    client = get_cohere_client()

    try:
        response = client.rerank(
            model=model,
            query=query,
            documents=documents,
            top_n=top_n,
            return_documents=True,
        )

        if quiet:
            for result in response.results:
                print(f"{result.relevance_score:.4f}")
        else:
            table = Table(title=f"Reranking: {query}")
            table.add_column("Rank", style="cyan")
            table.add_column("Score", style="magenta")
            table.add_column("Document", style="white")

            for i, result in enumerate(response.results, 1):
                score_str = f"{result.relevance_score:.4f}"
                doc = documents[result.index]
                if len(doc) > 60:
                    doc = doc[:57] + "..."
                table.add_row(str(i), score_str, doc)

            console.print(table)

    except Exception as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
