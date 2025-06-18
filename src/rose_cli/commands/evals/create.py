from typing import Any

import typer
from rich.console import Console

from ...utils import get_client

console = Console()


def create_eval(
    name: str = typer.Option(..., "--name", "-n", help="Evaluation name"),
    file_id: str = typer.Option(..., "--file", "-f", help="Dataset file ID"),
    criteria_type: str = typer.Option("text-similarity", "--criteria", "-c", help="Testing criteria type"),
):
    """Create a new evaluation."""
    client = get_client()
    try:
        # Build data source config
        data_source_config: Any = {
            "type": "stored_completions",
            "completion_tag_suffix": file_id,
        }

        # Build testing criteria based on type
        criteria: Any = [
            {
                "type": "text_similarity",
                "name": "similarity_check",
                "input": "{{item.prompt}}",
                "reference": "{{item.expected}}",
                "evaluation_metric": "cosine",
                "pass_threshold": 0.8,
            }
        ]

        response = client.evals.create(
            name=name,
            data_source_config=data_source_config,
            testing_criteria=criteria,
        )

        console.print(f"[green]Created evaluation: {response.id}[/green]")
        console.print(f"Name: {response.name}")
        console.print(f"Created: {response.created_at}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
