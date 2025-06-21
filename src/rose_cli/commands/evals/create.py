from typing import Any

import typer
from rich.console import Console

from ...utils import get_client

console = Console()


def create_eval(
    name: str = typer.Option(..., "--name", "-n", help="Evaluation name"),
    file_id: str = typer.Option(..., "--file", "-f", help="Dataset file ID"),
    grader_type: str = typer.Option(
        "text_similarity", "--grader", "-g", help="Grader type: text_similarity, string_check, numeric_check"
    ),
    metric: str = typer.Option(
        "f1", "--metric", "-m", help="Evaluation metric: exact_match, f1, fuzzy_match, includes, numeric_exact"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only output the evaluation ID"),
):
    """Create a new evaluation."""
    client = get_client()
    try:
        # Build data source config
        data_source_config: Any = {
            "type": "stored_completions",
            "completion_tag_suffix": file_id,
        }

        # Build testing criteria based on grader type
        if grader_type == "string_check":
            criteria: Any = [
                {
                    "type": "string_check",
                    "name": "exact_match_check",
                    "input": "{{sample.output_text}}",
                    "reference": "{{item.expected}}",
                    "operation": "eq",
                }
            ]
        elif grader_type == "numeric_check":
            criteria = [
                {
                    "type": "text_similarity",
                    "name": "numeric_check",
                    "input": "{{sample.output_text}}",
                    "reference": "{{item.expected}}",
                    "evaluation_metric": "numeric_exact",
                    "pass_threshold": 0.5,
                }
            ]
        else:
            # Default text similarity with configurable metric
            criteria = [
                {
                    "type": "text_similarity",
                    "name": "similarity_check",
                    "input": "{{sample.output_text}}",
                    "reference": "{{item.expected}}",
                    "evaluation_metric": metric,
                    "pass_threshold": 0.8,
                }
            ]

        response = client.evals.create(
            name=name,
            data_source_config=data_source_config,
            testing_criteria=criteria,
        )

        if quiet:
            print(response.id)
        else:
            console.print(f"[green]Created evaluation: {response.id}[/green]")
            console.print(f"Name: {response.name}")
            console.print(f"Created: {response.created_at}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
