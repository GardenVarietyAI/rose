import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from ...utils import get_client

console = Console()


def create_eval(
    name: str,
    description: Optional[str] = typer.Option(None, "--description", "-d", help="Description of the evaluation"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", exists=True, help="JSONL file with evaluation data"),
    dataset: Optional[str] = typer.Option(None, "--dataset", help="Use a standard dataset (gsm8k, humaneval, mmlu)"),
):
    """Create a new evaluation definition."""
    client = get_client()
    metadata = {}
    if description:
        metadata["description"] = description
    if file:
        content = []
        with open(file, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                if "input" in item and "expected" in item:
                    content.append({"item": {"input": item["input"], "expected": item["expected"]}})
                else:
                    content.append({"item": item})
        data_source_config = {
            "type": "stored_completions",
            "metadata": {"source": "inline", "content": json.dumps(content)},
        }
    elif dataset:
        name = dataset
        data_source_config = {"type": "stored_completions", "metadata": {"source": dataset}}
    else:
        data_source_config = {"type": "stored_completions", "metadata": {"source": "default"}}

    # Handle specific dataset configurations
    if dataset == "gsm8k":
        metadata.update(
            {
                "name": "GSM8K Math Problems",
                "description": "Grade school math problems",
                "source": "gsm8k",
                "url": "https://github.com/openai/grade-school-math",
                "max_samples": 100,
            }
        )
    elif dataset == "humaneval":
        metadata.update(
            {
                "name": "HumanEval",
                "description": "Python programming problems",
                "source": "humaneval",
                "url": "https://github.com/openai/human-eval",
                "max_samples": 50,
            }
        )
    elif dataset == "mmlu":
        metadata.update(
            {
                "name": "MMLU",
                "description": "Massive Multitask Language Understanding",
                "source": "mmlu",
                "url": "https://github.com/hendrycks/test",
                "max_samples": 100,
            }
        )

    data = {
        "name": name,
        "data_source": data_source_config,
        "metadata": metadata,
    }
    try:
        response = client.post("/v1/evals", json=data)
        response.raise_for_status()
        result = response.json()
        console.print(f"[green]Created evaluation: {result['id']}[/green]")
        console.print(f"Name: {result['name']}")
        console.print(f"Created: {result['created_at']}")
    except Exception as e:
        console.print(f"[red]Error creating eval: {e}[/red]")
