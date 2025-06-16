"""Evaluation commands for the Rose CLI."""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from ..utils import get_client
app = typer.Typer()
console = Console()
@app.command(name="create")

def create_eval(
    name: str,
    description: Optional[str] = typer.Option(None, "--description", "-d", help="Description of the evaluation"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", exists=True, help="JSONL file with evaluation data"),
    dataset: Optional[str] = typer.Option(None, "--dataset", help="Use a standard dataset (gsm8k, humaneval, mmlu)")
):
    """Create a new evaluation definition."""
    client = get_client()
    metadata = {}
    if description:
        metadata["description"] = description
    if file:
        content = []
        with open(file, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                if "input" in item and "expected" in item:
                    content.append({"item": {"input": item["input"], "expected": item["expected"]}})
                else:
                    content.append({"item": item})
        data_source_config = {
            "type": "stored_completions",
            "metadata": {
                "source": "inline",
                "content": json.dumps(content)
            }
        }
    elif dataset:
        name = dataset
        data_source_config = {
            "type": "stored_completions",
            "metadata": {
                "source": dataset
            }
        }
    else:
        data_source_config = {
            "type": "stored_completions",
            "metadata": {
                "source": "default"
            }
        }
    if dataset == "gsm8k":
        testing_criteria = [{
            "type": "string_check",
            "name": "exact_match",
            "input": "{{sample.output}}",
            "reference": "{{item.expected}}",
            "operation": "eq"
        }]
    elif dataset == "humaneval":
        testing_criteria = [{
            "type": "string_check",
            "name": "code_exact_match",
            "input": "{{sample.output}}",
            "reference": "{{item.expected}}",
            "operation": "eq"
        }]
    elif dataset == "mmlu":
        testing_criteria = [{
            "type": "string_check",
            "name": "multiple_choice",
            "input": "{{sample.output}}",
            "reference": "{{item.expected}}",
            "operation": "eq"
        }]
    else:
        testing_criteria = [{
            "type": "string_check",
            "name": "exact_match",
            "input": "{{sample.output}}",
            "reference": "{{item.expected}}",
            "operation": "eq"
        }]
    try:
        eval_obj = client.evals.create(
            name=name,
            data_source_config=data_source_config,
            testing_criteria=testing_criteria,
            metadata=metadata if metadata else None
        )
        console.print(f"[green]Created evaluation: {eval_obj.id}[/green]")
        panel = Panel(
            f"[bold]Name:[/bold] {eval_obj.name}\n"
            f"[bold]ID:[/bold] {eval_obj.id}\n"
            f"[bold]Created:[/bold] {eval_obj.created_at}\n"
            f"[bold]Criteria:[/bold] {len(getattr(eval_obj, 'testing_criteria', []))} tests",
            title="Evaluation Created",
            box=box.ROUNDED
        )
        console.print(panel)
    except Exception as e:
        console.print(f"[red]Error creating evaluation: {e}[/red]")
        raise typer.Exit(1)
@app.command(name="list")

def list_evals(
    limit: int = typer.Option(20, "--limit", "-l", help="Number of evaluations to show")
):
    """List evaluation definitions."""
    client = get_client()
    try:
        evals = client.evals.list(limit=limit)
        eval_list = list(evals.data) if hasattr(evals, 'data') else list(evals)
        if not eval_list:
            console.print("[yellow]No evaluations found[/yellow]")
            return
        table = Table(title="Evaluations", box=box.ROUNDED)
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Created", style="blue")
        table.add_column("Type", style="yellow")
        for eval_obj in eval_list:
            if hasattr(eval_obj, 'data_source_config'):
                eval_type = getattr(eval_obj.data_source_config, 'metadata', {}).get('source', 'unknown')
            else:
                eval_type = eval_obj.get("data_source_config", {}).get("metadata", {}).get("source", "unknown")
            table.add_row(
                eval_obj.id if hasattr(eval_obj, 'id') else eval_obj["id"],
                eval_obj.name if hasattr(eval_obj, 'name') else eval_obj["name"],
                str(eval_obj.created_at if hasattr(eval_obj, 'created_at') else eval_obj["created_at"]),
                eval_type
            )
        console.print(table)
    except Exception as e:
        console.print(f"[red]Error listing evaluations: {e}[/red]")
        raise typer.Exit(1)
@app.command(name="run")

def run_eval(
    eval_id: str,
    model: str = typer.Option(..., "--model", "-m", help="Model to evaluate"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Name for this run"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", help="Maximum number of samples to evaluate"),
    wait: bool = typer.Option(False, "--wait", "-w", help="Wait for completion")
):
    """Run an evaluation against a model.
    You can either provide an eval_id of an existing evaluation, or use a standard dataset name
    like 'gsm8k', 'humaneval', or 'mmlu' which will create the evaluation automatically.
    """
    client = get_client()
    try:
        if not eval_id.startswith("eval_"):
            try:
                evals = client.evals.list(limit=100)
                found_eval = None
                for e in evals.data:
                    if getattr(e, 'name', None) == eval_id:
                        found_eval = e
                        break
                if found_eval:
                    eval_id = found_eval.id
                    console.print(f"[yellow]Found evaluation: {eval_id}[/yellow]")
                else:
                    standard_datasets = ['gsm8k', 'humaneval', 'mmlu']
                    if eval_id.lower() in standard_datasets:
                        console.print(f"[yellow]Creating evaluation for {eval_id}...[/yellow]")
                        if eval_id.lower() == "gsm8k":
                            testing_criteria = [{
                                "type": "string_check",
                                "name": "exact_match",
                                "input": "{{sample.output}}",
                                "reference": "{{item.expected}}",
                                "operation": "eq"
                            }]
                        else:
                            testing_criteria = [{
                                "type": "string_check",
                                "name": "exact_match",
                                "input": "{{sample.output}}",
                                "reference": "{{item.expected}}",
                                "operation": "eq"
                            }]
                        eval_obj = client.evals.create(
                            name=eval_id.lower(),
                            data_source_config={
                                "type": "stored_completions",
                                "metadata": {"source": eval_id.lower()}
                            },
                            testing_criteria=testing_criteria
                        )
                        eval_id = eval_obj.id
                        console.print(f"[green]Created evaluation: {eval_id}[/green]")
                    else:
                        console.print(f"[red]Evaluation '{eval_id}' not found[/red]")
                        raise typer.Exit(1)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not look up eval by name: {e}[/yellow]")
        request_data = {
            "data_source": {
                "type": "stored_completions",
                "model": model
            }
        }
        if max_samples is not None:
            request_data["data_source"]["max_samples"] = max_samples
        if name:
            request_data["name"] = name
        else:
            request_data["name"] = f"{model} run"
        run = client.evals.runs.create(
            eval_id=eval_id,
            **request_data
        )
        console.print(f"[green]Started evaluation run: {run.id}[/green]")
        console.print(f"Model: [cyan]{model}[/cyan]")
        console.print(f"Status: [yellow]{run.status}[/yellow]")
        if max_samples:
            console.print(f"Max samples: [cyan]{max_samples}[/cyan]")
        if wait:
            console.print("\nWaiting for completion...")
            while True:
                time.sleep(2)
                run = client.evals.runs.retrieve(
                    eval_id=eval_id,
                    run_id=run.id
                )
                status = run.status
                console.print(f"Status: [yellow]{status}[/yellow]", end="\r")
                if status in ["completed", "failed"]:
                    break
            console.print()
            if status == "completed":
                results = getattr(run, 'results', {})
                panel = Panel(
                    f"[bold]Samples:[/bold] {results.get('total_samples', 0)}\n"
                    f"[bold]Exact Match:[/bold] {results.get('exact_match', 0):.2%}\n"
                    f"[bold]F1 Score:[/bold] {results.get('f1', 0):.2%}\n"
                    f"[bold]Substring:[/bold] {results.get('substring_match', 0):.2%}",
                    title=f"Results for {model}",
                    box=box.ROUNDED
                )
                console.print(panel)
            else:
                error_msg = getattr(run, 'error', 'Unknown error')
                console.print(f"[red]Evaluation failed: {error_msg}[/red]")
    except Exception as e:
        console.print(f"[red]Error running evaluation: {e}[/red]")
        raise typer.Exit(1)
@app.command(name="results")

def show_results(
    eval_id: str,
    run_id: str,
    samples: bool = typer.Option(False, "--samples", "-s", help="Show individual sample results")
):
    """Show results from an evaluation run."""
    client = get_client()
    try:
        run = client.evals.runs.retrieve(
            eval_id=eval_id,
            run_id=run_id
        )
        console.print(f"\n[bold]Evaluation Run: {run.name}[/bold]")
        console.print(f"Model: [cyan]{run.model}[/cyan]")
        console.print(f"Status: [yellow]{run.status}[/yellow]")
        if run.status == "completed" and hasattr(run, 'results'):
            results = run.results
            table = Table(title="Overall Results", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Score", style="green")
            table.add_row("Total Samples", str(results.get("total_samples", 0)))
            table.add_row("Exact Match", f"{results.get('exact_match', 0):.2%}")
            table.add_row("F1 Score", f"{results.get('f1', 0):.2%}")
            table.add_row("Substring Match", f"{results.get('substring_match', 0):.2%}")
            if "number_match" in results:
                table.add_row("Number Match", f"{results.get('number_match', 0):.2%}")
            console.print(table)
            if samples:
                try:
                    samples_iter = client.evals.runs.output_items.list(
                        eval_id=eval_id,
                        run_id=run_id,
                        limit=10
                    )
                    sample_list = list(samples_iter.data) if hasattr(samples_iter, 'data') else list(samples_iter)[:10]
                    console.print("\n[bold]Sample Results:[/bold]")
                    for i, sample in enumerate(sample_list, 1):
                        scores = getattr(sample, 'scores', {})
                        exact = "✓" if scores.get("exact_match", 0) > 0.5 else "✗"
                        console.print(f"\n[cyan]Sample {i}:[/cyan] {exact}")
                        console.print(f"  Input: {getattr(sample, 'input', '')[:100]}...")
                        console.print(f"  Expected: {getattr(sample, 'expected', '')[:100]}...")
                        console.print(f"  Got: {getattr(sample, 'output', '')[:100]}...")
                except Exception as e:
                    console.print(f"[yellow]Could not retrieve samples: {e}[/yellow]")
    except Exception as e:
        console.print(f"[red]Error showing results: {e}[/red]")
        raise typer.Exit(1)
@app.command(name="delete")

def delete_eval(
    eval_id: str,
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation")
):
    """Delete an evaluation definition."""
    if not confirm:
        confirm = typer.confirm("Are you sure you want to delete this evaluation?")
        if not confirm:
            console.print("[yellow]Deletion cancelled[/yellow]")
            return
    client = get_client()
    try:
        client.evals.delete(eval_id)
        console.print(f"[green]Deleted evaluation: {eval_id}[/green]")
    except Exception as e:
        console.print(f"[red]Error deleting evaluation: {e}[/red]")
        raise typer.Exit(1)