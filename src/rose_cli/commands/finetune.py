import json
import os
import tempfile
import time
from typing import Optional

import typer
from rich.table import Table

from ..utils import console, get_client, get_endpoint_url

app = typer.Typer()


@app.command()
def list(
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Base URL"),
    local: bool = typer.Option(True, "--local/--remote", "-l", help="Use local service"),
    table: bool = typer.Option(False, "--table", "-t", help="Show as table"),
):
    endpoint_url = get_endpoint_url(url, local)
    client = get_client(endpoint_url)
    try:
        jobs = client.fine_tuning.jobs.list()
        if table:
            table_obj = Table(title="Fine-tuning Jobs")
            table_obj.add_column("Job ID", style="cyan")
            table_obj.add_column("Status", style="green")
            table_obj.add_column("Base Model", style="yellow")
            table_obj.add_column("Fine-tuned Model", style="magenta")
            for job in jobs.data:
                status_style = "green" if job.status == "succeeded" else "yellow" if job.status == "running" else "red"
                table_obj.add_row(
                    job.id[:20] + "...",
                    f"[{status_style}]{job.status}[/{status_style}]",
                    job.model,
                    job.fine_tuned_model or "-",
                )
            console.print(table_obj)
        else:
            for job in jobs.data:
                console.print(f"{job.id}\t{job.status}\t{job.model}\t{job.fine_tuned_model or '-'}")
    except Exception as e:
        console.print(f"[red]error: {e}[/red]")


@app.command()
def create(
    file_id: str = typer.Option(..., "--file", "-f", help="Training file ID"),
    model: str = typer.Option("qwen-coder", "--model", "-m", help="Base model"),
    suffix: Optional[str] = typer.Option(None, "--suffix", help="Model suffix"),
    validation_file: Optional[str] = typer.Option(None, "--validation-file", "-v", help="Validation file ID"),
    method: Optional[str] = typer.Option(
        None, "--method", help='Training method JSON (e.g., \'{"type": "dpo", "hyperparameters": {"beta": 0.2}}\')'
    ),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Base URL"),
    local: bool = typer.Option(True, "--local/--remote", "-l", help="Use local service"),
):
    endpoint_url = get_endpoint_url(url, local)
    client = get_client(endpoint_url)
    try:
        method_obj = None
        if method:
            try:
                method_obj = json.loads(method)
            except json.JSONDecodeError as e:
                print(f"error: Invalid method JSON: {e}", file=typer.get_text_stream("stderr"))
                return
        job = client.fine_tuning.jobs.create(
            training_file=file_id, model=model, suffix=suffix, validation_file=validation_file, method=method_obj
        )
        print(job.id)
    except Exception as e:
        print(f"error: {e}", file=typer.get_text_stream("stderr"))


@app.command()
def status(
    job_id: str = typer.Argument(..., help="Job ID"),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Base URL"),
    local: bool = typer.Option(True, "--local/--remote", "-l", help="Use local service"),
):
    endpoint_url = get_endpoint_url(url, local)
    client = get_client(endpoint_url)
    try:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"{job.status}")
        if job.fine_tuned_model:
            print(f"{job.fine_tuned_model}")
    except Exception as e:
        print(f"error: {e}", file=typer.get_text_stream("stderr"))


@app.command()
def events(
    job_id: str = typer.Argument(..., help="Job ID"),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Base URL"),
    local: bool = typer.Option(True, "--local/--remote", "-l", help="Use local service"),
    tail: int = typer.Option(10, "--tail", "-n", help="Number of recent events"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow events live"),
):
    endpoint_url = get_endpoint_url(url, local)
    client = get_client(endpoint_url)
    try:
        if follow:
            console.print(f"[dim]Following events for {job_id} (Ctrl+C to stop)[/dim]")
            last_event_id = None
            try:
                while True:
                    events = client.fine_tuning.jobs.list_events(job_id)
                    new_events = []
                    for event in events.data:
                        if last_event_id is None or event.id != last_event_id:
                            new_events.append(event)
                        else:
                            break
                    if new_events:
                        for event in reversed(new_events):
                            timestamp = time.strftime("%H:%M:%S", time.localtime(event.created_at))
                            level_style = "green" if event.level == "info" else "red"
                            console.print(
                                f"[dim]{timestamp}[/dim] "
                                f"[{level_style}]{event.level.upper()}[/{level_style}]: {event.message}"
                            )
                        last_event_id = new_events[0].id
                    time.sleep(2)
            except KeyboardInterrupt:
                console.print("[dim]Stopped following events[/dim]")
        else:
            events = client.fine_tuning.jobs.list_events(job_id)
            for event in events.data[-tail:]:
                timestamp = time.strftime("%H:%M:%S", time.localtime(event.created_at))
                level_style = "green" if event.level == "info" else "red"
                console.print(
                    f"[dim]{timestamp}[/dim] [{level_style}]{event.level.upper()}[/{level_style}]: {event.message}"
                )
    except Exception as e:
        console.print(f"[red]error: {e}[/red]")


@app.command()
def pause(
    job_id: str = typer.Argument(..., help="Job ID"),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Base URL"),
    local: bool = typer.Option(True, "--local/--remote", "-l", help="Use local service"),
):
    endpoint_url = get_endpoint_url(url, local)
    client = get_client(endpoint_url)
    try:
        job = client.fine_tuning.jobs.pause(job_id)
        console.print(f"[green]✅ Job {job_id} paused successfully[/green]")
        console.print(f"Status: {job.status}")
    except Exception as e:
        console.print(f"[red]error: {e}[/red]")


@app.command()
def resume(
    job_id: str = typer.Argument(..., help="Job ID"),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Base URL"),
    local: bool = typer.Option(True, "--local/--remote", "-l", help="Use local service"),
):
    endpoint_url = get_endpoint_url(url, local)
    client = get_client(endpoint_url)
    try:
        job = client.fine_tuning.jobs.resume(job_id)
        console.print(f"[green]✅ Job {job_id} resumed successfully[/green]")
        console.print(f"Status: {job.status}")
    except Exception as e:
        console.print(f"[red]error: {e}[/red]")


@app.command()
def test(
    model: str = typer.Option("qwen-coder", "--model", "-m", help="Base model"),
    suffix: str = typer.Option("test", "--suffix", help="Model suffix"),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Base URL"),
    local: bool = typer.Option(True, "--local/--remote", "-l", help="Use local service"),
    monitor: bool = typer.Option(True, "--monitor/--no-monitor", help="Monitor job progress"),
):
    """Create and run a test fine-tuning job with sample data."""
    endpoint_url = get_endpoint_url(url, local)
    client = get_client(endpoint_url)
    sample_data = [
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there! How can I help you today?"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
            ]
        },
        {"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "2+2 equals 4."}]},
        {
            "messages": [
                {"role": "user", "content": "Tell me a joke"},
                {"role": "assistant", "content": "Why don't scientists trust atoms? Because they make up everything!"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What's the weather like?"},
                {
                    "role": "assistant",
                    "content": "I don't have access to current weather data, "
                    "but you can check your local weather service!",
                },
            ]
        },
    ]
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for example in sample_data:
                f.write(json.dumps(example) + "\n")
            temp_file = f.name
        console.print(f"[dim]Created sample training data: {len(sample_data)} examples[/dim]")
        console.print("[yellow]Uploading training file...[/yellow]")
        with open(temp_file, "rb") as f:
            file_obj = client.files.create(file=f, purpose="fine-tune")
        console.print(f"[green]✅ File uploaded: {file_obj.id}[/green]")
        console.print("[yellow]Creating fine-tuning job...[/yellow]")
        job = client.fine_tuning.jobs.create(training_file=file_obj.id, model=model, suffix=suffix)
        console.print(f"[green]✅ Job created: {job.id}[/green]")
        console.print(f"Status: {job.status}")
        if monitor:
            console.print("[dim]Monitoring job progress (Ctrl+C to stop)...[/dim]")
            try:
                while True:
                    time.sleep(3)
                    job = client.fine_tuning.jobs.retrieve(job.id)
                    console.print(f"Status: {job.status}")
                    if job.status in ["succeeded", "failed"]:
                        break
                console.print(f"[green]Final status: {job.status}[/green]")
                if hasattr(job, "fine_tuned_model") and job.fine_tuned_model:
                    console.print(f"Fine-tuned model: {job.fine_tuned_model}")
                if hasattr(job, "error") and job.error:
                    console.print(f"[red]Error: {job.error}[/red]")
            except KeyboardInterrupt:
                console.print(f"[dim]Stopped monitoring. Job ID: {job.id}[/dim]")
        else:
            console.print(f"Job ID: {job.id}")
            console.print("Use 'finetune status {job.id}' to check progress")
    except Exception as e:
        console.print(f"[red]error: {e}[/red]")
    finally:
        if "temp_file" in locals():
            os.unlink(temp_file)
