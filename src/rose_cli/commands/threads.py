"""Thread management commands using OpenAI SDK."""
import json
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from ..utils import get_client
app = typer.Typer()
console = Console()
@app.command("create")

def create_thread(
    user_id: Optional[str] = typer.Option(None, help="User ID for the thread"),
    session_id: Optional[str] = typer.Option(None, help="Session ID"),
    conversation_type: Optional[str] = typer.Option(None, help="Type of conversation (chat, assistant, etc)"),
    source: Optional[str] = typer.Option(None, help="Source of the thread (api, web, cli)"),
    base_url: Optional[str] = typer.Option(None, help="Override base URL"),
):
    """Create a new thread."""
    client = get_client(base_url)
    metadata = {}
    if user_id:
        metadata["user_id"] = user_id
    if session_id:
        metadata["session_id"] = session_id
    if conversation_type:
        metadata["conversation_type"] = conversation_type
    if source:
        metadata["source"] = source
    try:
        thread = client.beta.threads.create(metadata=metadata)
        console.print(f"Created thread: [green]{thread.id}[/green]")
        if thread.metadata:
            console.print("Metadata:")
            for key, value in thread.metadata.items():
                console.print(f"   {key}: {value}")
    except Exception as e:
        console.print(f"Error: {e}", style="red")
@app.command("list")

def list_threads(
    limit: int = typer.Option(20, help="Number of threads to list"),
    base_url: Optional[str] = typer.Option(None, help="Override base URL"),
):
    """List threads."""
    client = get_client(base_url)
    try:
        import httpx
        base = base_url or "http://localhost:8004/v1"
        response = httpx.get(f"{base}/threads?limit={limit}")
        threads = response.json()
        table = Table(title="Threads")
        table.add_column("ID", style="cyan")
        table.add_column("Created", style="yellow")
        table.add_column("Messages", style="green")
        table.add_column("Metadata", style="blue")
        for thread in threads.get("data", []):
            metadata_str = json.dumps(thread.get("metadata", {}), indent=0).replace("\n", " ")
            table.add_row(
                thread["id"],
                str(thread["created_at"]),
                str(thread.get("message_count", 0)),
                metadata_str[:50] + "..." if len(metadata_str) > 50 else metadata_str,
            )
        console.print(table)
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
@app.command("get")

def get_thread(
    thread_id: str = typer.Argument(..., help="Thread ID to retrieve"),
    base_url: Optional[str] = typer.Option(None, help="Override base URL"),
):
    """Get a specific thread."""
    client = get_client(base_url)
    try:
        thread = client.beta.threads.retrieve(thread_id)
        console.print(f"Thread: [cyan]{thread.id}[/cyan]")
        console.print(f"Created: {thread.created_at}")
        if thread.metadata:
            console.print("Metadata:")
            for key, value in thread.metadata.items():
                console.print(f"   {key}: {value}")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
@app.command("add-message")

def add_message(
    thread_id: str = typer.Argument(..., help="Thread ID"),
    content: str = typer.Argument(..., help="Message content"),
    role: str = typer.Option("user", help="Message role (user/assistant)"),
    model_used: Optional[str] = typer.Option(None, help="Model used for the message"),
    token_count: Optional[str] = typer.Option(None, help="Token count"),
    response_time_ms: Optional[str] = typer.Option(None, help="Response time in ms"),
    base_url: Optional[str] = typer.Option(None, help="Override base URL"),
):
    """Add a message to a thread."""
    client = get_client(base_url)
    metadata = {}
    if model_used:
        metadata["model_used"] = model_used
    if token_count:
        metadata["token_count"] = token_count
    if response_time_ms:
        metadata["response_time_ms"] = response_time_ms
    try:
        message = client.beta.threads.messages.create(
            thread_id=thread_id, role=role, content=content, metadata=metadata
        )
        console.print(f"Added message: [green]{message.id}[/green]")
        console.print(f"Role: {message.role}")
        console.print(f"Content: {content[:100]}..." if len(content) > 100 else f"Content: {content}")
        if message.metadata:
            console.print("Metadata:")
            for key, value in message.metadata.items():
                console.print(f"   {key}: {value}")
    except Exception as e:
        console.print(f"Error: {e}", style="red")
@app.command("messages")

def list_messages(
    thread_id: str = typer.Argument(..., help="Thread ID"),
    limit: int = typer.Option(20, help="Number of messages to list"),
    order: str = typer.Option("desc", help="Sort order (asc/desc)"),
    base_url: Optional[str] = typer.Option(None, help="Override base URL"),
):
    """List messages in a thread."""
    client = get_client(base_url)
    try:
        messages = client.beta.threads.messages.list(thread_id=thread_id, limit=limit, order=order)
        console.print(f"Messages in thread [cyan]{thread_id}[/cyan]:")
        console.print()
        for msg in messages.data:
            console.print(f"[yellow]{msg.role}[/yellow] ({msg.id}):")
            for content in msg.content:
                if content.type == "text":
                    console.print(f"  {content.text.value}")
            if msg.metadata:
                metadata_str = " | ".join([f"{k}={v}" for k, v in msg.metadata.items()])
                console.print(f"  [dim]{metadata_str}[/dim]")
            console.print()
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
@app.command("delete")

def delete_thread(
    thread_id: str = typer.Argument(..., help="Thread ID to delete"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    base_url: Optional[str] = typer.Option(None, help="Override base URL"),
):
    """Delete a thread and all its messages."""
    if not confirm:
        confirm = typer.confirm(f"Are you sure you want to delete thread {thread_id}?")
        if not confirm:
            console.print("Cancelled")
            return
    client = get_client(base_url)
    try:
        response = client.beta.threads.delete(thread_id)
        console.print(f"Deleted thread: [red]{thread_id}[/red]")
    except Exception as e:
        console.print(f"Error: {e}", style="red")