from rich.console import Console
from rich.prompt import Confirm

from ...utils import get_client

console = Console()


def delete_eval(eval_id: str):
    """Delete an evaluation and all its runs."""
    client = get_client()
    if not Confirm.ask(f"Are you sure you want to delete evaluation {eval_id} and all its runs?"):
        console.print("Cancelled.")
        return
    try:
        response = client.delete(f"/v1/evals/{eval_id}")
        response.raise_for_status()
        console.print(f"[red]Deleted evaluation: {eval_id}[/red]")
    except Exception as e:
        console.print(f"[red]Error deleting eval: {e}[/red]")
