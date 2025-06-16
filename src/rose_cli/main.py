"""ROSE CLI - Main entry point."""
import typer

from .commands import (
    assistants,
    chat,
    cleanup,
    compare,
    complete,
    evals,
    files,
    finetune,
    models,
    responses,
    threads,
    upload,
)

app = typer.Typer(
    help="ROSE - Run your own LLM server",
    no_args_is_help=True,
)
app.add_typer(chat.app, name="chat", help="Chat with models")
app.add_typer(complete.app, name="complete", help="Generate completions")
app.add_typer(models.app, name="models", help="Model management")
app.add_typer(finetune.app, name="finetune", help="Fine-tuning operations")
app.add_typer(files.app, name="files", help="File operations")
app.add_typer(upload.app, name="upload", help="Upload files")
app.add_typer(compare.app, name="compare", help="Compare model responses")
app.add_typer(threads.app, name="threads", help="Thread management")
app.add_typer(assistants.app, name="assistants", help="Assistant management")
app.add_typer(cleanup.app, name="cleanup", help="Cleanup models, files, and jobs")
app.add_typer(responses.app, name="responses", help="Responses API operations")
app.add_typer(evals.app, name="eval", help="Evaluation operations")

def main():
    """Main entry point for the CLI."""
    app()
if __name__ == "__main__":
    main()