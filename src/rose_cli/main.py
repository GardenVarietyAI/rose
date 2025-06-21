"""ROSE CLI - Main entry point."""

import typer

from rose_cli import actors, assistants, chat, completions, evals, files, finetune, models, responses, runs, threads

app = typer.Typer(
    help="ROSE - Run your own LLM server",
    no_args_is_help=True,
)

app.add_typer(actors.app, name="actors", help="Explore actors")
app.add_typer(chat.app, name="chat", help="Chat with models")
app.add_typer(completions.app, name="completions", help="Generate completions")
app.add_typer(models.app, name="models", help="Model management")
app.add_typer(files.app, name="files", help="File operations")
app.add_typer(finetune.app, name="finetune", help="Fine-tuning operations")
app.add_typer(threads.app, name="threads", help="Thread management")
app.add_typer(assistants.app, name="assistants", help="Assistant management")
app.add_typer(runs.app, name="runs", help="Run management")
app.add_typer(responses.app, name="responses", help="Responses API operations")
app.add_typer(evals.app, name="evals", help="Evaluation operations")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
