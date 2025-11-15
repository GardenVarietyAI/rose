"""ROSE CLI - Main entry point."""

import typer

from rose_cli import actors, models, responses

app = typer.Typer(
    help="ROSE - Run your own LLM server",
    no_args_is_help=True,
)

app.add_typer(actors.app, name="actors", help="Explore actors")
app.add_typer(models.app, name="models", help="Model management")
app.add_typer(responses.app, name="responses", help="Responses API operations")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
