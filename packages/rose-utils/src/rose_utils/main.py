"""Rose utilities CLI main entry point."""

import typer

from rose_utils.convert import convert_model
from rose_utils.quantize import quantize_model

app = typer.Typer(name="rose-utils", help="Rose model processing utilities")

app.command("convert", help="Convert sentence-transformers model to ONNX format")(convert_model)
app.command("quantize", help="Quantize a safetensors model")(quantize_model)


def main() -> None:
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
