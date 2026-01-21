#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
import sys
import uuid
from pathlib import Path


def main() -> None:
    print("Create a new lens")
    print("-" * 40)

    at_name = input("@name (lowercase, alphanumeric only): ").strip().lower()
    if not at_name or not at_name.isalnum():
        print("Error: at_name must be alphanumeric")
        sys.exit(1)

    label = input("Label (display name): ").strip()
    if not label:
        print("Error: label is required")
        sys.exit(1)

    lens_uuid = str(uuid.uuid4())

    filename = f"{at_name}.md"
    output_path = Path(__file__).parent.parent / "lenses" / filename

    if output_path.exists():
        overwrite = input(f"{filename} already exists. Overwrite? (y/N): ").strip().lower()
        if overwrite != "y":
            print("Aborted")
            sys.exit(0)

    content = f"""---
uuid: {lens_uuid}
at_name: {at_name}
label: {label}
---

Write your lens system prompt here.
"""

    output_path.write_text(content, encoding="utf-8")
    print(f"\nCreated: {output_path}")
    print(f"UUID: {lens_uuid}")


if __name__ == "__main__":
    main()
