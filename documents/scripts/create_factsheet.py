#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
import sys
import uuid
from pathlib import Path


def main() -> None:
    print("Create a new factsheet")
    print("-" * 40)

    tag = input("#tag (lowercase, alphanumeric only): ").strip().lower()
    if not tag or not tag.isalnum():
        print("Error: tag must be alphanumeric")
        sys.exit(1)

    title = input("Title: ").strip()
    if not title:
        print("Error: title is required")
        sys.exit(1)

    factsheet_uuid = str(uuid.uuid4())

    filename = f"{tag}.md"
    output_path = Path(__file__).parent.parent / "factsheets" / filename

    if output_path.exists():
        overwrite = input(f"{filename} already exists. Overwrite? (y/N): ").strip().lower()
        if overwrite != "y":
            print("Aborted")
            sys.exit(0)

    content = f"""---
uuid: {factsheet_uuid}
tag: {tag}
title: {title}
---

# {title}

Write your factsheet content here.
"""

    output_path.write_text(content, encoding="utf-8")
    print(f"\nCreated: {output_path}")
    print(f"UUID: {factsheet_uuid}")


if __name__ == "__main__":
    main()
