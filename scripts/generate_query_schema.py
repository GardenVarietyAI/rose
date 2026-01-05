#!/usr/bin/env python3

import json
from pathlib import Path

from rose_server.schemas.query import QueryRequest


def main() -> None:
    out_path = Path("src/rose_server/static/app/schemas/query-request.schema.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    schema = QueryRequest.model_json_schema()
    out_path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
