#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
import argparse
import hashlib
import json
import logging
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)


def _stable_bucket(seed: str, line: str) -> int:
    digest = hashlib.sha256((seed + "\n" + line).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % 10_000


def _iter_jsonl_lines(path: Path) -> Iterator[tuple[int, str]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                line = raw.rstrip("\n")
                if line.strip() == "":
                    continue
                yield line_no, line
    else:
        items = json.loads(path.read_text(encoding="utf-8"))
        for idx, item in enumerate(items, start=1):
            question = item.get("Question")
            cot = item.get("Complex_CoT")
            response = item.get("Response")
            messages = {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": f"{cot}\n\n{response}".strip()},
                ]
            }
            yield idx, json.dumps(messages, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build medical_o1 train/valid JSONL from a source JSONL.")
    parser.add_argument("--source", required=True, help="Path to data source (.jsonl or .json).")
    parser.add_argument("--out-dir", required=True, help="Output directory to write train.jsonl and valid.jsonl.")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="Validation ratio (default: 0.1).")
    parser.add_argument("--seed", default="42", help="Split seed (default: 42).")
    parser.add_argument("--max-items", type=int, default=None, help="Optional max items to write.")
    args = parser.parse_args()

    source = Path(args.source).expanduser()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    threshold = int(args.valid_ratio * 10_000)
    train_path = out_dir / "train.jsonl"
    valid_path = out_dir / "valid.jsonl"

    train_count = 0
    valid_count = 0
    max_items = args.max_items

    with train_path.open("w", encoding="utf-8") as train_f, valid_path.open("w", encoding="utf-8") as valid_f:
        for _, line in _iter_jsonl_lines(source):
            bucket = _stable_bucket(args.seed, line)
            if bucket < threshold:
                valid_f.write(line + "\n")
                valid_count += 1
            else:
                train_f.write(line + "\n")
                train_count += 1

            if max_items is not None and (train_count + valid_count) >= max_items:
                break

    logger.info("Wrote train: %s (%s items)", train_path, train_count)
    logger.info("Wrote valid: %s (%s items)", valid_path, valid_count)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
