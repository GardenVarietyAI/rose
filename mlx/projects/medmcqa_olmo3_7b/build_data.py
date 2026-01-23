#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["pandas", "pyarrow"]
# ///
import argparse
import hashlib
import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pandas as pd  # pyright: ignore[reportMissingImports,reportMissingModuleSource]

logger = logging.getLogger(__name__)

OPTIONS = ["A", "B", "C", "D"]


def _stable_bucket(seed: str, line: str) -> int:
    digest = hashlib.sha256((seed + "\n" + line).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % 10_000


def _format_question(row: pd.Series) -> str:
    question = row["question"]
    opa = row["opa"]
    opb = row["opb"]
    opc = row["opc"]
    opd = row["opd"]
    return f"{question}\nA. {opa}\nB. {opb}\nC. {opc}\nD. {opd}"


def _format_answer(row: pd.Series) -> str:
    cop = row["cop"]
    option_letter = OPTIONS[cop]
    option_text = row[f"op{option_letter.lower()}"]
    return f"{option_letter}. {option_text}"


def _iter_rows(df: pd.DataFrame) -> Iterator[tuple[str, str]]:
    for _, row in df.iterrows():
        prompt = _format_question(row)
        response = _format_answer(row)
        yield prompt, response


def main() -> None:
    parser = argparse.ArgumentParser(description="Build medmcqa train/valid JSONL from parquet.")
    parser.add_argument("--source", required=True, help="Path to source parquet file.")
    parser.add_argument("--out-dir", required=True, help="Output directory for train.jsonl and valid.jsonl.")
    parser.add_argument("--valid-ratio", type=float, required=True, help="Validation ratio.")
    parser.add_argument("--seed", required=True, help="Split seed.")
    args = parser.parse_args()

    source = Path(args.source).expanduser()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(source)
    logger.info("Loaded %d rows from %s", len(df), source)

    threshold = int(args.valid_ratio * 10_000)
    train_path = out_dir / "train.jsonl"
    valid_path = out_dir / "valid.jsonl"

    train_count = 0
    valid_count = 0

    with train_path.open("w", encoding="utf-8") as train_f, valid_path.open("w", encoding="utf-8") as valid_f:
        for prompt, response in _iter_rows(df):
            messages = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]
            }
            line = json.dumps(messages, ensure_ascii=False)

            bucket = _stable_bucket(args.seed, line)
            if bucket < threshold:
                valid_f.write(line + "\n")
                valid_count += 1
            else:
                train_f.write(line + "\n")
                train_count += 1

    logger.info("Wrote train: %s (%s items)", train_path, train_count)
    logger.info("Wrote valid: %s (%s items)", valid_path, valid_count)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
