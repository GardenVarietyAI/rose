#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["mlx-lm>=0.30.2", "pyyaml"]
# ///
import argparse
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Optional, TypedDict

import yaml  # pyright: ignore[reportMissingImports]
from mlx_lm.utils import load_tokenizer  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)


class Msg(TypedDict):
    role: str
    content: str


def token_len(tokenizer: Any, messages: list[Msg]) -> int:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        return len(ids)

    text = "\n".join(m["content"] for m in messages)
    ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    return len(ids)


def content_hash(messages: list[Msg]) -> str:
    text = "\n".join(f"{m['role']}:{m['content']}" for m in messages)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def parse_messages(obj: Any) -> list[Msg]:
    if not isinstance(obj, dict):
        raise ValueError

    msgs = obj.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise ValueError

    out: list[Msg] = []
    for m in msgs:
        if not isinstance(m, dict):
            raise ValueError
        role = m.get("role")
        content = m.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError
        out.append({"role": role, "content": content})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--headroom", type=int, default=16)
    parser.add_argument("--output")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    max_seq_length: int = config["max_seq_length"]

    model_dir = Path(args.model).expanduser().resolve()
    in_path = Path(args.input).expanduser().resolve()
    out_path: Optional[Path] = Path(args.output).expanduser().resolve() if args.output else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    limit = max_seq_length - args.headroom

    tok = load_tokenizer(model_dir)

    counts: Counter[str] = Counter()
    max_seen = 0
    seen_hashes: set[str] = set()

    out_f = out_path.open("w", encoding="utf-8") if out_path else None
    try:
        with in_path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                    msgs = parse_messages(obj)
                except Exception:
                    counts["bad"] += 1
                    continue

                counts["total"] += 1
                n = token_len(tok, msgs)
                if n > max_seen:
                    max_seen = n

                if n > limit:
                    counts["over"] += 1
                    continue

                h = content_hash(msgs)
                if h in seen_hashes:
                    counts["dupes"] += 1
                    continue
                seen_hashes.add(h)

                counts["kept"] += 1
                if out_f:
                    out_f.write(line + "\n")
    finally:
        if out_f:
            out_f.close()

    logger.info("Checked dataset: %s", in_path)
    logger.info("Model tokenizer: %s", model_dir)
    logger.info("Limit: %s (max_seq_length=%s, headroom=%s)", limit, max_seq_length, args.headroom)
    logger.info("Total items: %s", counts["total"])
    logger.info("Bad items: %s", counts["bad"])
    logger.info("Over limit: %s", counts["over"])
    logger.info("Duplicates: %s", counts["dupes"])
    logger.info("Kept: %s", counts["kept"])
    logger.info("Max tokens seen: %s", max_seen)
    if out_path:
        logger.info("Filtered output: %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
