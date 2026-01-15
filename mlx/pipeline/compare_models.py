#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["mlx-lm>=0.30.2"]
# ///
import argparse
import json
import logging
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)


def _iter_prompts(path: Path, limit: int | None) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as f:
        count = 0
        for raw in f:
            prompt = raw.rstrip("\n")
            if prompt.strip() == "" or prompt.lstrip().startswith("#"):
                continue
            yield prompt
            count += 1
            if limit is not None and count >= limit:
                return


def _generate(
    *,
    model_dir: Path,
    adapter_path: Path | None,
    prompt: str,
    max_tokens: int,
    temp: float,
    top_p: float | None,
    top_k: int | None,
    min_p: float | None,
    seed: int | None,
    system_prompt: str | None,
    verbose: bool,
) -> str:
    config: dict[str, str | int | float | bool | None] = {
        "--model": str(model_dir),
        "--prompt": prompt,
        "--max-tokens": max_tokens,
        "--temp": temp,
        "--top-p": top_p,
        "--top-k": top_k,
        "--min-p": min_p,
        "--seed": seed,
        "--system-prompt": system_prompt,
        "--verbose": "T" if verbose else "F",
    }
    if adapter_path is not None:
        config["--adapter-path"] = str(adapter_path)

    cmd: list[str] = [sys.executable, "-m", "mlx_lm", "generate"]
    for flag, value in config.items():
        if value is None or value is False:
            continue
        if value is True:
            cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        cmd_str = " ".join(str(part) for part in exc.cmd)
        logger.error("Command failed (%s): %s", exc.returncode, cmd_str)
        sys.exit(exc.returncode)
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two MLX models side-by-side on a prompt set.")
    parser.add_argument("--model-a", required=True, help="Path to model A directory.")
    parser.add_argument("--model-b", required=True, help="Path to model B directory.")
    parser.add_argument("--adapter-a", help="Optional adapter path for model A.")
    parser.add_argument("--adapter-b", help="Optional adapter path for model B.")
    parser.add_argument("--name-a", default="A", help="Label for model A in output (default: A).")
    parser.add_argument("--name-b", default="B", help="Label for model B in output (default: B).")
    parser.add_argument("--prompts", required=True, help="Path to prompts file.")
    parser.add_argument("--output", required=True, help="Path to write JSONL results.")
    parser.add_argument("--meta-output", help="Optional path to write meta JSONL (1 line).")
    parser.add_argument("--max-tokens", type=int, default=768, help="Max tokens to generate (default: 768).")
    parser.add_argument("--temp", type=float, default=0.0, help="Temperature (default: 0.0).")
    parser.add_argument("--top-p", type=float, help="Top-p; omit to use mlx_lm default.")
    parser.add_argument("--top-k", type=int, help="Top-k; omit to use mlx_lm default.")
    parser.add_argument("--min-p", type=float, help="Min-p; omit to use mlx_lm default.")
    parser.add_argument("--seed", type=int, help="PRNG seed; omit to use mlx_lm default.")
    parser.add_argument("--system-prompt", help="System prompt; omit to use tokenizer default.")
    parser.add_argument("--limit", type=int, help="Optional max prompts to run.")
    parser.add_argument("--verbose", action="store_true", help="Log verbose generation output.")
    args = parser.parse_args()

    model_a = Path(args.model_a).expanduser().resolve()
    model_b = Path(args.model_b).expanduser().resolve()
    adapter_a = Path(args.adapter_a).expanduser().resolve() if args.adapter_a else None
    adapter_b = Path(args.adapter_b).expanduser().resolve() if args.adapter_b else None
    prompts_path = Path(args.prompts).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    meta_output_path = Path(args.meta_output).expanduser().resolve() if args.meta_output else None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    meta = {
        "created_at_unix": int(started_at),
        "prompts_path": str(prompts_path),
        "model_a": {"name": args.name_a, "model": str(model_a), "adapter": str(adapter_a) if adapter_a else None},
        "model_b": {"name": args.name_b, "model": str(model_b), "adapter": str(adapter_b) if adapter_b else None},
        "decode": {
            "max_tokens": args.max_tokens,
            "temp": args.temp,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "seed": args.seed,
            "system_prompt": args.system_prompt,
        },
    }

    meta_line = json.dumps(meta, ensure_ascii=False) + "\n"
    if meta_output_path is not None:
        meta_output_path.write_text(meta_line, encoding="utf-8")

    with output_path.open("w", encoding="utf-8") as out_f:
        if meta_output_path is None:
            out_f.write(meta_line)
        for prompt in _iter_prompts(prompts_path, args.limit):
            a_text = _generate(
                model_dir=model_a,
                adapter_path=adapter_a,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temp=args.temp,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                seed=args.seed,
                system_prompt=args.system_prompt,
                verbose=args.verbose,
            )
            b_text = _generate(
                model_dir=model_b,
                adapter_path=adapter_b,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temp=args.temp,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                seed=args.seed,
                system_prompt=args.system_prompt,
                verbose=args.verbose,
            )
            record = {
                "prompt": prompt,
                f"{args.name_a}": a_text,
                f"{args.name_b}": b_text,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Wrote comparison: %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
