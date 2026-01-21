#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["mlx-lm>=0.30.2", "pyyaml"]
# ///
import argparse
import datetime
import hashlib
import json
import logging
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlx_lm  # pyright: ignore[reportMissingImports]
import yaml  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)


def _iter_file_chunks(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                return
            yield chunk


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    for chunk in _iter_file_chunks(path):
        digest.update(chunk)
    return digest.hexdigest()


def _count_jsonl_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _git_commit(repo_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        logger.warning("git not found; cannot record commit for %s", repo_dir)
        return None
    except subprocess.CalledProcessError as exc:
        logger.warning("git rev-parse failed in %s (exit %s)", repo_dir, exc.returncode)
        return None
    commit = result.stdout.strip()
    return commit if commit else None


def _read_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj if isinstance(obj, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a run manifest JSON for reproducibility.")
    parser.add_argument("--output", required=True, help="Path to write manifest JSON.")
    parser.add_argument("--hf-model-name", required=True, help="HuggingFace model name.")
    parser.add_argument("--hf-snapshot-hash", required=True, help="HuggingFace snapshot hash.")
    parser.add_argument("--model-name", required=True, help="Name for this fine-tuned model.")
    parser.add_argument("--dataset-name", required=True, help="Name of the dataset.")
    parser.add_argument("--dataset-split-seed", help="Dataset split seed (optional).")
    parser.add_argument("--exported-model-name", required=True, help="Final GGUF filename.")
    parser.add_argument("--quant-type", required=True, help="Quantization type (e.g., Q4_K_M).")
    parser.add_argument("--train", required=True, help="Path to training JSONL (for hashing).")
    parser.add_argument("--valid", required=True, help="Path to validation JSONL (for hashing).")
    parser.add_argument("--train-config", help="Path to training YAML config (for recording hyperparameters).")
    parser.add_argument("--adapter", required=True, help="Path to adapter weights file (for hashing).")
    parser.add_argument("--gguf", required=True, help="Path to final GGUF file (for hashing).")
    parser.add_argument("--llama-cpp", required=True, help="Path to llama.cpp directory.")
    parser.add_argument("--seed", type=int, help="Training seed.")
    parser.add_argument("--iters", type=int, help="Training iterations.")
    parser.add_argument("--batch-size", type=int, help="Batch size.")
    parser.add_argument("--num-layers", type=int, help="Number of layers fine-tuned.")
    args = parser.parse_args()

    output_path = Path(args.output).expanduser().resolve()
    train_path = Path(args.train).expanduser().resolve()
    valid_path = Path(args.valid).expanduser().resolve()
    train_config_path = Path(args.train_config).expanduser().resolve() if args.train_config else None
    adapter_path = Path(args.adapter).expanduser().resolve()
    gguf_path = Path(args.gguf).expanduser().resolve()
    llama_cpp_dir = Path(args.llama_cpp).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_sha256 = _sha256_file(train_path)
    valid_sha256 = _sha256_file(valid_path)
    adapter_sha256 = _sha256_file(adapter_path)
    gguf_sha256 = _sha256_file(gguf_path)
    train_samples = _count_jsonl_lines(train_path)
    valid_samples = _count_jsonl_lines(valid_path)

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    train_config: dict[str, Any] | None = None
    if train_config_path is not None:
        train_config = _read_yaml_config(train_config_path)

    manifest: dict[str, object] = {
        "date": datetime.datetime.now(datetime.UTC).isoformat(),
        "model_hf_name": args.hf_model_name,
        "model_hf_snapshot": args.hf_snapshot_hash,
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "dataset_split_seed": args.dataset_split_seed,
        "dataset_train_sha256": train_sha256,
        "dataset_valid_sha256": valid_sha256,
        "dataset_train_samples": train_samples,
        "dataset_valid_samples": valid_samples,
        "training_seed": args.seed,
        "training_iters": args.iters,
        "training_batch_size": args.batch_size,
        "training_num_layers": args.num_layers,
        "env_python_version": python_version,
        "env_mlx_lm_version": mlx_lm.__version__,
        "env_llama_cpp_commit": _git_commit(llama_cpp_dir),
        "artifact_exported_name": args.exported_model_name,
        "artifact_quant_type": args.quant_type,
        "artifact_adapter_sha256": adapter_sha256,
        "artifact_gguf_sha256": gguf_sha256,
    }
    if train_config_path is not None and train_config is not None:
        manifest["training_config"] = train_config

    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote manifest: %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
