#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_DIR = "projects/medical_o1_olmo3_7b"
HF_CACHE = str(Path.home() / ".cache" / "huggingface" / "hub")

HF_MODEL_NAME = "allenai/OLMo-3-7B-Instruct"
HF_SNAPSHOT_HASH = "096bb5469fe34348bc88d851a69edb3bf6f40df4"
HF_SOURCE_MODEL_RELATIVE = "models--allenai--OLMo-3-7B-Instruct/snapshots/{HF_SNAPSHOT_HASH}"

HF_DATASET_SNAPSHOT_HASH = "fc2c9e8a37b38f38da6d449564a8c350b244aef4"
SOURCE_DATASET_RELATIVE = (
    "datasets--FreedomIntelligence--medical-o1-reasoning-SFT/snapshots/{HF_DATASET_SNAPSHOT_HASH}/medical_o1_sft.json"
)

HEADROOM_TOKENS = 16
VALID_RATIO = 0.1
SPLIT_SEED = 42

QUANT_TYPE = "Q4_K_M"
COMPARE_OUT_BASE_FILENAME = "compare_base_vs_finetuned"
COMPARE_MAX_TOKENS = 768
COMPARE_TEMP = 0.0

LLAMA_CPP_DIR = "../vendor/llama.cpp"

MLX_DIR = Path(__file__).resolve().parent


def main() -> None:
    date_str = time.strftime("%Y-%m-%d")
    run_name = sys.argv[1] if len(sys.argv) > 1 else f"medical_o1_olmo3_7b_{date_str}_{int(time.time())}"

    source_hf_model = str(Path(HF_CACHE) / HF_SOURCE_MODEL_RELATIVE.format(HF_SNAPSHOT_HASH=HF_SNAPSHOT_HASH))

    source_jsonl = str(
        Path(HF_CACHE) / SOURCE_DATASET_RELATIVE.format(HF_DATASET_SNAPSHOT_HASH=HF_DATASET_SNAPSHOT_HASH)
    )
    train_config = str(Path(PROJECT_DIR) / "train_config.yaml")
    prompts_file = str(Path(PROJECT_DIR) / "prompts/smoke.txt")

    artifacts_dir = Path("artifacts") / run_name
    models_dir = artifacts_dir / "models"
    adapters_dir = artifacts_dir / "adapters"
    data_out_dir = artifacts_dir / "data"

    base_model = models_dir / "base_model"
    train_jsonl = data_out_dir / "train.jsonl"
    valid_jsonl = data_out_dir / "valid.jsonl"

    fp16_gguf = models_dir / f"{run_name}-fp16.gguf"
    quant_gguf = models_dir / f"{run_name}-{QUANT_TYPE}.gguf"

    compare_out = models_dir / f"{COMPARE_OUT_BASE_FILENAME}.jsonl"
    compare_meta_out = models_dir / f"{COMPARE_OUT_BASE_FILENAME}.meta.jsonl"

    fused_model_dir = models_dir / "model"
    manifest_out = models_dir / "manifest.json"
    training_log = models_dir / "training.log"
    best_checkpoint_json = models_dir / "best_checkpoint.json"

    train_jsonl_filtered = str(train_jsonl) + ".filtered"
    valid_jsonl_filtered = str(valid_jsonl) + ".filtered"

    steps: list[Any] = [
        [
            "pipeline/convert_hf_to_mlx.py",
            {
                "--model": source_hf_model,
                "--output": str(base_model),
                "--dtype": "bfloat16",
            },
        ],
        [
            f"{PROJECT_DIR}/scripts/build_data.py",
            {
                "--source": source_jsonl,
                "--out-dir": str(data_out_dir),
                "--valid-ratio": VALID_RATIO,
                "--seed": SPLIT_SEED,
            },
        ],
        [
            "pipeline/validate_data.py",
            {
                "--model": str(base_model),
                "--input": str(train_jsonl),
                "--config": train_config,
                "--headroom": HEADROOM_TOKENS,
                "--output": train_jsonl_filtered,
            },
        ],
        [
            "pipeline/validate_data.py",
            {
                "--model": str(base_model),
                "--input": str(valid_jsonl),
                "--config": train_config,
                "--headroom": HEADROOM_TOKENS,
                "--output": valid_jsonl_filtered,
            },
        ],
        [
            "pipeline/finetune.py",
            {
                "--model": str(base_model),
                "--train": train_jsonl_filtered,
                "--valid": valid_jsonl_filtered,
                "--output": str(adapters_dir),
                "--config": train_config,
                "--log-file": str(training_log),
            },
        ],
        [
            "pipeline/select_best_checkpoint.py",
            {
                "--log": str(training_log),
                "--adapter-dir": str(adapters_dir),
                "--output": str(best_checkpoint_json),
            },
        ],
        [
            "pipeline/merge.py",
            {
                "--model": str(base_model),
                "--checkpoint-metadata": str(best_checkpoint_json),
                "--output": str(fused_model_dir),
                "--dequantize": True,
            },
        ],
        [
            "pipeline/convert_to_fp16_gguf.py",
            {
                "--model": str(fused_model_dir),
                "--output": str(fp16_gguf),
                "--llama-cpp": LLAMA_CPP_DIR,
            },
        ],
        [
            "pipeline/convert_to_gguf.py",
            {
                "--input": str(fp16_gguf),
                "--output": str(quant_gguf),
                "--quant": QUANT_TYPE,
                "--llama-cpp": LLAMA_CPP_DIR,
            },
        ],
        [
            "pipeline/compare_models.py",
            {
                "--model-a": str(base_model),
                "--model-b": str(fused_model_dir),
                "--prompts": prompts_file,
                "--output": str(compare_out),
                "--meta-output": str(compare_meta_out),
                "--temp": COMPARE_TEMP,
                "--max-tokens": COMPARE_MAX_TOKENS,
            },
        ],
        [
            "pipeline/write_manifest.py",
            {
                "--output": str(manifest_out),
                "--hf-model-name": HF_MODEL_NAME,
                "--hf-snapshot-hash": HF_SNAPSHOT_HASH,
                "--model-name": run_name,
                "--dataset-name": "medical_o1",
                "--dataset-split-seed": SPLIT_SEED,
                "--exported-model-name": quant_gguf.name,
                "--quant-type": QUANT_TYPE,
                "--train": train_jsonl_filtered,
                "--valid": valid_jsonl_filtered,
                "--train-config": train_config,
                "--checkpoint-metadata": str(best_checkpoint_json),
                "--gguf": str(quant_gguf),
                "--llama-cpp": LLAMA_CPP_DIR,
            },
        ],
    ]

    for script, config in steps:
        print(f"\nRunning {Path(script).name}...\n")
        cmd = ["uv", "run", "--prerelease=allow", "--script", script]
        for flag, value in config.items():
            if value is None or value is False:
                continue
            if value is True:
                cmd.append(flag)
            else:
                cmd.extend([flag, str(value)])
        subprocess.run(cmd, cwd=str(MLX_DIR), check=True)


if __name__ == "__main__":
    main()
