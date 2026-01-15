#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_DIR = "projects/medical_o1"
HF_CACHE = str(Path.home() / ".cache" / "huggingface" / "hub")
HF_MODEL_NAME = "mlx-community/Olmo-3-7B-Instruct-bf16"
HF_SNAPSHOT_HASH = "c76f027c14b0aeb00e4443a9788c3116b479bf7a"
HF_DATASET_SNAPSHOT_HASH = "fc2c9e8a37b38f38da6d449564a8c350b244aef4"
SOURCE_DATASET_RELATIVE = (
    "datasets--FreedomIntelligence--medical-o1-reasoning-SFT/snapshots/{HF_DATASET_SNAPSHOT_HASH}/medical_o1_sft.json"
)
BASE_MODEL_RELATIVE = "models--mlx-community--Olmo-3-7B-Instruct-bf16/snapshots/{HF_SNAPSHOT_HASH}"

SEED = 42
ITERS = 600
STEPS_PER_EVAL = 50
VAL_BATCHES = 5
BATCH_SIZE = 1
NUM_LAYERS = 16
MAX_SEQ_LENGTH = 2048
HEADROOM_TOKENS = 16

VALID_RATIO = 0.1
SPLIT_SEED = 42

QUANT_TYPE = "Q4_K_M"
COMPARE_OUT_BASE_FILENAME = "compare_base_vs_finetuned"
COMPARE_MAX_TOKENS = 768
COMPARE_TEMP = 0.0
STRICT_SYSTEM_PROMPT = "Return only the final answer. No chain-of-thought. No explanation. No Markdown."

LLAMA_CPP_DIR = "../vendor/llama.cpp"


def _build_cmd(script: str, config: dict[str, Any]) -> list[str]:
    cmd = ["uv", "run", "--prerelease=allow", "--script", script]
    for flag, value in config.items():
        if value is None or value is False:
            continue
        if value is True:
            cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    return cmd


def _run(script: str, config: dict[str, Any], *, cwd: Path) -> None:
    subprocess.run(_build_cmd(script, config), cwd=str(cwd), check=True)


def _run_capture(script: str, config: dict[str, Any], *, cwd: Path) -> str:
    return subprocess.check_output(_build_cmd(script, config), cwd=str(cwd), text=True).strip()


def main() -> None:
    mlx_dir = Path(__file__).resolve().parent
    date_str = time.strftime("%Y-%m-%d")
    run_name = sys.argv[1] if len(sys.argv) > 1 else f"medical_o1_{date_str}_{int(time.time())}"

    base_model = str(Path(HF_CACHE) / BASE_MODEL_RELATIVE.format(HF_SNAPSHOT_HASH=HF_SNAPSHOT_HASH))

    source_jsonl = str(
        Path(HF_CACHE) / SOURCE_DATASET_RELATIVE.format(HF_DATASET_SNAPSHOT_HASH=HF_DATASET_SNAPSHOT_HASH)
    )
    train_config = str(Path(PROJECT_DIR) / "train_config.yaml")
    prompts_file = str(Path(PROJECT_DIR) / "prompts/smoke.txt")

    artifacts_dir = Path("artifacts") / run_name
    models_dir = artifacts_dir / "models"
    adapters_dir = artifacts_dir / "adapters"
    data_out_dir = artifacts_dir / "data"

    train_jsonl = data_out_dir / "train.jsonl"
    valid_jsonl = data_out_dir / "valid.jsonl"

    fp16_gguf = models_dir / f"{run_name}-fp16.gguf"
    quant_gguf = models_dir / f"{run_name}-{QUANT_TYPE}.gguf"

    compare_out = models_dir / f"{COMPARE_OUT_BASE_FILENAME}.jsonl"
    compare_out_strict = models_dir / f"{COMPARE_OUT_BASE_FILENAME}.strict.jsonl"
    compare_meta_out = models_dir / f"{COMPARE_OUT_BASE_FILENAME}.meta.jsonl"
    compare_meta_out_strict = models_dir / f"{COMPARE_OUT_BASE_FILENAME}.strict.meta.jsonl"

    fused_model_dir = models_dir / "model"
    manifest_out = models_dir / "manifest.json"
    training_log = models_dir / "training.log"

    print("\nRunning build_data.py...\n")

    build_data_config: dict[str, Any] = {
        "--source": source_jsonl,
        "--out-dir": str(data_out_dir),
        "--valid-ratio": VALID_RATIO,
        "--seed": SPLIT_SEED,
    }
    _run(str(Path(PROJECT_DIR) / "scripts/build_data.py"), build_data_config, cwd=mlx_dir)

    print("\nRunning validate_data.py (train)...\n")

    validate_train_config: dict[str, Any] = {
        "--model": base_model,
        "--input": str(train_jsonl),
        "--max-seq-length": MAX_SEQ_LENGTH,
        "--headroom": HEADROOM_TOKENS,
        "--output": str(train_jsonl) + ".filtered",
    }
    _run("pipeline/validate_data.py", validate_train_config, cwd=mlx_dir)

    print("\nRunning validate_data.py (valid)...\n")

    validate_valid_config: dict[str, Any] = {
        "--model": base_model,
        "--input": str(valid_jsonl),
        "--max-seq-length": MAX_SEQ_LENGTH,
        "--headroom": HEADROOM_TOKENS,
        "--output": str(valid_jsonl) + ".filtered",
    }
    _run("pipeline/validate_data.py", validate_valid_config, cwd=mlx_dir)

    print("\nRunning finetune.py...\n")

    (mlx_dir / training_log).parent.mkdir(parents=True, exist_ok=True)
    finetune_config: dict[str, Any] = {
        "--model": base_model,
        "--train": str(train_jsonl) + ".filtered",
        "--valid": str(valid_jsonl) + ".filtered",
        "--output": str(adapters_dir),
        "--config": train_config,
        "--iters": ITERS,
        "--steps-per-eval": STEPS_PER_EVAL,
        "--val-batches": VAL_BATCHES,
        "--batch-size": BATCH_SIZE,
        "--num-layers": NUM_LAYERS,
        "--max-seq-length": MAX_SEQ_LENGTH,
        "--seed": SEED,
        "--log-file": str(training_log),
    }
    _run("pipeline/finetune.py", finetune_config, cwd=mlx_dir)

    print("\nRunning select_best_checkpoint.py...\n")

    select_best_config: dict[str, Any] = {
        "--log": str(training_log),
        "--adapter-dir": str(adapters_dir),
    }
    best_adapter = _run_capture("pipeline/select_best_checkpoint.py", select_best_config, cwd=mlx_dir)

    print("\nRunning merge.py...\n")

    merge_config: dict[str, Any] = {
        "--model": base_model,
        "--adapter": best_adapter,
        "--output": str(fused_model_dir),
        "--dequantize": True,
    }
    _run("pipeline/merge.py", merge_config, cwd=mlx_dir)

    print("\nRunning convert_to_fp16_gguf.py...\n")

    fp16_config: dict[str, Any] = {
        "--model": str(fused_model_dir),
        "--output": str(fp16_gguf),
        "--llama-cpp": LLAMA_CPP_DIR,
    }
    _run("pipeline/convert_to_fp16_gguf.py", fp16_config, cwd=mlx_dir)

    print("\nRunning convert_to_gguf.py...\n")

    quant_config: dict[str, Any] = {
        "--input": str(fp16_gguf),
        "--output": str(quant_gguf),
        "--quant": QUANT_TYPE,
        "--llama-cpp": LLAMA_CPP_DIR,
    }
    _run("pipeline/convert_to_gguf.py", quant_config, cwd=mlx_dir)

    print("\nRunning compare_models.py...\n")

    compare_config: dict[str, Any] = {
        "--model-a": base_model,
        "--model-b": str(fused_model_dir),
        "--prompts": prompts_file,
        "--output": str(compare_out),
        "--meta-output": str(compare_meta_out),
        "--temp": COMPARE_TEMP,
        "--max-tokens": COMPARE_MAX_TOKENS,
    }
    _run("pipeline/compare_models.py", compare_config, cwd=mlx_dir)

    print("\nRunning compare_models.py (strict system prompt)...\n")

    compare_strict_config: dict[str, Any] = {
        "--model-a": base_model,
        "--model-b": str(fused_model_dir),
        "--prompts": prompts_file,
        "--output": str(compare_out_strict),
        "--meta-output": str(compare_meta_out_strict),
        "--temp": COMPARE_TEMP,
        "--max-tokens": COMPARE_MAX_TOKENS,
        "--system-prompt": STRICT_SYSTEM_PROMPT,
    }
    _run("pipeline/compare_models.py", compare_strict_config, cwd=mlx_dir)

    print("\nRunning write_manifest.py...\n")

    manifest_config: dict[str, Any] = {
        "--output": str(manifest_out),
        "--hf-model-name": HF_MODEL_NAME,
        "--hf-snapshot-hash": HF_SNAPSHOT_HASH,
        "--model-name": run_name,
        "--dataset-name": "medical_o1",
        "--dataset-split-seed": SPLIT_SEED,
        "--exported-model-name": quant_gguf.name,
        "--quant-type": QUANT_TYPE,
        "--train": str(train_jsonl) + ".filtered",
        "--valid": str(valid_jsonl) + ".filtered",
        "--train-config": train_config,
        "--adapter": best_adapter,
        "--gguf": str(quant_gguf),
        "--llama-cpp": LLAMA_CPP_DIR,
        "--seed": SEED,
        "--iters": ITERS,
        "--batch-size": BATCH_SIZE,
        "--num-layers": NUM_LAYERS,
    }
    _run("pipeline/write_manifest.py", manifest_config, cwd=mlx_dir)


if __name__ == "__main__":
    main()
