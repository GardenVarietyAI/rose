#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
import logging
import sys

from pipeline.orchestrator import Config, Orchestrator

logging.basicConfig(level=logging.INFO, format="%(message)s")

HF_CACHE = "~/.cache/huggingface/hub"

config = Config(
    project_name="code_alpaca_olmo3_7b",
    hf_model_name="allenai/OLMo-3-7B-Instruct",
    hf_model_hash="096bb5469fe34348bc88d851a69edb3bf6f40df4",
    hf_model_path=f"{HF_CACHE}/models--allenai--OLMo-3-7B-Instruct/snapshots/096bb5469fe34348bc88d851a69edb3bf6f40df4",
    dataset_path=f"{HF_CACHE}/datasets--HuggingFaceH4--CodeAlpaca_20K/snapshots/798c567f69c8f4b12fc191015e59ee34e9afe00d/data/train-00000-of-00001.parquet",
    dataset_name="code_alpaca",
    eval_tasks="mmlu_computer_security,mmlu_machine_learning",
)

if __name__ == "__main__":
    run_name = sys.argv[1] if len(sys.argv) > 1 else None
    Orchestrator(config, run_name).run()
