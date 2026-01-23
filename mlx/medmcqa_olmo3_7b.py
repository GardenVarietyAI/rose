#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
import logging
import os
import sys

from pipeline.orchestrator import Config, Orchestrator

logging.basicConfig(level=logging.INFO, format="%(message)s")

HF_CACHE = os.environ.get("HF_HOME", "~/.cache/huggingface/hub")

config = Config(
    project_name="medmcqa_olmo3_7b",
    hf_model_name="allenai/OLMo-3-7B-Instruct",
    hf_model_hash="096bb5469fe34348bc88d851a69edb3bf6f40df4",
    hf_model_path=f"{HF_CACHE}/models--allenai--OLMo-3-7B-Instruct/snapshots/096bb5469fe34348bc88d851a69edb3bf6f40df4",
    dataset_path=f"{HF_CACHE}/datasets--openlifescienceai--medmcqa/snapshots/91c6572c454088bf71b679ad90aa8dffcd0d5868/data/train-00000-of-00001.parquet",
    dataset_name="medmcqa",
    eval_tasks="medqa_4options,mmlu_professional_medicine,mmlu_clinical_knowledge",
)

if __name__ == "__main__":
    run_name = sys.argv[1] if len(sys.argv) > 1 else None
    Orchestrator(config, run_name).run()
