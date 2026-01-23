import datetime
import logging
import os
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS steps (
    run_id TEXT NOT NULL,
    name TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    elapsed_seconds REAL NOT NULL,
    PRIMARY KEY (run_id, name)
);
"""


@dataclass
class Config:
    project_name: str
    hf_model_name: str
    hf_model_hash: str
    hf_model_path: str
    dataset_path: str
    dataset_name: str
    eval_tasks: str
    quant_type: str = "Q4_K_M"
    headroom_tokens: int = 16
    valid_ratio: float = 0.1
    split_seed: int = 42
    eval_batch_size: int = 1
    early_stop_patience: int = 3
    early_stop_min_delta: float = 0.001
    artifacts_dir: str = "./artifacts"
    llama_cpp_dir: str = "../vendor/llama.cpp"
    database_path: str = "./experiments.db"


class Orchestrator:
    def __init__(self, config: Config, run_name: str | None = None) -> None:
        self.config = config

        date_str = time.strftime("%Y-%m-%d")
        self.run_name = run_name or f"{config.project_name}_{date_str}_{int(time.time())}"

        self.artifacts_dir = Path(config.artifacts_dir) / self.run_name
        self.models_dir = self.artifacts_dir / "models"
        self.adapters_dir = self.artifacts_dir / "adapters"
        self.data_out_dir = self.artifacts_dir / "data"

    def run(self) -> None:
        conn = self._init_db(Path(self.config.database_path))

        if not self._run_exists(conn, self.run_name):
            self._create_run(conn, self.run_name)

        steps = self._build_steps()

        for step_name, step_config in steps.items():
            if self._is_step_completed(conn, self.run_name, step_name):
                logger.info("Skipping %s (already completed)", step_name)
                continue

            script = step_config.pop("_script")
            logger.info("[%s] Running %s...", step_name, Path(script).name)
            started = time.time()

            cmd = ["uv", "run", "--prerelease=allow", "--script", script]
            for flag, value in step_config.items():
                if value is None or value is False:
                    continue
                if value is True:
                    cmd.append(flag)
                else:
                    cmd.extend([flag, str(value)])

            subprocess.run(cmd, check=True)
            elapsed = time.time() - started
            self._mark_step_completed(conn, self.run_name, step_name, elapsed)
            logger.info("[%s] Completed in %.1fs", step_name, elapsed)

        logger.info("Run %s completed successfully.", self.run_name)

    def _init_db(self, db_path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(db_path)
        conn.executescript(DB_SCHEMA)
        conn.commit()
        return conn

    def _create_run(self, conn: sqlite3.Connection, run_id: str) -> None:
        now = datetime.datetime.now(datetime.UTC).isoformat()
        conn.execute("INSERT INTO runs (id, created_at) VALUES (?, ?)", (run_id, now))
        conn.commit()

    def _run_exists(self, conn: sqlite3.Connection, run_id: str) -> bool:
        cur = conn.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,))
        return cur.fetchone() is not None

    def _is_step_completed(self, conn: sqlite3.Connection, run_id: str, step_name: str) -> bool:
        cur = conn.execute("SELECT 1 FROM steps WHERE run_id = ? AND name = ?", (run_id, step_name))
        return cur.fetchone() is not None

    def _mark_step_completed(self, conn: sqlite3.Connection, run_id: str, step_name: str, elapsed: float) -> None:
        now = datetime.datetime.now(datetime.UTC).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO steps (run_id, name, completed_at, elapsed_seconds) VALUES (?, ?, ?, ?)",
            (run_id, step_name, now, elapsed),
        )
        conn.commit()

    def _build_steps(self) -> dict[str, dict[str, Any]]:
        cfg = self.config

        project_dir = f"projects/{cfg.project_name}"
        train_config = str(Path(project_dir) / "train_config.yaml")

        base_model = self.models_dir / "base_model"
        train_jsonl = self.data_out_dir / "train.jsonl"
        valid_jsonl = self.data_out_dir / "valid.jsonl"
        train_jsonl_filtered = str(train_jsonl) + ".filtered"
        valid_jsonl_filtered = str(valid_jsonl) + ".filtered"

        fp16_gguf = self.models_dir / f"{self.run_name}-fp16.gguf"
        quant_gguf = self.models_dir / f"{self.run_name}-{cfg.quant_type}.gguf"

        fused_model_dir = self.models_dir / "model"
        manifest_out = self.models_dir / "manifest.json"
        checkpoint_metadata = self.models_dir / "checkpoint.meta.json"
        eval_base_results = self.models_dir / "eval_base_results.json"
        eval_final_results = self.models_dir / "eval_final_results.json"

        return {
            "convert_hf_to_mlx": {
                "_script": "pipeline/convert_hf_to_mlx.py",
                "--model": os.path.expanduser(cfg.hf_model_path),
                "--output": str(base_model),
                "--dtype": "bfloat16",
            },
            "build_data": {
                "_script": f"{project_dir}/build_data.py",
                "--source": os.path.expanduser(cfg.dataset_path),
                "--out-dir": str(self.data_out_dir),
                "--valid-ratio": cfg.valid_ratio,
                "--seed": cfg.split_seed,
            },
            "validate_train": {
                "_script": "pipeline/validate_data.py",
                "--model": str(base_model),
                "--input": str(train_jsonl),
                "--config": train_config,
                "--headroom": cfg.headroom_tokens,
                "--output": train_jsonl_filtered,
            },
            "validate_valid": {
                "_script": "pipeline/validate_data.py",
                "--model": str(base_model),
                "--input": str(valid_jsonl),
                "--config": train_config,
                "--headroom": cfg.headroom_tokens,
                "--output": valid_jsonl_filtered,
            },
            "finetune": {
                "_script": "pipeline/finetune.py",
                "--model": str(base_model),
                "--train": train_jsonl_filtered,
                "--valid": valid_jsonl_filtered,
                "--output": str(self.adapters_dir),
                "--config": train_config,
                "--checkpoint-metadata": str(checkpoint_metadata),
                "--patience": cfg.early_stop_patience,
                "--min-delta": cfg.early_stop_min_delta,
            },
            "merge": {
                "_script": "pipeline/merge.py",
                "--model": str(base_model),
                "--adapters": str(self.adapters_dir),
                "--checkpoint-metadata": str(checkpoint_metadata),
                "--output": str(fused_model_dir),
                "--dequantize": True,
            },
            "convert_to_fp16_gguf": {
                "_script": "pipeline/convert_to_fp16_gguf.py",
                "--model": str(fused_model_dir),
                "--output": str(fp16_gguf),
                "--llama-cpp": cfg.llama_cpp_dir,
            },
            "convert_to_gguf": {
                "_script": "pipeline/convert_to_gguf.py",
                "--input": str(fp16_gguf),
                "--output": str(quant_gguf),
                "--quant": cfg.quant_type,
                "--llama-cpp": cfg.llama_cpp_dir,
            },
            "evaluate_base": {
                "_script": "pipeline/evaluate_base.py",
                "--model": cfg.hf_model_name,
                "--tasks": cfg.eval_tasks,
                "--output": str(eval_base_results),
                "--batch-size": cfg.eval_batch_size,
            },
            "evaluate_final": {
                "_script": "pipeline/evaluate_final.py",
                "--model": str(fused_model_dir),
                "--tasks": cfg.eval_tasks,
                "--output": str(eval_final_results),
                "--batch-size": cfg.eval_batch_size,
            },
            "write_manifest": {
                "_script": "pipeline/write_manifest.py",
                "--output": str(manifest_out),
                "--hf-model-name": cfg.hf_model_name,
                "--hf-snapshot-hash": cfg.hf_model_hash,
                "--model-name": self.run_name,
                "--dataset-name": cfg.dataset_name,
                "--dataset-split-seed": cfg.split_seed,
                "--exported-model-name": quant_gguf.name,
                "--quant-type": cfg.quant_type,
                "--train": train_jsonl_filtered,
                "--valid": valid_jsonl_filtered,
                "--train-config": train_config,
                "--checkpoint-metadata": str(checkpoint_metadata),
                "--adapters": str(self.adapters_dir),
                "--gguf": str(quant_gguf),
                "--llama-cpp": cfg.llama_cpp_dir,
            },
            "cleanup": {
                "_script": "pipeline/cleanup.py",
                "--base-model": str(base_model),
                "--fused-model": str(fused_model_dir),
                "--fp16-gguf": str(fp16_gguf),
            },
        }
