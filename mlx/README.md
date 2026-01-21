# MLX Fine-Tuning Pipeline

This repo is organized as:

- `mlx/pipeline/`: shared scripts for training/testing/merging on MLX (generic).
- `mlx/projects/`: per-project folders with data/prompts/eval sets tailored to a specific model/dataset.
- `scripts/`: cross-project utilities (e.g., GGUF conversion via `llama.cpp`).
- `mlx/artifacts/`: local artifacts (base models, adapters, fused models, GGUFs).

## Requirements

- `uv` (all Python scripts use `uv run`)
- `cmake` (only needed if you build `llama.cpp`)

## Pipeline scripts

| Script | Purpose |
|--------|---------|
| `mlx/pipeline/validate_data.py` | Filter JSONL by max sequence length (also dedupes) |
| `mlx/pipeline/finetune.py` | LoRA fine-tune adapters (`mlx_lm lora`) |
| `mlx/pipeline/merge.py` | Fuse adapters into a standalone HF directory (`mlx_lm.fuse`) |
| `mlx/pipeline/select_best_checkpoint.py` | Pick the best adapter checkpoint from `training.log` |
| `mlx/pipeline/compare_models.py` | Compare two models on a prompt list (writes JSONL + meta JSONL) |
| `mlx/pipeline/evaluate.py` | Perplexity eval on a dataset (`mlx_lm perplexity` / `mlx_lm lora --test`) |
| `mlx/pipeline/write_manifest.py` | Write a run manifest JSON (includes `training_config`) |
| `mlx/pipeline/smoke_test.py` | Quick generation test on a fused model |
| `mlx/pipeline/convert_to_fp16_gguf.py` | Convert HF model to fp16 GGUF |
| `mlx/pipeline/convert_to_gguf.py` | Quantize fp16 GGUF to target format (e.g., Q4_K_M) |

## GGUF conversion

1. Build llama.cpp (one-time): `python scripts/build_llama_cpp.py`
2. Convert to fp16: `uv run mlx/pipeline/convert_to_fp16_gguf.py --model <model-dir> --output <model>-fp16.gguf`
3. Quantize: `uv run mlx/pipeline/convert_to_gguf.py --input <model>-fp16.gguf --output <model>-Q4_K_M.gguf`

## Projects

- `mlx/projects/commonsense/`: commonsense dataset + prompts.
- `mlx/projects/medical_o1/`: medical dataset fine-tuning project.

## Running GGUF models with llama-cli

Non-interactive single prompt:

```bash
vendor/llama.cpp/build/bin/llama-cli \
  -m mlx/artifacts/<run>/models/<run>-Q4_K_M.gguf \
  -p "Your prompt here" \
  -n 256 \
  --no-conversation
```

Flags:
- `-m`: Path to GGUF model
- `-p`: Prompt text
- `-n`: Max tokens to generate
- `--no-conversation`: Disable interactive mode (exit after generation)
