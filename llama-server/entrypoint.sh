#!/bin/sh
set -e

MODEL_PATH=$(find /root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-GGUF/snapshots -name "*.gguf" \( -type f -o -type l \) | head -n 1)

if [ -z "$MODEL_PATH" ]; then
    echo "Error: No .gguf model file found"
    echo "Searching in: /root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-GGUF/snapshots"
    find /root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-GGUF/snapshots -ls 2>&1 | head -20
    exit 1
fi

echo "Found model at: $MODEL_PATH"

exec /app/llama-server -m "$MODEL_PATH" "$@"
