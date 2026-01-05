#!/bin/sh
set -e

if [ ! -f "$LLAMA_MODEL_PATH" ]; then
  echo "ERROR: Model file not found: $LLAMA_MODEL_PATH" >&2
  exit 1
fi

exec /app/llama-server \
  -m "$LLAMA_MODEL_PATH" \
  --host 0.0.0.0 --port 8080 \
  -ngl -1 \
  -c 32768 \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --batch-size 4096 --ubatch-size 1024 \
  -t 16 -tb 16 \
  --metrics
