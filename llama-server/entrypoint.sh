#!/bin/sh
set -e

if [ ! -f "$LLAMA_MODEL_PATH" ]; then
  echo "ERROR: Model file not found: $LLAMA_MODEL_PATH" >&2
  exit 1
fi

exec /app/llama-server \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl -1 \
  -c 2048 \
  -n 512 \
  -m "$LLAMA_MODEL_PATH"
