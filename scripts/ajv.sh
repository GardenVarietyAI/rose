#!/usr/bin/env bash
set -euo pipefail

echo "Generating JSON schemas..."
uv run python scripts/generate_query_schema.py

echo "Generating Ajv standalone validators..."
docker compose --profile tools run --rm ajv \
  node scripts/generate_ajv_validators.mjs

echo "✓ Ajv generation complete"
