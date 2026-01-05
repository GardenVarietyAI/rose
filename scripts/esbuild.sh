#!/usr/bin/env bash
set -euo pipefail

echo "Building JavaScript bundle..."
docker compose --profile tools run --rm esbuild \
  src/rose_server/static/app/app.js \
  --minify \
  --bundle \
  --platform=browser \
  --outfile=src/rose_server/static/dist/app.min.js

echo "Building CSS bundle..."
docker compose --profile tools run --rm esbuild \
  src/rose_server/static/app/app.css \
  --minify \
  --bundle \
  --outfile=src/rose_server/static/dist/app.min.css

echo "✓ Build complete"
