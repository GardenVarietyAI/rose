# Build Scripts

## build.sh

- Builds both JavaScript and CSS bundles using esbuild via Docker Compose.

**Usage:**
```bash
mise esbuild
# or
./scripts/build.sh
```

## ajv.sh

- Generates JSON Schema + standalone Ajv validators used by the frontend.

**Usage:**
```bash
mise ajv
# or
./scripts/ajv.sh
```

## convert_to_gguf.py

- Converts a fused HuggingFace model directory to GGUF fp16 + quantized GGUF via `vendor/llama.cpp`.

**Usage:**
```bash
./mlx/pipeline/convert_to_gguf.py --model mlx/artifacts/models/<name>/model --output mlx/artifacts/models/<name>/<name>-Q4_K_M.gguf --quant Q4_K_M
```
