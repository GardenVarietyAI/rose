# ^ROSE

 [![CI](https://github.com/GardenVarietyAI/rose-server/actions/workflows/ci.yml/badge.svg)](https://github.com/GardenVarietyAI/rose-server/actions/workflows/ci.yml)

An LLM server implementing a subset of OpenAI’s API for local model experimentation

This is pre-release software. Use at your own risk!

## Features

- Local Model Inference
- LoRA Fine-Tuning
- Embeddings
- SSE Streaming Support

### OpenAI-Compatible API

- `v1/chat/completions`
- `v1/embeddings`
- `v1/fine_tuning`
- `v1/responses`

## Quick Start

### Installation

```bash
# Install mise (https://mise.jdx.dev/getting-started.html)
curl https://mise.run | sh

# Clone the repository
git clone --branch 0.1.3 --depth 1 git@github.com:GardenVarietyAI/rose.git
cd rose

# Install deps
mise install
mise exec -- uv venv
uv pip install --group cli --group trainer --group utils
```

### Initialize Database

```bash
# Set up the database schema using dbmate
cp env.example .env
mkdir -p data
dbmate --no-dump-schema up
```

### Build the Inference Service

```bash
uv run maturin develop -F metal --release
```

### Download a Model

```bash
# Download a Hugging Face model
uv run rose models download Qwen/Qwen3-0.6B
```

### Download and Convert an Embedding Model

```bash
# This is a temporary step that will be smoothed out in a future release
uv run rose models download Qwen/Qwen3-Embedding-0.6B
uv run rose-utils convert data/models/Qwen--Qwen3-Embedding-0.6B data/models/Qwen3-Embedding-0.6B-ONNX
```

### Running the Services

```bash
uv run rose-server
uv run rose-trainer
```

### Start a chat

```bash
# Chat with Qwen/Qwen3-0.6B
uv run rose chat --model Qwen/Qwen3-0.6B
```

## Model Library

| Model | Size | Download |
| ----- | ---- | -------- |
| Qwen/Qwen3-0.6B | 0.6B | `uv run rose models download Qwen/Qwen3-0.6B` |
| Qwen/Qwen3-1.7B | 1.7B | `uv run rose models download Qwen/Qwen3-1.7B` |
| Qwen/Qwen3-1.7B-Base | 1.7B | `uv run rose models download Qwen/Qwen3-1.7B-Base` |
| Qwen/Qwen3-0.6B-GGUF | 0.6B | `uv run rose models download Qwen/Qwen3-0.6B-GGUF` |

## Documentation

Full documentation is available in the `docs/` directory:

- [OpenAI Compatibility](docs/openai-compatibility.md)
- [Using the ROSE CLI](docs/using-the-rose-cli.md)
- [Available Models](docs/available-models.md)
- [API Reference](docs/api-reference.md)
- [Development](docs/development.md)
- [License](docs/license.md)
