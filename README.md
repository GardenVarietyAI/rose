# ^ROSE

 [![CI](https://github.com/GardenVarietyAI/rose-server/actions/workflows/ci.yml/badge.svg)](https://github.com/GardenVarietyAI/rose-server/actions/workflows/ci.yml)

An LLM server implementing a subset of OpenAI’s API for local model experimentation

This is pre-release software. Use at your own risk!

## Features

- **Local Model Inference** - Hugging Face Transformers
- **Fine-Tuning** - LoRA-based pipeline
- **Vector Storage** - Backed by ChromaDB
- **Embeddings** - Using bge-small-en-v1.5 as the default
- **Streaming Support** - SSE for real-time completions

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
git clone --branch 0.1.1 --depth 1 git@github.com:GardenVarietyAI/rose.git
cd rose

# Install deps
mise install
uv venv
uv pip install --group cli
```

### Initialize Database

```bash
# Set up the database schema using dbmate
mv env.example .env
mkdir -p data
dbmate up
```

### Download a Model

```bash
# Download a Hugging Face model
uv run rose models download Qwen/Qwen2.5-1.5B-Instruct
```

### Running the Service

```bash
# Run via convenience script:
./start.sh
```

### Start a chat

```bash
# Chat with Qwen/Qwen2.5-1.5B-Instruct
uv run rose chat --model Qwen/Qwen2.5-1.5B-Instruct
```

## Documentation

Full documentation is available in the `docs/` directory:

- [OpenAI Compatibility](docs/openai-compatibility.md)
- [Using the ROSE CLI](docs/using-the-rose-cli.md)
- [Assistant Workflows](docs/assistant-workflows.md)
- [Available Models](docs/available-models.md)
- [API Reference](docs/api-reference.md)
- [Development](docs/development.md)
- [License](docs/license.md)
