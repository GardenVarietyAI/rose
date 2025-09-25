# ^ROSE

 [![CI](https://github.com/GardenVarietyAI/rose-server/actions/workflows/ci.yml/badge.svg)](https://github.com/GardenVarietyAI/rose-server/actions/workflows/ci.yml)

> This project is under active development. Expect rapid changes and improvements.

**ROSE** is a local AI provider: A self-hosted OpenAI-compatible API built on the Qwen3 model family.

Run your own LLM server locally with chat completions, embeddings, reranking and fine-tuning.

Why ROSE?

- **Drop-in OpenAI compatible API** - Just change the base URL
- **Built on Qwen3** - Chat, embeddings and reranking
- **Local-first design** - All data stored in SQLite, backed up with Litestream

## Features

- Chat Completions
- Responses & Tool Calling
- LoRA Fine-Tuning
- Embeddings
- Reranker
- SSE Streaming Support

### OpenAI-Compatible Endpoints

- `v1/chat/completions`
- `v1/responses`
- `v1/embeddings`
- `v1/vector_stores`
- `v1/fine_tuning`

### Other Endpoints

- `v1/reranker`

## Quick Start

### Installation

```bash
# Install mise (https://mise.jdx.dev/getting-started.html)
curl https://mise.run | sh

# Clone the repository
git clone --branch 0.1.4 --depth 1 git@github.com:GardenVarietyAI/rose.git
cd rose

# Install deps
mise install
mise exec -- uv venv

# Install all workspace packages and dependencies
uv sync

# Or install specific packages:
uv sync --package rose-server
uv sync --package rose-cli
uv sync --no-dev
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
mise dev

# For metal support instead of CPU
mise dev-metal
```

### Download a Model

```bash
# Download a Hugging Face model
uv run rose models download Qwen/Qwen3-1.7B
```

### Download an Embedding Model

```bash
# The base embedding model is currently required
uv run rose models download Qwen/Qwen3-Embedding-0.6B
uv run rose models download Qwen/Qwen3-Embedding-0.6B-GGUF
```

### Download a Reranker

```bash
# The base reranker model is currently required
uv run rose models download Qwen/Qwen3-Reranker-0.6B
uv run rose models download QuantFactory--Qwen3-Reranker-0.6B-GGUF
```

### Running the Services

```bash
uv run rose-server --port 8004
uv run rose-trainer
```

### Start a chat

```bash
# Chat with Qwen/Qwen3-0.6B
uv run rose chat --model Qwen/Qwen3-1.7B
```

### Codex Configuration

Add a `~/.codex/config.toml` and then add Rose as a provider

```
# Recall that in TOML, root keys must be listed before tables.
model = "Qwen--Qwen3-1.7B-GGUF"
model_provider = "rose"
experimental_instructions_file = "PATH/TO/rose/codex.md"

[model_providers.rose]
name = "ROSE"
base_url = "http://localhost:8004/v1"
wire_api = "responses"
query_params = {}
```

## Model Library

| Model                                 | Size | Download                                                            |
| ------------------------------------- | ---- | ------------------------------------------------------------------- |
| Qwen/Qwen3-0.6B                       | 0.6B | `uv run rose models download Qwen/Qwen3-0.6B`                       |
| Qwen/Qwen3-0.6B-GGUF                  | 0.6B | `uv run rose models download Qwen/Qwen3-0.6B-GGUF`                  |
| Qwen/Qwen3-1.7B                       | 1.7B | `uv run rose models download Qwen/Qwen3-1.7B`                       |
| Qwen/Qwen3-1.7B-Base                  | 1.7B | `uv run rose models download Qwen/Qwen3-1.7B-Base`                  |
| Qwen/Qwen3-1.7B-GGUF                  | 1.7B | `uv run rose models download Qwen/Qwen3-1.7B-GGUF`                  |
| Qwen/Qwen3-4B                         | 4B   | `uv run rose models download Qwen/Qwen3-4B`                         |
| Qwen/Qwen3-4B-GGUF                    | 4B   | `uv run rose models download Qwen/Qwen3-4B-GGUF`                    |
| Qwen/Qwen3-Embedding-0.6B             | 0.6B | `uv run rose models download Qwen/Qwen3-Embedding-0.6B`             |
| Qwen/Qwen3-Embedding-0.6B-GGUF        | 0.6B | `uv run rose models download Qwen/Qwen3-Embedding-0.6B-GGUF`        |
| Qwen/Qwen3-Reranker-0.6B              | 0.6B | `uv run rose models download Qwen/Qwen3-Reranker-0.6B`              |
| QuantFactory/Qwen3-Reranker-0.6B-GGUF | 0.6B | `uv run rose models download QuantFactory/Qwen3-Reranker-0.6B-GGUF` |

## Documentation

Full documentation is available in the `docs/` directory:

- [OpenAI Compatibility](docs/openai-compatibility.md)
- [Using the ROSE CLI](docs/using-the-rose-cli.md)
- [Available Models](docs/available-models.md)
- [API Reference](docs/api-reference.md)
- [Development](docs/development.md)
- [License](docs/license.md)
