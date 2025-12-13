# ^ROSE

 [![CI](https://github.com/GardenVarietyAI/rose-server/actions/workflows/ci.yml/badge.svg)](https://github.com/GardenVarietyAI/rose-server/actions/workflows/ci.yml)

> This project is under active development. Expect rapid changes and improvements.

A thin wrapper around llama-cpp-python, FastAPI and SQLite.

## Quick Start

### Installation

```bash
# Clone the repository
git clone --branch 0.1.4 --depth 1 git@github.com:GardenVarietyAI/rose.git
cd rose

# Install dependencies
uv sync
```

### Download Models

```bash
# Download models from HuggingFace
huggingface-cli download Qwen/Qwen3-0.6B-GGUF qwen3-0.6b-q8_0.gguf
huggingface-cli download Qwen/Qwen3-Embedding-0.6B-GGUF qwen3-embedding-0.6b-q8_0.gguf
```

### Running the Server

```bash
uv run rose-server
```

Server runs on http://localhost:8004. Database files are created automatically in the project root.

## API Endpoints

- `POST /v1/chat/completions` - Chat with streaming support
- `POST /v1/embeddings` - Generate embeddings
- `POST /v1/rerank` - Rerank documents by relevance
- `GET /v1/models` - List available models
