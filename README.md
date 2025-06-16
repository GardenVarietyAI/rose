# ROSE Server

 [![CI](https://github.com/GardenVarietyAI/rose-server/actions/workflows/ci.yml/badge.svg)](https://github.com/GardenVarietyAI/rose-server/actions/workflows/ci.yml)

An LLM server implementing a subset of OpenAI’s API for local model experimentation

This is pre-release software. Use at your own risk!

## Features

- **OpenAI-Compatible API** - Core endpoints for chat, embeddings, and file management
- **Local Model Inference** - Hugging Face Transformers + PyTorch, GPU-accelerated
- **Fine-Tuning** - LoRA-based pipeline with checkpointing and monitoring
- **Vector Storage** - Integrated ChromaDB for embeddings
- **Embeddings** - Multi-model support with caching
- **Assistants API** - Basic thread/message support with function calling
- **Responses API** - Stateless chat endpoint with optional storage
- **Streaming Support** - SSE for real-time completions

## Documentation

Full documentation is available in the `docs/` directory:

- [OpenAI Compatibility](docs/openai-compatibility.md)
- [Using the ROSE CLI](docs/using-the-rose-cli.md)
- [Fine-Tuning Workflows](docs/fine-tuning-workflows.md)
- [Assistant Workflows](docs/assistant-workflows.md)
- [Model Evaluation](docs/model-evaluation.md)
- [Available Models](docs/available-models.md)
- [API Reference](docs/api-reference.md)
- [Configuration](docs/configuration.md)
- [Development](docs/development.md)
- [Troubleshooting](docs/troubleshooting.md)
- [License](docs/license.md)

## Quick Start

Set your environment variables:

```bash
export OPENAI_API_KEY=sk-dummy-key
export OPENAI_BASE_URL=http://localhost:8004/v1
```

### Prerequisites

- Python 3.13+
- Poetry
- CUDA-capable GPU (recommended) or Apple Silicon Mac
- 16GB+ RAM (32GB recommended for fine-tuning)

### Installation

```bash
# Clone the repository
git clone git@github.com:GardenVarietyAI/rose-server.git
cd rose-server

# Install base dependencies
poetry install

# Optional: CLI tools
poetry install --with cli

# Optional: NVIDIA monitoring
poetry install --with nvidia
```

### Running the Service

```bash
# Start API service (port 8004)
poetry run rose-server

# Start background worker (for fine-tuning and jobs)
poetry run rose-worker

# Or run both via convenience script:
./start.sh
```

On startup, the service will:
- Initialize SQLite database
- Set up ChromaDB (with local fallback)
- Load models
- Start job queue
- Detect hardware and auto-optimize
