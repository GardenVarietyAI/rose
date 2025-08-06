# ^ROSE

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
- [Assistant Workflows](docs/assistant-workflows.md)
- [Available Models](docs/available-models.md)
- [API Reference](docs/api-reference.md)
- [Development](docs/development.md)
- [License](docs/license.md)

## Quick Start

Set your environment variables:

```bash
export OPENAI_API_KEY=sk-dummy-key
export OPENAI_BASE_URL=http://localhost:8004/v1
```

### Installation

```bash
# Install mise (https://mise.jdx.dev/getting-started.html)
curl https://mise.run | sh

# Clone the repository
git clone git@github.com:GardenVarietyAI/rose.git
cd rose

# Install deps
mise install
uv pip install --group cli
```

### Running the Service

```bash
# Start API service (port 8004)
uv run rose-server

# Start inference server (Port 8005)
uv run rose-inference

# Start trainer
uv run rose-trainer

# Or run all via convenience script:
./start.sh
```

On startup, the service will:
- Initialize SQLite database
- Set up ChromaDB (with local fallback)
- Load models
- Start job queue
- Detect hardware and auto-optimize
