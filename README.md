# ^ROSE

 [![CI](https://github.com/GardenVarietyAI/rose-server/actions/workflows/ci.yml/badge.svg)](https://github.com/GardenVarietyAI/rose-server/actions/workflows/ci.yml)

> This project is under active development. Expect rapid changes and improvements.

A FastAPI wrapper around any OpenAI compatible v1/chat/completions endpoint and SQLite.

## Quick Start

### Installation

```bash
# Clone the repository
git clone --branch 0.1.5 --depth 1 git@github.com:GardenVarietyAI/rose.git
cd rose

# Install dependencies
uv sync
```

### Download Models from HuggingFace

```bash
hf download Qwen/Qwen3-0.6B-GGUF qwen3-0.6b-q8_0.gguf
```

### Running the Server

```bash
uv run rose-server
```

#### Docker Compose

```bash
docker compose up -d
```

#### Standalone Docker (point at an existing LLM server)

```bash
docker build -t rose-server .

docker run --rm -p 8004:8004 \
  -e SETTINGS_ENV_FILE=/dev/null \
  -e OPENAI_BASE_URL=http://host.docker.internal:8080/v1 \
  -e OPENAI_API_KEY=your-key-or-empty \
  -e LLAMA_MODEL_PATH=/models/your-model.gguf \
  -v $(pwd)/rose_20251223.db:/app/rose_20251223.db
  rose-server
```

Server runs on http://localhost:8004.

## API Endpoints

- `POST /v1/chat/completions` - Generate a completion
- `GET /v1/models` - List available models
- `GET /v1/search` - Search past messages

## NLTK datasets licensing

This repository vendors a small subset of **NLTK data packages** under `vendor/nltk_data/`.

See https://github.com/nltk/nltk_data/blob/gh-pages/DATASET-LICENSES.md for licensing information.
