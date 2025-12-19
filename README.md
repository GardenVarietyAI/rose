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

Server runs on http://localhost:8004.

## API Endpoints

- `POST /v1/chat/completions` - Generate a completion
- `GET /v1/models` - List available models
- `GET /v1/search` - Search past messages

## NLTK datasets licensing

This repository vendors a small subset of **NLTK data packages** under `vendor/nltk_data/`.

See https://github.com/nltk/nltk_data/blob/gh-pages/DATASET-LICENSES.md for licensing information.
