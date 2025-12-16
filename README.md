# ^ROSE

 [![CI](https://github.com/GardenVarietyAI/rose-server/actions/workflows/ci.yml/badge.svg)](https://github.com/GardenVarietyAI/rose-server/actions/workflows/ci.yml)

> This project is under active development. Expect rapid changes and improvements.

A FastAPI wrapper around llama-cpp-python and SQLite.

## Quick Start

### Installation

```bash
# Clone the repository
git clone --branch 0.1.5 --depth 1 git@github.com:GardenVarietyAI/rose.git
cd rose

# Install dependencies
uv sync
```

### Download Models

```bash
# Download models from HuggingFace
hf download Qwen/Qwen3-0.6B-GGUF qwen3-0.6b-q8_0.gguf
```

### Running the Server

```bash
uv run rose-server
```

Server runs on http://localhost:8004. Database files are created automatically in the project root.

The spell check dictionary (MIT licensed) is downloaded automatically on first run.

## API Endpoints

- `POST /v1/chat/completions` - Chat with streaming support
- `GET /v1/models` - List available models

## Rebuild Search Index

1. Stop the server

2. Drop the messages table

```sql
DROP TABLE messages_fts;
```

3. Start the server to recreate the table

```sh
uv run rose-server
```

4. Rebuild the index

```sql
INSERT INTO messages_fts(messages_fts) VALUES('rebuild');
```
