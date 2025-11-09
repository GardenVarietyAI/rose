## Using the ROSE CLI

The `rose` CLI provides commands for managing models, files, vectors stores, and making inference requests.

### Model Management

```bash
# List available models
uv run rose models list

# Download a model
uv run rose models download Qwen/Qwen3-1.7B

# Get model details
uv run rose models get Qwen/Qwen3-1.7B
```

### File Operations

```bash
# Upload a file
uv run rose files upload data.jsonl

# List files
uv run rose files list

# Get file details
uv run rose files get file_abc123

# Delete a file
uv run rose files delete file_abc123
```

### Vector Store Management

```bash
# Create a vector store
uv run rose vectorstores create my-store

# List vector stores
uv run rose vectorstores list

# Add files to vector store
uv run rose vectorstores add store_123 --file file_abc123

# Search a vector store
uv run rose vectorstores search store_123 --query "find documents about Python"
```

### Making Inference Requests

```bash
# Create a response
uv run rose responses create --model Qwen/Qwen3-1.7B --input "What is Python?"

# Stream a response
uv run rose responses create --model Qwen/Qwen3-1.7B --input "Write a story" --stream

# With system instructions
uv run rose responses create --model Qwen/Qwen3-1.7B --input "Code review" --system "You are an expert code reviewer"

# Retrieve stored response
uv run rose responses get response_xyz789
```

### Embeddings

```bash
# Generate embeddings
uv run rose embeddings create --model Qwen/Qwen3-Embedding-0.6B --input "Hello world"
```

### Reranking

```bash
# Rerank documents
uv run rose rerank --model Qwen/Qwen3-Reranker-0.6B --query "What is AI?" --documents doc1.txt doc2.txt
```

### Exploring Available Actors

```bash
# List available actors (models that can be used)
uv run rose actors list
```
