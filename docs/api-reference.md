## API Reference

### Inference

#### Responses API
```bash
POST /v1/responses
{
  "model": "Qwen/Qwen3-1.7B",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What is Python?"}
  ],
  "temperature": 0.7,
  "stream": true
}

GET /v1/responses/{response_id}
GET /v1/responses/chains
```

### Embeddings

```bash
POST /v1/embeddings
{
  "model": "Qwen/Qwen3-Embedding-0.6B",
  "input": ["Hello world", "How are you?"]
}
```

### Reranking

```bash
POST /v1/reranker
{
  "model": "Qwen/Qwen3-Reranker-0.6B",
  "query": "What is machine learning?",
  "documents": [
    {"text": "Machine learning is..."},
    {"text": "Deep learning is..."}
  ]
}
```

### Models

```bash
# List models
GET /v1/models

# Get model details
GET /v1/models/{model_id}

# Create/register a model
POST /v1/models
{
  "id": "my-model",
  "name": "My Custom Model",
  "type": "language_model",
  "config": {...}
}

# Delete a model
DELETE /v1/models/{model_id}
```

### Files

```bash
# Upload file
POST /v1/files
multipart/form-data: {
  "file": <binary>,
  "purpose": "assistants"
}

# List files
GET /v1/files

# Get file info
GET /v1/files/{file_id}

# Delete file
DELETE /v1/files/{file_id}
```

### Vector Stores

```bash
# Create vector store
POST /v1/vector_stores
{
  "name": "my-store",
  "file_ids": ["file_123", "file_456"]
}

# List vector stores
GET /v1/vector_stores

# Get vector store details
GET /v1/vector_stores/{vector_store_id}

# Delete vector store
DELETE /v1/vector_stores/{vector_store_id}

# Add files to vector store
POST /v1/vector_store_files
{
  "vector_store_id": "vs_123",
  "file_ids": ["file_789"]
}

# List files in vector store
GET /v1/vector_stores/{vector_store_id}/files
```

### Health & Info

```bash
# Health check
GET /health
```
