## OpenAI Compatibility

ROSE provides a simplified OpenAI-compatible API focused on inference, embeddings, reranking, and vector stores.

### Using the OpenAI Python Client

```python
from openai import OpenAI

# Configure client to use local ROSE service
client = OpenAI(
    api_key="dummy-key",  # Any value works for local service
    base_url="http://localhost:8004/v1"
)

# Inference with responses endpoint
response = client.with_raw_response.post(
    "/responses",
    json={
        "model": "Qwen/Qwen3-1.7B",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python?"}
        ],
        "stream": False
    }
)

# Embeddings
embeddings = client.embeddings.create(
    model="Qwen/Qwen3-Embedding-0.6B",
    input=["Hello world", "How are you?"]
)

# Files
file = client.files.create(
    file=open("data.jsonl", "rb"),
    purpose="assistants"
)
```

### Direct API Calls

```bash
# Inference
curl -X POST http://localhost:8004/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-1.7B",
    "messages": [
      {"role": "user", "content": "What is Python?"}
    ],
    "stream": false
  }'

# Embeddings
curl -X POST http://localhost:8004/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "input": ["Hello world"]
  }'

# List models
curl http://localhost:8004/v1/models
```
