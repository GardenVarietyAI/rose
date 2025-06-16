## Assistant Workflows

### Creating and Using Assistants

1. **Create an Assistant**

```bash
# Via API
curl -X POST http://localhost:8004/v1/assistants \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Code Helper",
    "instructions": "You are a helpful coding assistant. Always provide clear explanations with your code.",
    "model": "qwen-coder",
    "tools": [{"type": "file_search"}]
  }'
```

2. **Create a Thread and Run**

```bash
# Create thread
curl -X POST http://localhost:8004/v1/threads \
  -H "Content-Type: application/json" \
  -d '{}'

# Add message to thread
curl -X POST http://localhost:8004/v1/threads/thread_abc123/messages \
  -H "Content-Type: application/json" \
  -d '{
    "role": "user",
    "content": "How do I implement a binary search tree in Python?"
  }'

# Run the assistant
curl -X POST http://localhost:8004/v1/threads/thread_abc123/runs \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "asst_xyz789",
    "stream": true
  }'
```

3. **Retrieve Messages**

```bash
# Get thread messages
curl http://localhost:8004/v1/threads/thread_abc123/messages

# Using rose CLI
poetry run rose threads get thread_abc123
```