## OpenAI Compatibility

ROSE implements a subset of OpenAI's API, supporting common use cases with the official OpenAI Python client.

### Using the OpenAI Python Client

```python
from openai import OpenAI

# Configure client to use local ROSE service
client = OpenAI(
    api_key="dummy-key",  # Any value works for local service
    base_url="http://localhost:8004/v1"
)

# Chat completions
response = client.chat.completions.create(
    model="qwen2.5-0.5b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"}
    ],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="")

# Function calling
response = client.chat.completions.create(
    model="qwen-coder",
    messages=[{"role": "user", "content": "What's the weather in Boston?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }
    }]
)

# Embeddings
embeddings = client.embeddings.create(
    model="text-embedding-ada-002",
    input=["Hello world", "How are you?"]
)

# Files
file = client.files.create(
    file=open("training_data.jsonl", "rb"),
    purpose="fine-tune"
)

# Fine-tuning
job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="qwen2.5-0.5b",
    suffix="my-model"
)

# Assistants
assistant = client.beta.assistants.create(
    name="Code Helper",
    instructions="You are a helpful coding assistant.",
    model="qwen-coder"
)

# Threads
thread = client.beta.threads.create()
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="How do I sort a list in Python?"
)
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id
)
```

### Using the OpenAI CLI

The official OpenAI CLI works with ROSE for supported operations:

```bash
# Set environment variables
export OPENAI_API_KEY="any-value"
export OPENAI_BASE_URL="http://localhost:8004/v1"

# List models
openai api models.list

# Chat completion
openai api chat.completions.create \
  -m qwen2.5-0.5b \
  -g user "Hello, how are you?"

# Streaming chat
openai api chat.completions.create \
  -m qwen2.5-0.5b \
  -g user "Count to 5" \
  --stream

# Multi-turn conversation
openai api chat.completions.create \
  -m qwen2.5-0.5b \
  -g system "You are a helpful assistant" \
  -g user "What is 2+2?"

# File operations
openai api files.create -f training.jsonl -p fine-tune
openai api files.list
openai api files.retrieve -i file_abc123
openai api files.delete -i file_abc123
```

**Note**: The OpenAI CLI has limited command coverage. For full functionality, use the Python client or direct API calls.

### API Compatibility Matrix

| Feature | ROSE API | OpenAI Python SDK | OpenAI CLI |
|---------|----------|-------------------|------------|
| Chat Completions | Yes | Yes | Yes |
| Function Calling | Yes | Yes | Yes |
| Streaming | Yes | Yes | Yes |
| Embeddings | Yes | Yes | No |
| Files | Yes | Yes | Yes |
| Fine-tuning | Yes | Yes | No |
| Assistants | Yes | Yes | No |
| Threads | Yes | Yes | No |
| Vector Stores | Yes | Yes | No |
| Legacy Completions | No | No | No |

### What's NOT Implemented

- **Vision/Images** - No GPT-4V style image inputs or DALL-E generation
- **Audio** - No Whisper transcription or TTS
- **Code Interpreter** - Assistants can't execute code
- **Newer Features** - No o1 models, structured outputs, or reasoning tokens
- **Security** - No API keys, rate limiting, or user isolation
- **Legacy Completions** - Only chat completions endpoint supported

### Using with Other OpenAI-Compatible Tools

ROSE works with any tool that supports OpenAI API configuration:

```bash
# LangChain
export OPENAI_API_BASE="http://localhost:8004/v1"

# LlamaIndex
export OPENAI_API_BASE="http://localhost:8004/v1"

# Continue (VS Code extension)
# Set API base URL in settings to: http://localhost:8004/v1

# Cursor
# Use http://localhost:8004/v1 as the API endpoint
```