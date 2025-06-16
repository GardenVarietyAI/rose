## Using the ROSE CLI

The `rose` CLI provides a convenient command-line interface with enhanced features beyond the OpenAI CLI.

### Basic Chat

```bash
# Simple chat
poetry run rose chat "What is Python?"

# With custom model
poetry run rose chat "Explain decorators" --model qwen-coder

# With system prompt
poetry run rose chat "Fix this code: print(hello)" --system "You are a Python expert"

# Streaming response (default behavior)
poetry run rose chat "Write a long story"

# Without streaming
poetry run rose chat "Give me a quick answer" --no-stream
```

### Responses API (Stateless)

```bash
# Create a response
poetry run rose responses create "What is recursion?"

# Store for later retrieval
poetry run rose responses create "Explain async/await" --store
# Output: Response stored with ID: resp_abc123...

# Retrieve stored response
poetry run rose responses retrieve resp_abc123

# Stream response
poetry run rose responses create "Generate a Python class" --stream

# With custom instructions
poetry run rose responses create "List colors" --instructions "Be brief, max 3 items"

# Test storage functionality
poetry run rose responses test-storage
```

### Model Management

```bash
# List available models
poetry run rose models list

# Get model details
poetry run rose models get qwen-coder

# Pull a new model (if supported)
poetry run rose models pull phi-4
```

### File Operations

```bash
# Upload a file
poetry run rose upload training_data.jsonl --purpose fine-tune

# Or using the files command
poetry run rose files upload training_data.jsonl --purpose fine-tune

# List files
poetry run rose files list

# Get file details
poetry run rose files get file_abc123

# Delete a file
poetry run rose files delete file_abc123
```
