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

### Completions

```bash
# Generate a completion
poetry run rose completions "Once upon a time"

# With custom parameters
poetry run rose completions "def fibonacci(" --model qwen-coder --max-tokens 200

# Stream the completion
poetry run rose completions "Write a poem" --stream

# Echo the prompt in response
poetry run rose completions "The sky is" --echo
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
poetry run rose files upload training_data.jsonl --purpose fine-tune

# List files
poetry run rose files list

# Get file details
poetry run rose files get file_abc123

# Delete a file
poetry run rose files delete file_abc123
```

### Fine-Tuning

```bash
# List fine-tuning jobs
poetry run rose finetune list

# Create a fine-tuning job
poetry run rose finetune create --model qwen2.5-0.5b --file file_abc123 --suffix custom

# Get job details
poetry run rose finetune get job_xyz789

# List model checkpoints
poetry run rose finetune checkpoints job_xyz789

# Test a fine-tuned model
poetry run rose finetune test ft:qwen2.5-0.5b:custom:job_xyz789 "Your test prompt"

# Export a model
poetry run rose finetune export ft:qwen2.5-0.5b:custom:job_xyz789 ./exported_model

# Convert model format
poetry run rose finetune convert job_xyz789 --format gguf

# Cancel a running job
poetry run rose finetune cancel job_xyz789
```

### Thread Management (Assistants API)

```bash
# List threads
poetry run rose threads list

# Create a thread
poetry run rose threads create

# Get thread details
poetry run rose threads get thread_abc123

# Add a message to thread
poetry run rose threads add-message thread_abc123 "Hello, assistant!"

# List thread messages
poetry run rose threads list-messages thread_abc123

# Delete a thread
poetry run rose threads delete thread_abc123
```

### Assistant Management

```bash
# List assistants
poetry run rose assistants list

# Create an assistant
poetry run rose assistants create --name "Code Helper" --model qwen-coder

# Get assistant details
poetry run rose assistants get asst_abc123

# Update an assistant
poetry run rose assistants update asst_abc123 --name "Python Expert"

# Delete an assistant
poetry run rose assistants delete asst_abc123
```

### Evaluations

```bash
# List evaluations
poetry run rose evals list

# Create an evaluation
poetry run rose evals create --name "Color Recognition" --file eval_data.jsonl

# Get evaluation details
poetry run rose evals get eval_abc123

# Run an evaluation
poetry run rose evals run eval_abc123 --model qwen2.5-0.5b

# Delete an evaluation
poetry run rose evals delete eval_abc123
```

### Compare Models

```bash
# Compare local vs remote model responses
poetry run rose compare "What is machine learning?"

# With custom models
poetry run rose compare "Explain Python decorators" --local-model qwen-coder --remote-model gpt-4o

# With system prompt
poetry run rose compare "Fix this bug" --system "You are a debugging expert"
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

### Cleanup Operations

```bash
# Clean up fine-tuned models
poetry run rose cleanup models

# Clean up uploaded files
poetry run rose cleanup files

# Clean up fine-tuning jobs by status
poetry run rose cleanup jobs --status failed

# Clean up everything
poetry run rose cleanup all
```

### Advanced Usage

```bash
# Use remote endpoint
poetry run rose chat "Hello" --remote --url https://api.openai.com/v1

# Use local endpoint with custom URL
poetry run rose chat "Hello" --local --url http://localhost:8004/v1

# Batch operations with files
poetry run rose files list | grep training | xargs -I {} poetry run rose files delete {}
```

### Tips

1. **Default Behavior**: Most commands default to local service (http://localhost:8004/v1)
2. **Streaming**: Chat defaults to streaming, completions do not
3. **Model Selection**: If not specified, uses the server's default model
4. **File Formats**: Fine-tuning expects JSONL format with OpenAI-style messages
5. **Job Monitoring**: Fine-tuning jobs run asynchronously, use `finetune get` to check status
