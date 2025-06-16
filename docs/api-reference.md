## API Reference

### Core Endpoints

#### Chat Completions
```bash
POST /v1/chat/completions
{
  "model": "qwen-coder",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"}
  ],
  "temperature": 0.7,
  "max_tokens": 500,
  "stream": true
}
```

#### Responses API
```bash
POST /v1/responses
{
  "model": "qwen-coder",
  "input": "Explain Python generators",
  "instructions": "Be concise and provide examples",
  "store": true,
  "stream": false
}

GET /v1/responses/{response_id}
```

#### Embeddings
```bash
POST /v1/embeddings
{
  "model": "text-embedding-3-small",
  "input": ["Hello world", "How are you?"]
}
```

### Fine-Tuning Endpoints

```bash
# Create job
POST /v1/fine_tuning/jobs

# List jobs
GET /v1/fine_tuning/jobs

# Get job details
GET /v1/fine_tuning/jobs/{job_id}

# Stream events
GET /v1/fine_tuning/jobs/{job_id}/events

# Cancel job
POST /v1/fine_tuning/jobs/{job_id}/cancel
```

### File Management

```bash
# Upload file
POST /v1/files

# List files
GET /v1/files

# Get file info
GET /v1/files/{file_id}

# Get file content
GET /v1/files/{file_id}/content

# Delete file
DELETE /v1/files/{file_id}
```

### Evaluation Endpoints

```bash
# Create evaluation
POST /v1/evals

# List evaluations
GET /v1/evals

# Get evaluation details
GET /v1/evals/{eval_id}

# Create evaluation run
POST /v1/evals/{eval_id}/runs

# List evaluation runs
GET /v1/evals/{eval_id}/runs

# Get run details
GET /v1/evals/{eval_id}/runs/{run_id}

# Get sample results
GET /v1/evals/{eval_id}/runs/{run_id}/samples
GET /v1/evals/{eval_id}/runs/{run_id}/samples/{sample_id}
```
