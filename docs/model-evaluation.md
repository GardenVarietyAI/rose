## Model Evaluation

ROSE includes a comprehensive evaluation system to test model performance on standard benchmarks.

### Creating and Running Evaluations

1. **Create an Evaluation**

```bash
# Using standard dataset (GSM8K, HumanEval, MMLU)
curl -X POST http://localhost:8004/v1/evals \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Math Problem Solving",
    "description": "Test mathematical reasoning",
    "data_source": {
      "type": "dataset",
      "config": {
        "name": "gsm8k",
        "max_samples": 100
      }
    }
  }'

# Using inline test data (no network required)
curl -X POST http://localhost:8004/v1/evals \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Custom QA Test",
    "data_source": {
      "type": "inline",
      "config": {
        "content": [
          {
            "item": {
              "input": "What is 2 + 2?",
              "expected": "4"
            }
          },
          {
            "item": {
              "input": "What is the capital of France?",
              "expected": "Paris"
            }
          }
        ]
      }
    }
  }'
```

2. **Run an Evaluation**

```bash
# Run evaluation on a model
curl -X POST http://localhost:8004/v1/evals/{eval_id}/runs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Run",
    "model": "qwen-coder"
  }'

# Check run status
curl http://localhost:8004/v1/evals/{eval_id}/runs/{run_id}
```

3. **View Sample-Level Results**

```bash
# Get all samples with scores
curl http://localhost:8004/v1/evals/{eval_id}/runs/{run_id}/samples

# Get only failed samples for debugging
curl http://localhost:8004/v1/evals/{eval_id}/runs/{run_id}/samples?only_failed=true

# Get specific sample details
curl http://localhost:8004/v1/evals/{eval_id}/runs/{run_id}/samples/{sample_id}
```

### Supported Datasets

- **GSM8K** - Grade school math problems
- **HumanEval** - Code generation tasks
- **MMLU** - Massive multitask language understanding

### Evaluation Metrics

Each sample is scored using multiple metrics:
- **exact_match** - Exact string match (normalized)
- **f1** - Token-level F1 score
- **substring_match** - Partial match scoring
- **number_match** - For math problems (GSM8K)
