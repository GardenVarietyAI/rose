## Fine-Tuning Workflows

### Complete Fine-Tuning Example

1. **Prepare Training Data**

Create a JSONL file with conversation examples:
```jsonl
{"messages": [{"role": "user", "content": "Write a Python function for factorial"}, {"role": "assistant", "content": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"}]}
{"messages": [{"role": "user", "content": "What is recursion?"}, {"role": "assistant", "content": "Recursion is a programming technique where a function calls itself to solve a problem by breaking it down into smaller subproblems."}]}
```

2. **Upload Training Data**

```bash
poetry run rose upload training_data.jsonl --purpose fine-tune
# Output: File uploaded: file_abc123
```

3. **Create Fine-Tuning Job**

```bash
poetry run rose finetune create file_abc123 --model qwen-coder --suffix my-model-v1
# Output: Fine-tuning job created: ftjob_xyz789
```

4. **Monitor Progress**

```bash
# Check job status
poetry run rose finetune get ftjob_xyz789

# Stream training events
poetry run rose finetune events ftjob_xyz789

# List all jobs
poetry run rose finetune list
```

5. **Use the Fine-Tuned Model**

Once complete, the model is automatically available:
```bash
poetry run rose chat "Write factorial in Python" --model qwen-coder-ft-20240611_123456-my-model-v1
```

### Advanced Fine-Tuning Options

```bash
# With custom hyperparameters
poetry run rose finetune create file_abc123 \
  --model qwen-coder \
  --epochs 5 \
  --batch-size 8 \
  --learning-rate 2.0

# Create from Python
curl -X POST http://localhost:8004/v1/fine_tuning/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "training_file": "file_abc123",
    "model": "qwen-coder",
    "suffix": "custom-model",
    "hyperparameters": {
      "n_epochs": 3,
      "batch_size": "auto",
      "learning_rate_multiplier": 1.5,
      "validation_split": 0.1,
      "early_stopping_patience": 3,
      "warmup_ratio": 0.1,
      "use_lora": true,
      "lora_config": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05
      }
    }
  }'
```
