# Fine-Tuning Gotchas and Best Practices

## Qwen Model System Prompt Override

When fine-tuning Qwen models, be aware that the tokenizer's chat template automatically adds a default system prompt:

```
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
```

This happens even when you don't provide a system prompt in your training data, which can make your model overly conversational.

### Solution: Override with Empty System Prompt

To train a model for specific tasks (like color-to-hex conversion), include an empty system prompt in your training data:

```json
{
  "messages": [
    {"role": "system", "content": ""},
    {"role": "user", "content": "white"},
    {"role": "assistant", "content": "#FFFFFF"}
  ]
}
```

Or with a task-specific prompt:

```json
{
  "messages": [
    {"role": "system", "content": "Output only hex codes."},
    {"role": "user", "content": "white"},
    {"role": "assistant", "content": "#FFFFFF"}
  ]
}
```

### Testing Your Training Data Format

You can test how your data will be formatted:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Your training example
sample = {"messages": [{"role": "user", "content": "white"}, {"role": "assistant", "content": "#FFFFFF"}]}

# See what the tokenizer actually produces
formatted = tokenizer.apply_chat_template(
    sample["messages"],
    tokenize=False,
    add_generation_prompt=False
)
print(formatted)
```

## Memory Management on Apple Silicon

### Issue: MPS Memory Allocation Failures

Even with 10+ GB available, you may see errors like:
```
MPS backend out of memory (MPS allocated: 17.71 GB, other allocations: 17.13 GB, max allowed: 36.27 GB)
```

### Solutions:

1. **Set MPS High Watermark Ratio** (temporary fix):
   ```bash
   export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
   ```
   ⚠️ Warning: This disables the upper limit and may cause system instability.

2. **Use Smaller Batch Sizes**:
   The hardware optimizer tries to be smart but may overestimate available memory. Force smaller batches:
   ```python
   hyperparameters = {
       "batch_size": 1,  # Instead of "auto"
       "gradient_accumulation_steps": 16  # Increase this to maintain effective batch size
   }
   ```

3. **Clear Memory Between Jobs**:
   Restart the worker between jobs to ensure clean memory state:
   ```bash
   pkill -f rose-worker && sleep 2 && poetry run rose-worker &
   ```

## Other Common Issues

### Chat Templates and Special Tokens

Different models use different chat templates. Always check:
- What special tokens are added
- Whether the model expects specific formatting
- If the tokenizer adds unexpected prefixes/suffixes

### Training Loss Not Decreasing

If your loss plateaus or doesn't decrease:
1. Check your learning rate (try reducing by 10x)
2. Ensure your data is properly formatted
3. Verify the model is actually learning your examples (not just the template)
