## Configuration

### Environment Variables

```bash
# Data directories
DATA_DIR=./data                    # Base data directory
CHROMA_PERSIST_DIR=./data/chroma   # Vector store location
MODEL_OFFLOAD_DIR=./data/offload   # Model cache directory

# Service configuration
LOG_LEVEL=INFO                     # Logging level
CUDA_VISIBLE_DEVICES=0             # GPU selection

# Model settings
DEFAULT_MODEL=qwen2.5-0.5b         # Default model for inference
MAX_CONCURRENT_INFERENCE=1         # Concurrent inference operations
MAX_CONCURRENT_TRAINING=1          # Concurrent training operations
```

### Hardware Requirements

#### Apple Silicon (M1/M2)
- **Recommended**: 32GB RAM
- **Performance**:
  - Small models (0.5B): 5-10 min/1000 examples
  - Medium models (1.5B) with LoRA: 30-45 min/1000 examples
- **Automatic optimization**: Uses Metal Performance Shaders

#### NVIDIA GPUs
- **RTX 4090 (24GB)**: Can handle 7B models with LoRA
- **RTX 4070/4080 (12-16GB)**: Perfect for 3B models
- **Older GPUs (8-10GB)**: Works well with small models