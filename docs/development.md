### Running Tests

```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_responses.py

# With coverage
poetry run pytest --cov=src/llm_service
```

### Code Quality

```bash
# Run linting
poetry run ruff src/

# Fix linting issues
poetry run ruff src/ --fix

# Install pre-commit hooks
uv run pre-commit install

# Run the pre-commit hooks
uv run pre-commit run --all-files
```

### Building the Inference Service

```bash
cd src/rose_inference_rs
cargo clean
cd ../..
mise dev-metal
# or dev-cpu

# run rust only tests
cargo test --no-default-features
```

### Working with the FastEmbed Fork

The fork adds Qwen3 support.

```bash
git clone git@github.com:GardenVarietyAI/fastembed.git
git checkout fastembed-qwen3
uv pip install -e ./fastembed --force-reinstall
```
