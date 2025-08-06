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
```
