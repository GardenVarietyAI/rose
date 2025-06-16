## Development

```
feat:     new user-visible feature
fix:      bugfix or behavior correction
chore:    tooling, infra, or cleanup
refactor: internal code improvement, no behavior change
docs:     changes to documentation only
test:     test additions or changes
perf:     performance improvement
build:    for packaging, dependencies
ci:       for CI config
style:    for formatting-only changes (e.g. Prettier runs)
plan:     planning docs
```

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
