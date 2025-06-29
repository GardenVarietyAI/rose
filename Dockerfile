FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    cmake \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install poetry
RUN pip install --no-cache-dir poetry==1.8.2

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Configure poetry to not create virtual environments
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-interaction --no-ansi --no-root

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY README.md ./

# Install the project
RUN poetry install --no-interaction --no-ansi --only-root

# Set environment variables
ENV PYTHONPATH=/app/src
ENV ROSE_SERVER_HOST=0.0.0.0
ENV ROSE_SERVER_PORT=8004
ENV ROSE_SERVER_DATA_DIR=/app/data
ENV ROSE_SERVER_CHROMADB_PATH=/app/data/chroma
ENV ROSE_SERVER_MODEL_CACHE_DIR=/app/data/models
ENV ROSE_SERVER_FINE_TUNING_CHECKPOINT_DIR=/app/data/fine_tuning_checkpoints

# Create data directories
RUN mkdir -p /app/data/chroma /app/data/models /app/data/fine_tuning_checkpoints

# Expose port
EXPOSE 8004

# Default command
CMD ["poetry", "run", "uvicorn", "rose_server.app:app", "--host", "0.0.0.0", "--port", "8004"]
