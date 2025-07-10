FROM python:3.13.1-slim

# Install runtime dependencies and build tools needed for sentencepiece
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install --no-cache-dir poetry==1.8.2

WORKDIR /app

# Copy dependency files first (for caching)
COPY pyproject.toml poetry.lock ./

# Install dependencies using poetry
# This layer will be cached as long as pyproject.toml and poetry.lock don't change
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# Copy source code
COPY src/ ./src/

# Set Python path
ENV PYTHONPATH=/app/src
