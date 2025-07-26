FROM python:3.13.1-slim

# Install runtime dependencies and build tools needed for sentencepiece
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml ./

RUN uv sync --no-dev

COPY src/ ./src/

ENV PYTHONPATH=/app/src
