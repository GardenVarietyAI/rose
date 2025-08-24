# Build stage
FROM python:3.12-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy build files
COPY pyproject.toml README.md ./
COPY src/rose_inference_rs/ ./src/rose_inference_rs/
COPY candle/ ./candle/

# Build Python wheel with Rust extension
RUN uv build --wheel

# Runtime stage
FROM python:3.12-slim as runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy Python project files (excluding rose_inference)
COPY pyproject.toml README.md ./
COPY src/rose_server/ ./src/rose_server/
COPY src/rose_cli/ ./src/rose_cli/
COPY src/rose_trainer/ ./src/rose_trainer/

# Copy built wheel from builder stage
COPY --from=builder /app/dist/*.whl /tmp/

# Install dependencies and built wheel
RUN uv sync --no-dev && \
    uv pip install /tmp/*.whl && \
    rm /tmp/*.whl

ENV PYTHONPATH=/app/src
