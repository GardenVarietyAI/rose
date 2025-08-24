FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY . .

RUN uv pip install --system .

EXPOSE 8004
ENV ROSE_SERVER_HOST=0.0.0.0
CMD ["rose-server", "--port", "8004", "--host", "0.0.0.0"]
