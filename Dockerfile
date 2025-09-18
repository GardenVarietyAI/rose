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

ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN uv pip install --system --prerelease=allow ./packages/rose-server ./packages/rose-trainer

EXPOSE 8004
ENV ROSE_SERVER_HOST=0.0.0.0
ENV ROSE_SERVER_PORT=8004
CMD ["rose-server"]
