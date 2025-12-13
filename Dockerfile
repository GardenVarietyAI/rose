FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY . .

RUN uv pip install --system ./packages/rose-server && \
    rm -rf /tmp/* /root/.cache

EXPOSE 8004
CMD ["rose-server", "--host", "0.0.0.0", "--port", "8004"]
