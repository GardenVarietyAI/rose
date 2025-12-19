FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -e ./packages/rose-server && \
    rm -rf /tmp/* /root/.cache

EXPOSE 8004
CMD ["rose-server", "--host", "0.0.0.0", "--port", "8004"]
