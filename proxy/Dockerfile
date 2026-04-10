# Lightweight model-switching proxy
# Python 3 + Docker CLI (for `docker compose up -d` to swap models)

FROM python:3.12-slim AS base

# Docker CE CLI + compose plugin (no daemon)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg \
      -o /etc/apt/keyrings/docker.asc && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.asc] \
      https://download.docker.com/linux/debian bookworm stable" \
      > /etc/apt/sources.list.d/docker.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      docker-ce-cli docker-compose-plugin && \
    rm -rf /var/lib/apt/lists/*

COPY proxy.py /app/proxy.py
WORKDIR /app

EXPOSE 11434

CMD ["python3", "-u", "proxy.py"]
