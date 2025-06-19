#!/usr/bin/env bash
set -e

# Determine project root and ensure we're in repo root
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(dirname "$dir")"
cd "$project_root"

# 1. Ensure project dependencies are installed via uv
echo "Building project environment..."
uv build

# 2. Start Redis in Docker (reuse if exists)
container=redis
image=deepvoice-redis

if docker ps -q -f name="^${container}$" | grep -q .; then
  echo "Redis container already running, skipping start."
else
  echo "Building and starting Redis container..."
  docker build -t $image ./infra/redis
  docker run -d --name $container -e REDIS_PASSWORD="${REDIS_PASSWORD:-deepvoicepass}" -p 6379:6379 $image
fi

# 3. Run Celery worker and API server using uv
echo "Starting Celery worker (dvworker)..."
uv run dvworker &

echo "Starting API server (dvapi)..."
uv run dvapi &

# 4. Wait for services
echo "All services started. Press Ctrl+C to exit."
wait 