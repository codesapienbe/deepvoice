FROM python:3.10-slim

# Install ffmpeg for audio conversion
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files into container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir .

# Start Celery worker
CMD ["dvworker"] 