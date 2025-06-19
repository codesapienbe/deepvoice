FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files into container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir .

# Expose FastAPI port
EXPOSE 8000

# Start API service
CMD ["dvapi"] 