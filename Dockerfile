FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for health checks and compilation
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Prevent Python from buffering logs (useful for real-time training logs)
ENV PYTHONUNBUFFERED=1
# Centralize Hugging Face cache
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install all dependencies using the CPU-specific index for PyTorch
# This correctly resolves the 'torch==2.2.1+cpu' mentioned in your requirements.txt
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application code
COPY . .

# Ensure all necessary directories exist for data and MLflow
RUN mkdir -p /app/data/processed /app/mlruns /app/cache

# Port for the FastAPI application
EXPOSE 8000

# The startup command is managed by docker-compose.yml
CMD ["bash"]