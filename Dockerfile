FROM python:3.10-slim

WORKDIR /app

# System deps required for transformers + datasets
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Prevent python buffering (important for logs in docker)
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache

COPY requirements.txt .

# Install torch CPU first (faster + stable)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create required folders (important for fresh evaluator clone)
RUN mkdir -p /app/data/processed /app/mlruns /app/cache

EXPOSE 8000

# Let docker-compose control runtime command
CMD ["bash"]