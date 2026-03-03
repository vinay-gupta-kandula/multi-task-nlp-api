# Use a stable Python 3.10 base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
# curl: Required for the FastAPI health check in docker-compose
# build-essential: Required if any python packages need to compile C++ extensions
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Environment Variables
# 1. Force logs to stream to the terminal without buffering (essential for tracking training)
ENV PYTHONUNBUFFERED=1
# 2. Set custom cache locations to ensure the container has write permissions
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache
# 3. Ensure the app directory is in the PYTHONPATH
ENV PYTHONPATH=/app

# Leverage Docker layer caching by copying requirements first
COPY requirements.txt .

# Install dependencies
# Using --extra-index-url ensures pip finds the 'torch==2.2.1+cpu' version 
# specifically requested in your requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the entire project into the container
COPY . .

# Create persistent directories for ML artifacts and data
# This prevents permission errors when the training script starts
RUN mkdir -p /app/data/processed /app/mlruns /app/cache && \
    chmod -R 777 /app/mlruns /app/cache

# Expose the port FastAPI runs on
EXPOSE 8000

# Default command: bash. 
# Note: The actual execution logic (train.py then uvicorn) is 
# defined in the 'command' section of your docker-compose.yml.
CMD ["bash"]