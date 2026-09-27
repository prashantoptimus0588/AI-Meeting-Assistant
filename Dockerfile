FROM python:3.13-slim

# Prevent Python from creating .pyc files
# and ensure logs appear immediately
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required for audio/video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first for Docker layer caching
COPY pyproject.toml uv.lock ./

# Force CPU-only torch build instead of the default CUDA build
ENV UV_TORCH_BACKEND=cpu

# Install Python dependencies
RUN uv sync --frozen --no-dev

# Copy application source
COPY . .

# Make the virtual environment available
ENV PATH="/app/.venv/bin:$PATH"

# FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]