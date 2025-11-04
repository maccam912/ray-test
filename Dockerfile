# Custom Ray image without Anaconda, using UV for dependency management
FROM python:3.10-slim

# Install system dependencies required by Ray
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create ray user (to match rayproject conventions)
RUN useradd -m -u 1000 -s /bin/bash ray

# Switch to ray user
USER ray
WORKDIR /home/ray

# Copy project files
COPY --chown=ray:ray pyproject.toml uv.lock ./

# Install dependencies using UV (will use CPU-only PyTorch from config)
RUN uv sync --frozen --no-dev

# Copy the rest of the project
COPY --chown=ray:ray . .

# Set environment variables
ENV PATH="/home/ray/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["bash"]
