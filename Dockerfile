# Multi-stage Dockerfile for Document Anonymizer

# Stage 1: Builder
FROM python:3.12-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
ENV POETRY_VERSION=1.8.3
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_IN_PROJECT=1
ENV POETRY_VIRTUALENVS_CREATE=1

RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install --upgrade pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

ENV PATH="${PATH}:${POETRY_VENV}/bin"

WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --no-root --only main

# Copy source code
COPY src/ ./src/
COPY main.py ./
COPY README.md ./

# Install the project
RUN poetry install --only main

# Stage 2: Runtime
FROM python:3.12-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /data /models && \
    chown -R appuser:appuser /app /data /models

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/src /app/src
COPY --from=builder --chown=appuser:appuser /app/main.py /app/

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV ANONYMIZER_DATA_DIR="/data"
ENV ANONYMIZER_MODELS_DIR="/models"
ENV ANONYMIZER_CACHE_DIR="/data/cache"

# Switch to non-root user
USER appuser

# Create directories
RUN mkdir -p /data/input /data/output /data/cache /models

# Expose port for REST API (future feature)
EXPOSE 8000

# Default command
CMD ["python", "main.py", "--help"]

# Stage 3: Development (optional)
FROM runtime AS development

USER root

# Install development dependencies
RUN apt-get update && apt-get install -y \
    vim \
    less \
    && rm -rf /var/lib/apt/lists/*

# Copy poetry from builder
COPY --from=builder ${POETRY_VENV} ${POETRY_VENV}
ENV PATH="${PATH}:${POETRY_VENV}/bin"

# Copy all project files
COPY --chown=appuser:appuser . /app/

USER appuser

# Install all dependencies including dev
RUN poetry install

CMD ["/bin/bash"]