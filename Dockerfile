# ── Stage 1: dependency builder ────────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# CPU-only torch keeps the image ~1.5 GB instead of ~5 GB.
# To use GPU, replace this line with:
#   RUN pip install --no-cache-dir torch torchvision
RUN pip install --no-cache-dir --prefix=/install \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
# Install everything else (torch already present, pip skips it)
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Runtime system libraries required by OpenCV + FFmpeg (video frames)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app/        app/
COPY templates/  templates/
COPY static/     static/
COPY configs/    configs/

# Runtime directories (models mounted via volume in production)
RUN mkdir -p models outputs/uploads

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Workers = 1 keeps GPU/model memory sane; bump for CPU-only multi-core boxes
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "30"]
