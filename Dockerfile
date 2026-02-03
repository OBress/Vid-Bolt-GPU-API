# =============================================================================
# Vid-Bolt GPU API - Docker Image
# Target: NVIDIA RTX PRO 6000 Blackwell / Ubuntu 22.04 / CUDA 12.8
# =============================================================================

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

LABEL maintainer="Vid-Bolt Team"
LABEL description="GPU-accelerated image/video generation API"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# =============================================================================
# System Dependencies
# =============================================================================
# Add deadsnakes PPA for Python 3.12 (not in Ubuntu 22.04 default repos)
# Note: Using manual PPA addition to avoid Launchpad API dependency
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common gnupg \
    && echo "deb https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy main" > /etc/apt/sources.list.d/deadsnakes.list \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F23C5A6CF475977595C89F51BA6932366A755776 \
    && apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install pip for Python 3.12 (bootstrap via get-pip.py)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12
RUN python -m pip install --upgrade pip wheel setuptools

# =============================================================================
# Working Directory
# =============================================================================
WORKDIR /app

# =============================================================================
# PyTorch Installation (CUDA 12.8 for Blackwell)
# PyTorch 2.9.1 stable is available with cu128 (CUDA 12.8)
# =============================================================================
RUN pip install --no-cache-dir \
    torch==2.9.1 \
    torchvision==0.24.1 \
    torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128

# NOTE: xformers is NOT installed because it doesn't support Blackwell GPUs (compute capability 12.0)
# All libraries (LTX-2, LightX2V, etc.) automatically fall back to PyTorch native SDPA

# Triton - MUST be pinned to 3.5.0 for PyTorch 2.9.1 + SageAttention compatibility
RUN pip install --no-cache-dir triton==3.5.0

# =============================================================================
# Core Dependencies
# =============================================================================
COPY requirements.txt /tmp/requirements.txt

# Filter out torch/torchvision (already installed with CUDA) and install rest
# Also handle Windows line endings with tr -d '\r'
RUN cat /tmp/requirements.txt | tr -d '\r' | \
    grep -v "^torch==" | grep -v "^torchvision" | grep -v "^torchaudio" | grep -v "^#" | grep -v "^$" | \
    grep -v "xformers" | grep -v "triton" | grep -v "win32" > /tmp/filtered_requirements.txt \
    && pip install --no-cache-dir -r /tmp/filtered_requirements.txt

# Diffusers from source (for ZImagePipeline support)
RUN pip install --no-cache-dir git+https://github.com/huggingface/diffusers

# LightX2V - install from local vendored copy (has custom fixes for native resolution)
# Copy and install from local directory instead of GitHub
COPY LightX2V /app/LightX2V
RUN pip install --no-cache-dir -e /app/LightX2V || echo "LightX2V installation skipped"

# sgl-kernel for FP8 quantized inference (provides sgl_per_token_quant_fp8, fp8_scaled_mm)
RUN pip install --no-cache-dir sgl-kernel || echo "sgl-kernel installation skipped"

# SageAttention 2.x for ~2x faster attention on Blackwell GPUs
# Must install from GitHub - version 2.x not available on PyPI (only 1.0.x)
# Uses CUDA backend (not Triton) to avoid black output artifacts on sm_120
# --no-build-isolation: SageAttention setup.py imports torch, so use system torch
# TORCH_CUDA_ARCH_LIST: Required because Docker build has no GPU access
ENV TORCH_CUDA_ARCH_LIST="12.0"
RUN pip install --no-cache-dir --no-build-isolation git+https://github.com/thu-ml/SageAttention.git

# =============================================================================
# Copy Application Code
# =============================================================================

# Copy LTX-2 packages if they exist and install
COPY LTX-2/packages /app/LTX-2/packages
RUN pip install --no-cache-dir -e /app/LTX-2/packages/ltx-core \
    && pip install --no-cache-dir -e /app/LTX-2/packages/ltx-pipelines

# Copy Z-Image (for native inference if needed)
COPY Z-Image /app/Z-Image

# Copy main application
COPY app /app/app
COPY .env.example /app/.env.example

# =============================================================================
# Final PyTorch Version Lock
# Reinstall correct PyTorch/xformers after all other packages to fix any downgrades
# =============================================================================
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.9.1 \
    torchvision==0.24.1 \
    torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128

# NOTE: xformers is intentionally NOT installed - Blackwell GPU compatibility
# PyTorch native SDPA is used instead (automatic fallback in all libraries)

# Upgrade core libraries for Blackwell GPU compatibility (Issue #10)
RUN pip install --no-cache-dir --upgrade transformers peft diffusers accelerate

# =============================================================================
# Environment Configuration
# =============================================================================

# HuggingFace cache - mount volume here for persistence
ENV HF_HOME=/app/models
ENV HF_HUB_CACHE=/app/models/hub
ENV TRANSFORMERS_CACHE=/app/models/transformers
ENV TORCH_HOME=/app/models/torch

# Application settings
ENV MOCK_MODE=false
ENV LOG_LEVEL=INFO

# =============================================================================
# Copy LightX2V source for FP8 converter tool
# =============================================================================
COPY LightX2V /app/LightX2V

# =============================================================================
# Entrypoint Script (handles FP8 conversion on first startup)
# =============================================================================
COPY scripts/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# =============================================================================
# Expose & Run
# =============================================================================
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Entrypoint handles FP8 conversion check
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command (passed to entrypoint)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
