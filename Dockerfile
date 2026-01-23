# Base image (CUDA 12.4 + Python 3.11 + Torch 2.4)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# System dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libavcodec-extra \
    libsndfile1 \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# --- CRITICAL FIX: PYTORCH 2.5.1 ---
# Installs Torch 2.5.1 (Required for 'enable_gqa' attention features)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# --- CRITICAL FIX: FORCE INSTALL DIFFUSERS FROM GIT ---
# 1. Uninstall any existing version to prevent conflicts
# 2. Install directly from GitHub main branch to get LTXVideoPipeline
RUN pip uninstall -y diffusers || true && \
    pip install --no-cache-dir git+https://github.com/huggingface/diffusers.git

# --- Install other requirements ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

# Start
CMD [ "python", "-u", "handler.py" ]