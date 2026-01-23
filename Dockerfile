# Base image (CUDA 12.4)
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

# --- CRITICAL FIX: Install PyTorch 2.5.1 ---
# LTX-Video requires 'enable_gqa' in attention, which is only in Torch 2.5+
# We use the cu124 (CUDA 12.4) wheels to match the base image.
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# --- Install requirements ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

# Start
CMD [ "python", "-u", "handler.py" ]