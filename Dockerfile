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

# --- CRITICAL FIX 1: PYTORCH 2.5.1 ---
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# --- CRITICAL FIX 2: FORCE REINSTALL DIFFUSERS ---
# Added --force-reinstall to guarantee we overwrite the system version
RUN pip install --force-reinstall --no-cache-dir git+https://github.com/huggingface/diffusers.git

# --- Install other requirements ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

# Start
CMD [ "python", "-u", "handler.py" ]