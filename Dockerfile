# --- UPGRADED BASE IMAGE ---
# We switch to Torch 2.4.0 + CUDA 12.4 + Python 3.11
# This natively supports the new 'diffusers' features without crashing.
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

# --- Install requirements ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

# Start
CMD [ "python", "-u", "handler.py" ]