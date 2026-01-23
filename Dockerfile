# Base image (Has Torch 2.2.0 + CUDA 12.1 pre-installed)
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

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

# --- REMOVED MANUAL TORCH INSTALLATION ---
# We rely on the base image's Torch to ensure GPU drivers work.

# --- Install requirements ---
COPY requirements.txt .
# We use --no-deps for specific packages if needed, but standard install is usually fine 
# as long as requirements.txt doesn't force a torch upgrade.
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

# Start
CMD [ "python", "-u", "handler.py" ]