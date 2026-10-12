FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Strict dependency lock to prevent NumPy 2.x crash
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "numpy>=1.23.5,<2.0.0" \
    opencv-python-headless \
    onnxruntime-gpu \
    insightface>=0.7.3 \
    gfpgan \
    basicsr \
    facexlib \
    runpod

# Pre-download Buffalo_l models & Inswapper 128
RUN mkdir -p /root/.insightface/models/ && \
    mkdir -p /models && \
    wget -q https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip -O /tmp/buffalo_l.zip && \
    unzip -q /tmp/buffalo_l.zip -d /root/.insightface/models/buffalo_l && \
    rm /tmp/buffalo_l.zip && \
    wget -q https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx -O /models/inswapper_128.onnx && \
    wget -q https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth -O /models/GFPGANv1.4.pth

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
