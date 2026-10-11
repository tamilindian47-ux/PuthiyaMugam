FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Python packages
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# 3. Model weights pre-download
RUN mkdir -p /app/checkpoints /root/.insightface/models && \
    wget -q "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx" \
         -O /app/checkpoints/inswapper_128.onnx && \
    wget -q "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" \
         -O /app/checkpoints/GFPGANv1.4.pth && \
    python -c "import insightface; insightface.app.FaceAnalysis(name='buffalo_l', root='/root/.insightface')"

# 4. Copy handler code
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
