FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH

WORKDIR /app

# 1. System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    wget \
    curl \
    ca-certificates \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

# 2. Upgrade pip and install build dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir "numpy==1.26.4" cython

# 3. Install PyTorch with CUDA 12 first (provides all runtime cuDNN and CUDA bindings)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# 4. Install remaining dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir "numpy==1.26.4"

# 5. Bake Inswapper model into image
RUN mkdir -p /app/checkpoints && \
    wget --progress=dot:giga --tries=5 --timeout=60 \
      -O /app/checkpoints/inswapper_128.onnx \
      "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"

# 6. Download and extract buffalo_l manually
RUN mkdir -p /root/.insightface/models/buffalo_l && \
    ( curl -fL --retry 5 --retry-delay 5 -o /tmp/buffalo_l.zip "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip" || \
      curl -fL --retry 5 --retry-delay 5 -o /tmp/buffalo_l.zip "https://huggingface.co/techwavelaps/aceplus-face/resolve/main/insightface/models/buffalo_l.zip" ) && \
    unzip -q -o /tmp/buffalo_l.zip -d /root/.insightface/models/buffalo_l && \
    rm -f /tmp/buffalo_l.zip

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
