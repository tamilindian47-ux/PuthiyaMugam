FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Install OS packages & Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# 2. Upgrade pip and enforce NumPy 1.x
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir "numpy==1.26.4"

# 3. Install remaining locked dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 4. Bake model weights directly into container image
RUN mkdir -p /app/checkpoints /root/.insightface/models && \
    wget -q "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx" \
         -O /app/checkpoints/inswapper_128.onnx && \
    python -c "import insightface; insightface.app.FaceAnalysis(name='buffalo_l', root='/root/.insightface')"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
