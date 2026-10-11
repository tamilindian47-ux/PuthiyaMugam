FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4
ENV ORT_DISABLE_THREAD_AFFINITY=1
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

WORKDIR /app

# 1. Install OS packages, Python 3.10, and C/C++ build tools
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
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# 2. Upgrade pip and install build dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir "numpy==1.26.4" cython

# 3. Install remaining dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 4. Bake models into the image so they are not downloaded on boot
RUN mkdir -p /app/checkpoints /root/.insightface/models && \
    wget -q "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx" \
         -O /app/checkpoints/inswapper_128.onnx && \
    python3 -c "import insightface; insightface.app.FaceAnalysis(name='buffalo_l', root='/root/.insightface', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])"

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
