FROM runpod/base:0.4.0-cuda11.8.0

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
    unzip \
    && rm -rf /var/lib/apt/lists/*

# 2. Pin NumPy 1.x and install onnxruntime-gpu
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir "numpy==1.26.4" cython && \
    python3 -m pip install --no-cache-dir \
        onnxruntime-gpu \
        insightface==0.7.3 \
        opencv-python-headless \
        pillow \
        requests \
        tqdm \
        runpod && \
    python3 -m pip install --no-cache-dir "numpy==1.26.4"

# 3. Bake Inswapper model into image
RUN mkdir -p /app/checkpoints && \
    wget --progress=dot:giga --tries=5 --timeout=60 \
      -O /app/checkpoints/inswapper_128.onnx \
      "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"

# 4. Download and extract buffalo_l manually
RUN mkdir -p /root/.insightface/models/buffalo_l && \
    ( curl -fL --retry 5 --retry-delay 5 -o /tmp/buffalo_l.zip "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip" || \
      curl -fL --retry 5 --retry-delay 5 -o /tmp/buffalo_l.zip "https://huggingface.co/techwavelaps/aceplus-face/resolve/main/insightface/models/buffalo_l.zip" ) && \
    unzip -q -o /tmp/buffalo_l.zip -d /root/.insightface/models/buffalo_l && \
    rm -f /tmp/buffalo_l.zip

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
