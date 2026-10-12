FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV LD_LIBRARY_PATH=/opt/conda/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV INSWAPPER_PATH=/models/inswapper_128.onnx

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget unzip ffmpeg libgl1 libglib2.0-0 build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip uninstall -y onnxruntime onnxruntime-gpu || true && \
    python -m pip install \
    "numpy==1.26.4" \
    "opencv-python-headless==4.9.0.80" \
    "onnxruntime-gpu==1.16.3" \
    "insightface==0.7.3" \
    "pillow" \
    "gfpgan==1.3.8" \
    "basicsr==1.4.2" \
    "facexlib==0.3.0" \
    runpod && \
    python -m pip install "numpy==1.26.4" && \
    python -m pip uninstall -y onnxruntime || true

RUN mkdir -p /root/.insightface/models /models && \
    wget -q https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip -O /tmp/buffalo_l.zip && \
    unzip -q /tmp/buffalo_l.zip -d /root/.insightface/models/buffalo_l && \
    rm -f /tmp/buffalo_l.zip && \
    wget -q https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx -O /models/inswapper_128.onnx && \
    wget -q https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth -O /models/GFPGANv1.4.pth && \
    test -s /models/inswapper_128.onnx && \
    test -s /models/GFPGANv1.4.pth

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
