FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Force NumPy 1.x to prevent PyTorch C-ABI incompatibility
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir "numpy==1.26.4"

# 3. Python packages
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 4. Patch basicsr for torchvision compatibility
RUN python -c "import site, os; \
    paths = site.getsitepackages(); \
    f = [os.path.join(p, 'basicsr/data/degradations.py') for p in paths if os.path.exists(os.path.join(p, 'basicsr/data/degradations.py'))]; \
    open(f[0], 'w').write(open(f[0]).read().replace('torchvision.transforms.functional_tensor', 'torchvision.transforms.functional')) if f else None"

# 5. Fix onnxruntime-gpu
RUN pip uninstall -y onnxruntime && \
    pip install --no-cache-dir onnxruntime-gpu==1.17.1

# 6. Pre-download weights
RUN mkdir -p /app/checkpoints /root/.insightface/models && \
    wget -q "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx" \
         -O /app/checkpoints/inswapper_128.onnx && \
    wget -q "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" \
         -O /app/checkpoints/GFPGANv1.4.pth && \
    python -c "import insightface; insightface.app.FaceAnalysis(name='buffalo_l', root='/root/.insightface')"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
