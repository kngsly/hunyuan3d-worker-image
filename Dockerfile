FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    # Avoid host-specific TBB issues in numba parallel code paths.
    NUMBA_THREADING_LAYER=workqueue \
    HF_HOME=/app/.cache/huggingface \
    HY3DGEN_MODELS=/app/.cache/hy3dgen

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3.10-dev python3-pip \
      git git-lfs curl wget ca-certificates \
      build-essential ninja-build \
      libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Torch first (CUDA 12.1)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Pull the Space code so we match the model layout / pipeline code.
RUN git clone --depth 1 https://huggingface.co/spaces/tencent/Hunyuan3D-2.1 repo \
    && mv repo/* . \
    && mv repo/.git* . 2>/dev/null || true \
    && rm -rf repo

# If present in the Space, install optional wheel (ignore failures).
RUN (ls -1 custom_rasterizer-*.whl >/dev/null 2>&1 && pip install custom_rasterizer-*.whl) || true

COPY requirements-docker.txt /app/requirements-docker.txt
RUN pip install -r /app/requirements-docker.txt

# Do NOT download model weights at build time:
# - it makes CI flaky (rate limits / transient network)
# - it bloats the image and slows down pushes/pulls
# We download lazily on the first /generate request (HF cache under HF_HOME).
RUN python -c "import os; os.makedirs('/app/.cache/huggingface', exist_ok=True)"

COPY worker.py /app/worker.py
COPY server.py /app/server.py

RUN mkdir -p /outputs

EXPOSE 8000
CMD ["python", "server.py"]
