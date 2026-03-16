# ---------- Stage 1: build CUDA/C++ extensions with nvcc ----------
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3-pip \
      git git-lfs build-essential ninja-build pybind11-dev \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

# Torch (needed to compile CUDA extensions against)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

WORKDIR /build

# Pull the Space code
RUN git clone --depth 1 https://huggingface.co/spaces/tencent/Hunyuan3D-2.1 repo \
    && mv repo/* . \
    && mv repo/.git* . 2>/dev/null || true \
    && rm -rf repo

# Build custom_rasterizer from source (needs nvcc).
# TORCH_CUDA_ARCH_LIST avoids GPU auto-detection (no GPU at build time).
# Covers Ampere (8.0, 8.6), Ada Lovelace (8.9), Hopper (9.0) — i.e. A100, RTX 3090, RTX 4090, H100.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
RUN cd hy3dpaint/packages/custom_rasterizer && pip install .

# Build mesh_inpaint_processor (pybind11 C++, no CUDA)
# Ensure python3-config exists (compile_mesh_painter.sh uses it)
RUN ln -sf /usr/bin/python3.10-config /usr/bin/python3-config 2>/dev/null || true
RUN pip install pybind11 && \
    if [ -f hy3dpaint/DifferentiableRenderer/compile_mesh_painter.sh ]; then \
      (cd hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh) || echo "mesh_inpaint_processor build skipped"; \
    fi

# Verify
RUN python -c "import torch; import custom_rasterizer; print('custom_rasterizer OK')"

# ---------- Stage 2: runtime image ----------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
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

# Copy the Space code from builder (avoids re-cloning)
COPY --from=builder /build/ /app/

# Copy the built custom_rasterizer + kernel .so from builder's site-packages
COPY --from=builder /usr/local/lib/python3.10/dist-packages/custom_rasterizer/ \
                    /usr/local/lib/python3.10/dist-packages/custom_rasterizer/
COPY --from=builder /usr/local/lib/python3.10/dist-packages/custom_rasterizer_kernel.cpython-310-x86_64-linux-gnu.so \
                    /usr/local/lib/python3.10/dist-packages/

COPY requirements-docker.txt /app/requirements-docker.txt
RUN pip install -r /app/requirements-docker.txt

# Verify custom_rasterizer works with this torch (must import torch first for libc10.so)
RUN python -c "import torch; import custom_rasterizer; print('custom_rasterizer OK')"

# Model weights downloaded at startup via huggingface_hub
RUN mkdir -p /app/.cache/huggingface

# Patch mesh_utils.py: replace bpy-based OBJ→GLB conversion with trimesh
COPY patch_mesh_utils.py /tmp/patch_mesh_utils.py
RUN python /tmp/patch_mesh_utils.py && rm /tmp/patch_mesh_utils.py

COPY worker.py /app/worker.py
COPY server.py /app/server.py

RUN mkdir -p /outputs

EXPOSE 8000
CMD ["python", "server.py"]
