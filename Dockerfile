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

# custom_rasterizer: try the wheel from the cloned repo first, then download
# the pre-built wheel directly (LFS pointers may not resolve in shallow clones),
# then build from source as last resort.  The runtime base image lacks nvcc so
# building from source will fail — the pre-built wheel is the expected path.
RUN (ls -1 custom_rasterizer-*.whl >/dev/null 2>&1 && pip install custom_rasterizer-*.whl) || true
RUN if ! python -c "import custom_rasterizer" 2>/dev/null; then \
      echo "wheel from clone failed, downloading pre-built wheel..." \
      && wget -q https://huggingface.co/spaces/tencent/Hunyuan3D-2.1/resolve/main/custom_rasterizer-0.1-cp310-cp310-linux_x86_64.whl \
      && pip install custom_rasterizer-0.1-cp310-cp310-linux_x86_64.whl \
      && rm -f custom_rasterizer-0.1-cp310-cp310-linux_x86_64.whl; \
    fi
RUN if ! python -c "import custom_rasterizer" 2>/dev/null; then \
      echo "pre-built wheel failed, trying source build..." \
      && (cd hy3dpaint/packages/custom_rasterizer && pip install -e .) || echo "custom_rasterizer build FAILED"; \
    fi
# Build mesh_inpaint_processor (pybind11 C++ extension, no CUDA needed).
# Optional — guarded import in MeshRender.py, so non-fatal if it fails.
RUN pip install pybind11 && \
    if [ -f hy3dpaint/DifferentiableRenderer/compile_mesh_painter.sh ]; then \
      (cd hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh) || echo "mesh_inpaint_processor build skipped"; \
    fi

COPY requirements-docker.txt /app/requirements-docker.txt
RUN pip install -r /app/requirements-docker.txt

# Verify critical native extensions now that all deps (including torch) are installed
RUN python -c "import custom_rasterizer; print('custom_rasterizer OK')"

# Model weights are NOT baked into the image — they are downloaded at
# container startup via huggingface_hub.snapshot_download() and cached in
# HF_HOME.  This keeps the Docker image small and fast to pull.
RUN mkdir -p /app/.cache/huggingface

# Patch mesh_utils.py: replace bpy-based OBJ→GLB conversion with trimesh
# (bpy/Blender is not installed and is too heavy for this container).
COPY patch_mesh_utils.py /tmp/patch_mesh_utils.py
RUN python /tmp/patch_mesh_utils.py && rm /tmp/patch_mesh_utils.py

COPY worker.py /app/worker.py
COPY server.py /app/server.py

RUN mkdir -p /outputs

EXPOSE 8000
CMD ["python", "server.py"]

