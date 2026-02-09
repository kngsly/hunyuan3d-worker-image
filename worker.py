#!/usr/bin/env python3
"""
Shape-only Hunyuan3D-2.1 worker.

Why shape-only:
  - texture pipeline in the upstream Space often pulls in Blender `bpy`, which is unreliable to install.
  - for your goal (image -> glb), shape-only is the most reliable baseline.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from PIL import Image


def _lazy_import_pipeline():
    # These imports are heavy; do them only on first request.
    from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline
    return Hunyuan3DDiTFlowMatchingPipeline


_PIPELINE = None


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    model_path = os.environ.get("HY3D_MODEL_PATH", "tencent/Hunyuan3D-2.1")
    # The Space repo expects this subfolder for 2.1.
    subfolder = os.environ.get("HY3D_SUBFOLDER", "hunyuan3d-dit-v2-1")
    device = os.environ.get("HY3D_DEVICE", "cuda")

    Hunyuan3DDiTFlowMatchingPipeline = _lazy_import_pipeline()
    _PIPELINE = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_path,
        subfolder=subfolder,
        use_safetensors=True,
        device=device,
    )
    return _PIPELINE


def generate_glb_from_image_bytes(image_bytes: bytes, out_dir: Path) -> Path:
    if not image_bytes:
        raise ValueError("empty image upload")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    img = Image.open(__import__("io").BytesIO(image_bytes)).convert("RGBA")

    pipeline = _get_pipeline()
    mesh = pipeline(image=img)[0]

    fname = f"{uuid.uuid4().hex}.glb"
    out_path = out_dir / fname
    mesh.export(str(out_path))

    # Keep some timing info in logs (uvicorn stdout).
    dt = time.time() - t0
    print(f"[worker] generated {out_path} in {dt:.1f}s (bytes_in={len(image_bytes)})", flush=True)
    return out_path

