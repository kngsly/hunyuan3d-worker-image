#!/usr/bin/env python3
"""
Shape-only Hunyuan3D-2.1 worker.

Why shape-only:
  - texture pipeline in the upstream Space often pulls in Blender `bpy`, which is unreliable to install.
  - for your goal (image -> glb), shape-only is the most reliable baseline.
"""

from __future__ import annotations

import os
import sys
import threading
import traceback
import time
import uuid
from pathlib import Path

from PIL import Image


def _lazy_import_pipeline():
    # These imports are heavy; do them only on first request.
    # The Space layout isn't a standard pip package; add local dirs to sys.path.
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/hy3dshape")
    sys.path.insert(0, "/app/hy3dpaint")
    from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline
    return Hunyuan3DDiTFlowMatchingPipeline


_PIPELINE = None
_READY = {
    "status": "not_started",  # not_started | downloading | loading | ready | error
    "detail": "",
    "started_at": 0.0,
    "ready_at": 0.0,
}
_READY_LOCK = threading.Lock()


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    with _READY_LOCK:
        if _READY["status"] == "not_started":
            _READY["status"] = "loading"
            _READY["started_at"] = time.time()
            _READY["detail"] = "initializing pipeline"

    model_path = os.environ.get("HY3D_MODEL_PATH", "tencent/Hunyuan3D-2.1")
    # The Space repo expects this subfolder for 2.1.
    subfolder = os.environ.get("HY3D_SUBFOLDER", "hunyuan3d-dit-v2-1")
    device = os.environ.get("HY3D_DEVICE", "cuda")

    Hunyuan3DDiTFlowMatchingPipeline = _lazy_import_pipeline()
    _PIPELINE = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_path,
        subfolder=subfolder,
        # The official Space currently provides FP16 weights as a .ckpt, not .safetensors.
        # If we set use_safetensors=True, smart_load_model will look for model.fp16.safetensors and fail.
        use_safetensors=False,
        device=device,
    )
    with _READY_LOCK:
        _READY["status"] = "ready"
        _READY["detail"] = "pipeline loaded"
        _READY["ready_at"] = time.time()
    return _PIPELINE


def get_ready_state() -> dict:
    with _READY_LOCK:
        return dict(_READY)


def _preload_worker():
    try:
        print("[worker] preload: starting model preload", flush=True)
        _get_pipeline()
        st = get_ready_state()
        dt = 0.0
        if st.get("started_at") and st.get("ready_at"):
            dt = float(st["ready_at"]) - float(st["started_at"])
        print(f"[worker] preload: ready (load_time_sec={dt:.1f})", flush=True)
    except Exception:
        tb = traceback.format_exc()
        with _READY_LOCK:
            _READY["status"] = "error"
            _READY["detail"] = tb[-4000:] if tb else "unknown error"
        print("[worker] preload: ERROR\n" + (tb or "unknown error"), flush=True)


def start_preload_in_background():
    """
    Start a background thread to load the model so /ready can become true without waiting for the first /generate.
    Controlled by HY3D_PRELOAD=1/0 (default 1).
    """
    v = (os.environ.get("HY3D_PRELOAD", "1") or "1").strip().lower()
    if v in ("0", "false", "no", "off"):
        return
    with _READY_LOCK:
        if _READY["status"] != "not_started":
            return
        _READY["status"] = "downloading"
        _READY["started_at"] = time.time()
        _READY["detail"] = "preload scheduled"
    t = threading.Thread(target=_preload_worker, daemon=True)
    t.start()


def generate_glb_from_image_bytes(image_bytes: bytes, out_dir: Path) -> Path:
    if not image_bytes:
        raise ValueError("empty image upload")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    img = Image.open(__import__("io").BytesIO(image_bytes)).convert("RGBA")

    st = get_ready_state()
    print(f"[worker] generate: model_status={st.get('status')} detail={st.get('detail')!r}", flush=True)
    pipeline = _get_pipeline()
    mesh = pipeline(image=img)[0]

    fname = f"{uuid.uuid4().hex}.glb"
    out_path = out_dir / fname
    mesh.export(str(out_path))

    # Keep some timing info in logs (uvicorn stdout).
    dt = time.time() - t0
    print(f"[worker] generated {out_path} in {dt:.1f}s (bytes_in={len(image_bytes)})", flush=True)
    return out_path
