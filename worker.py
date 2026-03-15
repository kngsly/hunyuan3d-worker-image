#!/usr/bin/env python3
"""
Hunyuan3D-2.1 worker with texture and background removal support.

Supports:
  - Shape-only generation (default, or when texture flags are absent)
  - Textured generation via the Hunyuan3D paint pipeline
  - Background removal via rembg (BiRefNet) when preprocess_image is requested
  - Background model preload (controlled by HY3D_PRELOAD env, default=1)
"""

from __future__ import annotations

import io
import os
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

from PIL import Image


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def _lazy_import_shape_pipeline():
    """Import shape pipeline on first use (heavy imports)."""
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/hy3dshape")
    sys.path.insert(0, "/app/hy3dpaint")
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    return Hunyuan3DDiTFlowMatchingPipeline


def _lazy_import_paint_pipeline():
    """Import paint/texture pipeline on first use."""
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/hy3dshape")
    sys.path.insert(0, "/app/hy3dpaint")
    from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
    return Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig


# ---------------------------------------------------------------------------
# Global singletons
# ---------------------------------------------------------------------------

_SHAPE_PIPELINE = None
_PAINT_PIPELINE = None
_REMBG_SESSION = None

_READY_LOCK = threading.Lock()
_READY: dict = {
    "status": "not_started",   # not_started | downloading | loading | ready | error
    "detail": "",
    "started_at": None,
    "ready_at": None,
}


def get_ready_state() -> dict:
    """Return a snapshot of the current readiness state."""
    with _READY_LOCK:
        return dict(_READY)


def _set_ready(status: str, detail: str = ""):
    with _READY_LOCK:
        _READY["status"] = status
        _READY["detail"] = detail
        if status == "ready" and not _READY.get("ready_at"):
            _READY["ready_at"] = time.time()


# ---------------------------------------------------------------------------
# Model resolution (download / cache via huggingface_hub)
# ---------------------------------------------------------------------------

def _resolve_model_snapshot(model_id: str) -> str:
    """Resolve a HF model ID to a local snapshot directory, downloading if needed."""
    if os.path.isdir(model_id):
        return model_id

    from huggingface_hub import snapshot_download
    print(f"[worker] resolving model snapshot: {model_id}", flush=True)
    snapshot_dir = snapshot_download(model_id, repo_type="model")
    print(f"[worker] model snapshot ready: {snapshot_dir}", flush=True)
    return snapshot_dir


# ---------------------------------------------------------------------------
# Pipeline getters (lazy + cached)
# ---------------------------------------------------------------------------

def _get_shape_pipeline():
    global _SHAPE_PIPELINE
    if _SHAPE_PIPELINE is not None:
        return _SHAPE_PIPELINE

    model_path = os.environ.get("HY3D_MODEL_PATH", "tencent/Hunyuan3D-2.1")
    subfolder = os.environ.get("HY3D_SUBFOLDER", "hunyuan3d-dit-v2-1")
    device = os.environ.get("HY3D_DEVICE", "cuda")

    _set_ready("downloading", f"resolving model snapshot ({model_path})")
    model_source = _resolve_model_snapshot(model_path)

    _set_ready("loading", "loading shape pipeline")
    Hunyuan3DDiTFlowMatchingPipeline = _lazy_import_shape_pipeline()
    _SHAPE_PIPELINE = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_source,
        subfolder=subfolder,
        use_safetensors=False,
        device=device,
    )
    return _SHAPE_PIPELINE


def _get_paint_pipeline(texture_output_size: int | None = None):
    global _PAINT_PIPELINE

    # If a specific texture_output_size is requested that differs from
    # the cached pipeline's config, rebuild with the new size.
    if _PAINT_PIPELINE is not None:
        cached_tex_size = getattr(_PAINT_PIPELINE, "_hy3d_texture_size", None)
        if texture_output_size is None or texture_output_size == cached_tex_size:
            return _PAINT_PIPELINE
        print(f"[worker] paint pipeline texture_size changed ({cached_tex_size} -> {texture_output_size}), rebuilding", flush=True)
        _PAINT_PIPELINE = None

    _set_ready("loading", "loading paint pipeline")
    Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig = _lazy_import_paint_pipeline()

    # Configure the paint pipeline (matches gradio_app.py usage).
    # Paths are relative to /app (where the HF Space code is cloned).
    max_num_view = int(os.environ.get("HY3D_PAINT_VIEWS", "8"))
    resolution = int(os.environ.get("HY3D_PAINT_RESOLUTION", "768"))
    conf = Hunyuan3DPaintConfig(max_num_view, resolution)
    conf.realesrgan_ckpt_path = "/app/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    conf.multiview_cfg_path = "/app/hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    conf.custom_pipeline = "/app/hy3dpaint/hunyuanpaintpbr"

    # Apply texture_output_size from the request (1024, 2048, or 4096).
    # This maps to conf.texture_size (UV bake resolution).
    # render_size is the rasterization resolution for view baking.
    ALLOWED_TEXTURE_SIZES = {1024, 2048, 4096}
    if texture_output_size and texture_output_size in ALLOWED_TEXTURE_SIZES:
        conf.texture_size = texture_output_size
        conf.render_size = min(2048, texture_output_size)
        print(f"[worker] paint config: texture_size={conf.texture_size}, render_size={conf.render_size}", flush=True)

    _PAINT_PIPELINE = Hunyuan3DPaintPipeline(conf)
    _PAINT_PIPELINE._hy3d_texture_size = conf.texture_size  # tag for cache invalidation
    print(
        f"[worker] paint pipeline loaded (views={max_num_view}, resolution={resolution}, "
        f"texture_size={conf.texture_size}, render_size={conf.render_size})",
        flush=True,
    )
    return _PAINT_PIPELINE


def _get_rembg_session():
    global _REMBG_SESSION
    if _REMBG_SESSION is not None:
        return _REMBG_SESSION
    from rembg import new_session
    _REMBG_SESSION = new_session("birefnet-general")
    return _REMBG_SESSION


# ---------------------------------------------------------------------------
# Background preload
# ---------------------------------------------------------------------------

def _preload_worker():
    """Background thread: download models + load pipelines so first request is fast."""
    try:
        print("[worker] preload: starting model download + pipeline load", flush=True)
        with _READY_LOCK:
            _READY["started_at"] = time.time()

        _get_shape_pipeline()
        print("[worker] preload: shape pipeline loaded", flush=True)

        # Only preload paint pipeline if textures are likely to be used.
        if os.environ.get("HY3D_PRELOAD_PAINT", "1").strip().lower() not in ("0", "false", "no", "off"):
            _get_paint_pipeline()
            print("[worker] preload: paint pipeline loaded", flush=True)

        _set_ready("ready", "all pipelines loaded")
        st = get_ready_state()
        dt = (st.get("ready_at") or time.time()) - (st.get("started_at") or time.time())
        print(f"[worker] preload: ready (load_time_sec={dt:.1f})", flush=True)

    except Exception:
        tb = traceback.format_exc()
        _set_ready("error", tb[-4000:] if tb else "unknown error")
        print(f"[worker] preload: ERROR\n{tb}", flush=True)


def start_preload_in_background():
    """Kick off model preload in a daemon thread.  Controlled by HY3D_PRELOAD (default=1)."""
    v = os.environ.get("HY3D_PRELOAD", "1").strip().lower()
    if v in ("0", "false", "no", "off"):
        return
    with _READY_LOCK:
        if _READY["status"] != "not_started":
            return
        _READY["status"] = "downloading"
        _READY["detail"] = "preload scheduled"
        _READY["started_at"] = time.time()
    t = threading.Thread(target=_preload_worker, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def remove_background(img: Image.Image) -> Image.Image:
    """Remove background using rembg (BiRefNet model)."""
    from rembg import remove
    session = _get_rembg_session()
    print("[worker] rembg: running background removal", flush=True)
    t0 = time.time()
    result = remove(img, session=session)
    dt = time.time() - t0
    print(f"[worker] rembg: done in {dt:.1f}s", flush=True)
    return result.convert("RGBA")


def _is_truthy(value: str | None) -> bool:
    """Check if a form field value is truthy."""
    if not value:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _decimate_trimesh(mesh, target_faces: int):
    """Decimate a trimesh mesh to the target face count using quadric decimation."""
    face_count = len(mesh.faces) if hasattr(mesh, "faces") else 0
    if face_count <= 0 or face_count <= target_faces:
        print(f"[worker] decimation skipped (faces={face_count}, target={target_faces})", flush=True)
        return mesh
    print(f"[worker] decimating mesh: {face_count} -> {target_faces} faces", flush=True)
    t0 = time.time()
    try:
        mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
        dt = time.time() - t0
        final = len(mesh.faces) if hasattr(mesh, "faces") else 0
        print(f"[worker] decimation done in {dt:.1f}s (result={final} faces)", flush=True)
    except Exception as e:
        print(f"[worker] decimation failed, using original mesh: {e}", flush=True)
    return mesh


def _decimate_glb(glb_path: Path, target_faces: int):
    """Reload a GLB via trimesh, decimate, and overwrite the file."""
    import trimesh
    loaded = trimesh.load(str(glb_path), force="mesh")
    decimated = _decimate_trimesh(loaded, target_faces)
    decimated.export(str(glb_path))


# ---------------------------------------------------------------------------
# Main generation entry point
# ---------------------------------------------------------------------------

def generate_glb_from_image_bytes(
    image_bytes: bytes,
    out_dir: Path,
    want_textures: bool = False,
    preprocess_image: bool = False,
    seed: int | None = None,
    decimation_target: int | None = None,
    texture_output_size: int | None = None,
) -> dict:
    """
    Generate a GLB from input image bytes.

    Returns dict with:
      - glb_path: Path to the generated GLB
      - preprocessed_image_path: Path to the rembg-processed image (if applicable), or None
      - texture_status: "success", "skipped", or error message
    """
    if not image_bytes:
        raise ValueError("empty image upload")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    # Background removal
    preprocessed_image_path = None
    if preprocess_image:
        img = remove_background(img)
        # Save the preprocessed image so the main app can download it
        rembg_name = f"rembg_{uuid.uuid4().hex}.png"
        preprocessed_image_path = out_dir / rembg_name
        img.save(str(preprocessed_image_path), format="PNG")
        print(f"[worker] rembg: saved preprocessed image to {preprocessed_image_path}", flush=True)

    # Save the input image to disk (paint pipeline needs a file path)
    input_image_path = out_dir / f"_input_{uuid.uuid4().hex}.png"
    img.save(str(input_image_path), format="PNG")

    # Shape generation
    shape_pipeline = _get_shape_pipeline()
    print("[worker] generating shape...", flush=True)
    t_shape = time.time()
    mesh = shape_pipeline(image=img, seed=seed)[0] if seed is not None else shape_pipeline(image=img)[0]
    dt_shape = time.time() - t_shape
    print(f"[worker] shape generation done in {dt_shape:.1f}s", flush=True)

    # Export shape mesh to GLB (needed as input for both paint pipeline and decimation)
    shape_glb = out_dir / f"_shape_{uuid.uuid4().hex}.glb"
    mesh.export(str(shape_glb))
    print(f"[worker] shape mesh exported to {shape_glb} ({shape_glb.stat().st_size} bytes)", flush=True)

    # Decimation on the shape mesh (before texturing)
    if decimation_target is not None and decimation_target > 0:
        try:
            _decimate_glb(shape_glb, decimation_target)
            print(f"[worker] decimated shape mesh ({shape_glb.stat().st_size} bytes)", flush=True)
        except Exception as e:
            print(f"[worker] decimation failed, using original shape: {e}", flush=True)
            traceback.print_exc()

    # Texture generation — the paint pipeline takes FILE PATHS, not objects.
    texture_status = "skipped"
    final_glb = out_dir / f"{uuid.uuid4().hex}.glb"

    if want_textures:
        try:
            paint_pipeline = _get_paint_pipeline(texture_output_size=texture_output_size)
            print("[worker] generating textures...", flush=True)
            t_tex = time.time()

            # The paint pipeline writes an OBJ first, then converts to GLB
            # via convert_obj_to_glb().  The GLB path is the OBJ path with
            # the extension swapped to .glb.  So we must pass an .obj path.
            textured_obj = out_dir / f"_textured_{uuid.uuid4().hex}.obj"
            textured_glb = Path(str(textured_obj).replace(".obj", ".glb"))

            result_path = paint_pipeline(
                mesh_path=str(shape_glb),
                image_path=str(input_image_path),
                output_mesh_path=str(textured_obj),
                save_glb=True,
            )
            dt_tex = time.time() - t_tex
            print(f"[worker] texture generation done in {dt_tex:.1f}s", flush=True)
            print(f"[worker] paint pipeline returned: {result_path}", flush=True)

            # The pipeline returns the OBJ path; the GLB is at the .glb sibling.
            if textured_glb.is_file():
                textured_glb.rename(final_glb)
                texture_status = "success"
                print(f"[worker] textured GLB: {final_glb} ({final_glb.stat().st_size} bytes)", flush=True)
            elif Path(str(result_path)).is_file():
                # Fallback: if only the OBJ exists, use it (caller gets OBJ not GLB)
                Path(str(result_path)).rename(final_glb)
                texture_status = "success (obj only)"
                print(f"[worker] textured OBJ (no GLB conversion): {final_glb}", flush=True)
            else:
                print(f"[worker] paint pipeline produced no output, falling back to shape-only", flush=True)
                shape_glb.rename(final_glb)
                texture_status = "failed: no output file produced"

            # Clean up the OBJ and any sidecar files (.mtl, textures)
            for ext in (".obj", ".mtl"):
                p = Path(str(textured_obj).replace(".obj", ext))
                try:
                    if p.is_file():
                        p.unlink()
                except OSError:
                    pass
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[worker] texture generation FAILED: {e}\n{tb}", flush=True)
            texture_status = f"failed: {e}"
            # Fall back to shape-only
            if shape_glb.is_file():
                shape_glb.rename(final_glb)
    else:
        # No textures requested — just use the (possibly decimated) shape mesh
        shape_glb.rename(final_glb)

    # Clean up temp files
    for tmp in (shape_glb, input_image_path):
        try:
            if tmp.is_file():
                tmp.unlink()
        except OSError:
            pass

    file_size = final_glb.stat().st_size if final_glb.is_file() else 0
    dt_total = time.time() - t0
    print(
        f"[worker] generated {final_glb} in {dt_total:.1f}s "
        f"(bytes_in={len(image_bytes)}, bytes_out={file_size}, "
        f"textures={texture_status}, "
        f"preprocess={'yes' if preprocess_image else 'no'}, "
        f"decimation_target={decimation_target or 'none'})",
        flush=True,
    )
    return {
        "glb_path": final_glb,
        "preprocessed_image_path": preprocessed_image_path,
        "texture_status": texture_status,
    }
