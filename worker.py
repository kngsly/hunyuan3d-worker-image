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
    from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline
    return Hunyuan3DDiTFlowMatchingPipeline


def _lazy_import_paint_pipeline():
    """Import paint/texture pipeline on first use."""
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/hy3dshape")
    sys.path.insert(0, "/app/hy3dpaint")
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
    return Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig


# ---------------------------------------------------------------------------
# Global singletons
# ---------------------------------------------------------------------------

_SHAPE_PIPELINE = None
_PAINT_PIPELINE = None
_PAINT_LAST_ERROR = None  # last paint pipeline error for diagnostics
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


# Fallback chains per requested texture_output_size.
# On CUDA OOM, steps down to the next entry.  Keeps 8 views as long as
# possible (better back/side coverage), then drops views as last resort.
PAINT_TIER_CHAINS = {
    4096: [
        {"views": 8, "resolution": 768, "render_size": 2048, "texture_size": 4096, "label": "4096-ultra"},
        {"views": 8, "resolution": 512, "render_size": 1024, "texture_size": 4096, "label": "4096-high"},
        {"views": 8, "resolution": 512, "render_size": 1024, "texture_size": 2048, "label": "4096-med"},
        {"views": 6, "resolution": 512, "render_size": 1024, "texture_size": 2048, "label": "4096-low"},
        {"views": 6, "resolution": 512, "render_size": 1024, "texture_size": 1024, "label": "4096-min"},
    ],
    2048: [
        {"views": 8, "resolution": 768, "render_size": 2048, "texture_size": 2048, "label": "2048-ultra"},
        {"views": 8, "resolution": 512, "render_size": 1024, "texture_size": 2048, "label": "2048-high"},
        {"views": 8, "resolution": 512, "render_size": 1024, "texture_size": 1024, "label": "2048-med"},
        {"views": 6, "resolution": 512, "render_size": 1024, "texture_size": 1024, "label": "2048-low"},
    ],
    1024: [
        {"views": 8, "resolution": 512, "render_size": 1024, "texture_size": 1024, "label": "1024-high"},
        {"views": 8, "resolution": 512, "render_size": 512,  "texture_size": 1024, "label": "1024-med"},
        {"views": 6, "resolution": 512, "render_size": 512,  "texture_size": 1024, "label": "1024-low"},
    ],
}
# Default chain if no texture_output_size or unrecognized value
PAINT_TIER_CHAINS["default"] = PAINT_TIER_CHAINS[1024]


def _build_paint_pipeline(tier: dict, texture_output_size: int | None = None):
    """Build a paint pipeline for the given quality tier."""
    global _PAINT_PIPELINE

    Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig = _lazy_import_paint_pipeline()

    views = int(os.environ.get("HY3D_PAINT_VIEWS", str(tier["views"])))
    resolution = int(os.environ.get("HY3D_PAINT_RESOLUTION", str(tier["resolution"])))
    conf = Hunyuan3DPaintConfig(views, resolution)
    conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"

    # Tier defines all sizes — the chain is already tailored to the requested output size
    conf.render_size = tier["render_size"]
    conf.texture_size = tier["texture_size"]

    _PAINT_PIPELINE = Hunyuan3DPaintPipeline(conf)
    _PAINT_PIPELINE._hy3d_quality_tier = tier["label"]
    _PAINT_PIPELINE._hy3d_texture_size = conf.texture_size
    print(
        f"[worker] paint pipeline built: tier={tier['label']} views={views} resolution={resolution} "
        f"texture_size={conf.texture_size} render_size={conf.render_size}",
        flush=True,
    )
    return _PAINT_PIPELINE


def _get_paint_pipeline(texture_output_size: int | None = None):
    """Return the cached paint pipeline, building it at the highest quality tier if needed."""
    global _PAINT_PIPELINE

    if _PAINT_PIPELINE is not None:
        cached_tex_size = getattr(_PAINT_PIPELINE, "_hy3d_texture_size", None)
        if texture_output_size is None or texture_output_size == cached_tex_size:
            return _PAINT_PIPELINE
        print(f"[worker] paint pipeline texture_size changed ({cached_tex_size} -> {texture_output_size}), rebuilding", flush=True)
        _PAINT_PIPELINE = None

    _set_ready("loading", "loading paint pipeline")
    chain = PAINT_TIER_CHAINS.get(texture_output_size, PAINT_TIER_CHAINS["default"])
    return _build_paint_pipeline(chain[0], texture_output_size)


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

        # Paint pipeline preload is best-effort — if it fails, we can
        # still serve shape-only requests and retry paint on first use.
        if os.environ.get("HY3D_PRELOAD_PAINT", "1").strip().lower() not in ("0", "false", "no", "off"):
            try:
                _get_paint_pipeline()
                print("[worker] preload: paint pipeline loaded", flush=True)
            except Exception as exc:
                global _PAINT_LAST_ERROR
                tb = traceback.format_exc()
                # Store just the exception line for the ready detail (it gets truncated)
                _PAINT_LAST_ERROR = f"{type(exc).__name__}: {exc}"
                print(f"[worker] preload: paint pipeline failed (will retry on first use):\n{tb}", flush=True)

        paint_ok = _PAINT_PIPELINE is not None
        _set_ready("ready", f"shape=ok paint={'ok' if paint_ok else 'FAILED: ' + (_PAINT_LAST_ERROR or 'unknown')[:500]}")
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
    """Reload a GLB via trimesh, decimate, and overwrite the file (untextured only)."""
    import trimesh
    loaded = trimesh.load(str(glb_path), force="mesh")
    decimated = _decimate_trimesh(loaded, target_faces)
    decimated.export(str(glb_path))


def _decimate_textured_glb(glb_path: Path, target_faces: int):
    """Decimate a textured GLB using pymeshlab, preserving UVs and texture maps."""
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(glb_path))
    face_count = ms.current_mesh().face_number()
    if face_count <= target_faces:
        print(f"[worker] textured decimation skipped (faces={face_count}, target={target_faces})", flush=True)
        return
    print(f"[worker] textured decimation: {face_count} -> {target_faces} faces", flush=True)
    t0 = time.time()
    # Quadric edge collapse with texture preservation.
    # qualitythr=1.0 allows aggressive reduction to reach the target face count.
    # Previous value of 0.3 would stop early, producing ~40K faces regardless of target.
    ms.meshing_decimation_quadric_edge_collapse_with_texture(
        targetfacenum=target_faces,
        qualitythr=1.0,
        preserveboundary=True,
        preservenormal=True,
        preservetopology=False,
        planarquadric=True,
    )
    final = ms.current_mesh().face_number()
    dt = time.time() - t0
    print(f"[worker] textured decimation done in {dt:.1f}s (result={final} faces)", flush=True)
    ms.save_current_mesh(str(glb_path))


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
    extra_image_bytes_list: list[bytes] | None = None,
    octree_resolution: int | None = None,
    num_inference_steps: int | None = None,
    target_face_count: int | None = None,
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

    # Load extra reference images for multi-view texturing
    all_texture_images = [img]
    if extra_image_bytes_list:
        for idx, eb in enumerate(extra_image_bytes_list):
            try:
                extra_img = Image.open(io.BytesIO(eb)).convert("RGBA")
                if preprocess_image:
                    extra_img = remove_background(extra_img)
                all_texture_images.append(extra_img)
                print(f"[worker] loaded extra image {idx+1} for texture reference", flush=True)
            except Exception as e:
                print(f"[worker] failed to load extra image {idx+1}: {e}", flush=True)
    print(f"[worker] texture reference images: {len(all_texture_images)}", flush=True)

    # Shape generation — ensure pipeline is on GPU
    shape_pipeline = _get_shape_pipeline()
    import torch as _torch
    if hasattr(shape_pipeline, 'device') and str(getattr(shape_pipeline, 'device', '')) != 'cuda':
        try:
            shape_pipeline.to("cuda")
            print("[worker] moved shape pipeline back to GPU", flush=True)
        except Exception:
            pass
    # Build shape pipeline kwargs.
    # octree_resolution: marching cubes grid density. Default 384 -> ~40K faces.
    #   Higher values produce denser meshes (512 -> 100K+). Scales roughly cubically.
    # num_inference_steps: diffusion quality. Default 50. More = better SDF quality.
    shape_kwargs = {"image": img}
    if seed is not None:
        shape_kwargs["seed"] = seed
    if octree_resolution is not None:
        shape_kwargs["octree_resolution"] = octree_resolution
    if num_inference_steps is not None:
        shape_kwargs["num_inference_steps"] = num_inference_steps
    print(f"[worker] generating shape (octree_resolution={octree_resolution or 'default'}, steps={num_inference_steps or 'default'}, target_faces={target_face_count})...", flush=True)
    t_shape = time.time()
    mesh = shape_pipeline(**shape_kwargs)[0]
    dt_shape = time.time() - t_shape
    raw_face_count = len(mesh.faces) if hasattr(mesh, "faces") else 0
    print(f"[worker] shape generation done in {dt_shape:.1f}s ({raw_face_count} faces from pipeline)", flush=True)

    # Apply FaceReducer if target_face_count is set and the mesh exceeds it.
    # The upstream pipeline does NOT reduce faces internally — it outputs the raw
    # marching cubes mesh. We use hy3dshape's FaceReducer for controlled reduction.
    if target_face_count is not None and target_face_count > 0 and raw_face_count > target_face_count:
        try:
            try:
                from hy3dshape import FaceReducer
            except ImportError:
                from hy3dshape.postprocessors import FaceReducer
            t_reduce = time.time()
            face_reducer = FaceReducer()
            mesh = face_reducer(mesh, max_facenum=target_face_count)
            dt_reduce = time.time() - t_reduce
            reduced_count = len(mesh.faces) if hasattr(mesh, "faces") else "?"
            print(f"[worker] FaceReducer: {raw_face_count} -> {reduced_count} faces in {dt_reduce:.1f}s (target={target_face_count})", flush=True)
        except Exception as e:
            print(f"[worker] FaceReducer failed, using raw mesh: {e}", flush=True)

    # Export shape mesh — OBJ for paint pipeline (pymeshlab needs OBJ), GLB as fallback output.
    shape_obj = out_dir / f"_shape_{uuid.uuid4().hex}.obj"
    shape_glb = out_dir / f"_shape_{uuid.uuid4().hex}.glb"
    mesh.export(str(shape_obj))
    mesh.export(str(shape_glb))
    print(f"[worker] shape mesh exported: obj={shape_obj.stat().st_size}B glb={shape_glb.stat().st_size}B", flush=True)

    # Decimation happens AFTER texturing (on the final GLB) — see below.
    # The paint pipeline's remesh_mesh() re-meshes internally, so decimating
    # before texturing has no effect.

    # Free shape pipeline VRAM before texturing — RTX 4090 (24GB) can't hold both.
    # On high-VRAM GPUs (48GB+), skip offloading to avoid slow CPU-GPU transfers.
    if want_textures:
        import torch
        global _SHAPE_PIPELINE
        if _SHAPE_PIPELINE is not None:
            total_vram_gb = torch.cuda.mem_get_info()[1] / (1024**3)
            if total_vram_gb < 40:
                _SHAPE_PIPELINE.to("cpu")
                torch.cuda.empty_cache()
                print(f"[worker] offloaded shape pipeline to CPU, freed VRAM", flush=True)
            else:
                free_vram_gb = torch.cuda.mem_get_info()[0] / (1024**3)
                print(f"[worker] keeping shape pipeline on GPU ({total_vram_gb:.0f}GB total, {free_vram_gb:.0f}GB free)", flush=True)

    # Texture generation — the paint pipeline takes FILE PATHS, not objects.
    texture_status = "skipped"
    final_glb = out_dir / f"{uuid.uuid4().hex}.glb"

    if want_textures:
        import torch as _torch

        def _is_oom(exc):
            return "CUDA out of memory" in str(exc) or "OutOfMemoryError" in type(exc).__name__

        def _try_texture(tier, texture_output_size):
            """Attempt texture generation at a given quality tier. Returns (texture_status, success)."""
            global _PAINT_PIPELINE
            # Rebuild pipeline at this tier
            _PAINT_PIPELINE = None
            _torch.cuda.empty_cache()
            paint_pipeline = _build_paint_pipeline(tier, texture_output_size)

            textured_obj = out_dir / f"_textured_{uuid.uuid4().hex}.obj"
            textured_glb = Path(str(textured_obj).replace(".obj", ".glb"))

            # The paint pipeline only supports a single image (its list handling is buggy).
            # Pass the primary image as a file path.
            print(f"[worker] texture: trying tier={tier['label']} mesh={shape_obj}", flush=True)
            t_tex = time.time()
            result_path = paint_pipeline(
                mesh_path=str(shape_obj),
                image_path=str(input_image_path),
                output_mesh_path=str(textured_obj),
                save_glb=True,
            )
            dt_tex = time.time() - t_tex
            print(f"[worker] texture: tier={tier['label']} completed in {dt_tex:.1f}s", flush=True)

            # Check output files
            if textured_glb.is_file():
                glb_size = textured_glb.stat().st_size
                textured_glb.rename(final_glb)
                print(f"[worker] textured GLB: {final_glb} ({glb_size} bytes)", flush=True)
                status = f"success (tier={tier['label']})"
            elif Path(str(result_path)).is_file():
                Path(str(result_path)).rename(final_glb)
                status = f"success (obj only, tier={tier['label']})"
                print(f"[worker] textured OBJ: {final_glb}", flush=True)
            else:
                print(f"[worker] tier={tier['label']} produced no output", flush=True)
                return "failed: no output file produced", False

            # Clean up sidecar files
            for ext in (".obj", ".mtl", ".jpg"):
                p = Path(str(textured_obj).replace(".obj", ext))
                try:
                    if p.is_file():
                        p.unlink()
                except OSError:
                    pass
            return status, True

        # Pick the fallback chain for the requested texture size
        chain = PAINT_TIER_CHAINS.get(texture_output_size, PAINT_TIER_CHAINS["default"])
        print(f"[worker] texture: chain={[t['label'] for t in chain]} texture_output_size={texture_output_size}", flush=True)

        # Try quality tiers from highest to lowest, stepping down on OOM
        texture_status = "failed: all tiers exhausted"
        for tier_idx, tier in enumerate(chain):
            try:
                texture_status, success = _try_texture(tier, texture_output_size)
                if success:
                    break
            except Exception as e:
                _torch.cuda.empty_cache()
                if _is_oom(e) and tier_idx < len(chain) - 1:
                    next_tier = chain[tier_idx + 1]
                    print(f"[worker] texture OOM at tier={tier['label']}, stepping down to tier={next_tier['label']}", flush=True)
                    continue
                else:
                    tb = traceback.format_exc()
                    print(f"[worker] texture FAILED at tier={tier['label']}: {e}\n{tb}", flush=True)
                    texture_status = f"failed: {e}"
                    break

        # If all tiers failed, fall back to shape-only
        if not texture_status.startswith("success"):
            if shape_glb.is_file():
                shape_glb.rename(final_glb)
    else:
        # No textures requested — just use the (possibly decimated) shape mesh
        shape_glb.rename(final_glb)

    # Decimation on the final mesh.
    if decimation_target is not None and decimation_target > 0 and final_glb.is_file():
        try:
            if texture_status.startswith("success"):
                # Textured mesh: use pymeshlab which preserves UVs and textures
                _decimate_textured_glb(final_glb, decimation_target)
            else:
                # Untextured: trimesh is fine
                _decimate_glb(final_glb, decimation_target)
            print(f"[worker] decimated final mesh to target={decimation_target} ({final_glb.stat().st_size} bytes)", flush=True)
        except Exception as e:
            print(f"[worker] decimation failed, using undecimated mesh: {e}", flush=True)
            traceback.print_exc()

    # Clean up temp files
    for tmp in (shape_glb, shape_obj, input_image_path):
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
