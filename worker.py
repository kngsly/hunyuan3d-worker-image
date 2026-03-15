#!/usr/bin/env python3
"""
Hunyuan3D-2.1 worker with texture and background removal support.

Supports:
  - Shape-only generation (default, or when texture flags are absent)
  - Textured generation via the Hunyuan3D paint pipeline
  - Background removal via rembg (BiRefNet) when preprocess_image is requested
"""

from __future__ import annotations

import io
import os
import sys
import time
import traceback
import uuid
from pathlib import Path

from PIL import Image


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
    from hy3dpaint import Hunyuan3DPaintPipeline
    return Hunyuan3DPaintPipeline


_SHAPE_PIPELINE = None
_PAINT_PIPELINE = None
_REMBG_SESSION = None


def _get_shape_pipeline():
    global _SHAPE_PIPELINE
    if _SHAPE_PIPELINE is not None:
        return _SHAPE_PIPELINE

    model_path = os.environ.get("HY3D_MODEL_PATH", "tencent/Hunyuan3D-2.1")
    subfolder = os.environ.get("HY3D_SUBFOLDER", "hunyuan3d-dit-v2-1")
    device = os.environ.get("HY3D_DEVICE", "cuda")

    Hunyuan3DDiTFlowMatchingPipeline = _lazy_import_shape_pipeline()
    _SHAPE_PIPELINE = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_path,
        subfolder=subfolder,
        use_safetensors=False,
        device=device,
    )
    return _SHAPE_PIPELINE


def _get_paint_pipeline():
    global _PAINT_PIPELINE
    if _PAINT_PIPELINE is not None:
        return _PAINT_PIPELINE

    model_path = os.environ.get("HY3D_MODEL_PATH", "tencent/Hunyuan3D-2.1")
    device = os.environ.get("HY3D_DEVICE", "cuda")

    Hunyuan3DPaintPipeline = _lazy_import_paint_pipeline()
    _PAINT_PIPELINE = Hunyuan3DPaintPipeline.from_pretrained(
        model_path,
        subfolder="hunyuan3d-paint-v2-1",
        use_safetensors=True,
        device=device,
    )
    return _PAINT_PIPELINE


def _get_rembg_session():
    global _REMBG_SESSION
    if _REMBG_SESSION is not None:
        return _REMBG_SESSION
    from rembg import new_session
    _REMBG_SESSION = new_session("birefnet-general")
    return _REMBG_SESSION


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


def generate_glb_from_image_bytes(
    image_bytes: bytes,
    out_dir: Path,
    want_textures: bool = False,
    preprocess_image: bool = False,
    seed: int | None = None,
    decimation_target: int | None = None,
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

    # Shape generation
    shape_pipeline = _get_shape_pipeline()
    print("[worker] generating shape...", flush=True)
    t_shape = time.time()
    mesh = shape_pipeline(image=img, seed=seed)[0] if seed is not None else shape_pipeline(image=img)[0]
    dt_shape = time.time() - t_shape
    print(f"[worker] shape generation done in {dt_shape:.1f}s", flush=True)

    # -- Decimation BEFORE texturing --
    # We decimate the shape mesh first, then paint textures onto the
    # decimated mesh. This avoids the problem of trimesh stripping
    # UV/material data when reloading a textured GLB.
    did_decimate = False
    if decimation_target is not None and decimation_target > 0:
        try:
            # Export shape-only mesh to a temp GLB, reload via trimesh,
            # decimate, and save back.
            tmp_shape = out_dir / f"_shape_{uuid.uuid4().hex}.glb"
            mesh.export(str(tmp_shape))
            _decimate_glb(tmp_shape, decimation_target)
            did_decimate = True

            # Reload the decimated mesh back into the format the paint
            # pipeline expects.  Try the Hunyuan3D Mesh loader first;
            # fall back to passing the trimesh object directly.
            try:
                sys.path.insert(0, "/app")
                sys.path.insert(0, "/app/hy3dshape")
                from hy3dgen.shapegen.utils import Mesh as HY3DMesh
                mesh = HY3DMesh.load(str(tmp_shape))
                print(f"[worker] reloaded decimated mesh via HY3DMesh.load", flush=True)
            except Exception as e1:
                print(f"[worker] HY3DMesh.load failed ({e1}), trying trimesh reload", flush=True)
                try:
                    import trimesh
                    mesh = trimesh.load(str(tmp_shape))
                    print(f"[worker] reloaded decimated mesh via trimesh", flush=True)
                except Exception as e2:
                    print(f"[worker] trimesh reload also failed ({e2}), using original mesh", flush=True)
                    did_decimate = False

            # Clean up temp file
            try:
                tmp_shape.unlink()
            except OSError:
                pass
        except Exception as e:
            print(f"[worker] pre-texture decimation failed, using original: {e}", flush=True)
            traceback.print_exc()

    # Texture generation (if requested)
    texture_status = "skipped"
    if want_textures:
        try:
            paint_pipeline = _get_paint_pipeline()
            print("[worker] generating textures...", flush=True)
            t_tex = time.time()
            mesh = paint_pipeline(mesh, image=img)
            dt_tex = time.time() - t_tex
            print(f"[worker] texture generation done in {dt_tex:.1f}s", flush=True)
            texture_status = "success"
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[worker] texture generation FAILED: {e}\n{tb}", flush=True)
            texture_status = f"failed: {e}"

    # Export final mesh
    fname = f"{uuid.uuid4().hex}.glb"
    out_path = out_dir / fname
    mesh.export(str(out_path))
    file_size = out_path.stat().st_size
    print(f"[worker] exported {out_path} ({file_size} bytes)", flush=True)

    # If we didn't decimate before texturing (no target or failed),
    # and textures are OFF, we can still decimate the exported GLB.
    if not did_decimate and decimation_target is not None and decimation_target > 0 and not want_textures:
        try:
            _decimate_glb(out_path, decimation_target)
        except Exception as e:
            print(f"[worker] post-export decimation failed, keeping original: {e}", flush=True)

    dt_total = time.time() - t0
    print(
        f"[worker] generated {out_path} in {dt_total:.1f}s "
        f"(bytes_in={len(image_bytes)}, textures={texture_status}, "
        f"preprocess={'yes' if preprocess_image else 'no'}, "
        f"decimation_target={decimation_target or 'none'}, "
        f"decimated_before_paint={did_decimate})",
        flush=True,
    )
    return {
        "glb_path": out_path,
        "preprocessed_image_path": preprocessed_image_path,
        "texture_status": texture_status,
    }
