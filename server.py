#!/usr/bin/env python3
"""
FastAPI server for Hunyuan3D-2.1 worker.

Endpoints:
  - GET  /health  -> {"status":"OK"}
  - GET  /ready   -> {"ready": bool, "status": ..., "detail": ...}
  - POST /generate (multipart) -> NDJSON stream with keepalive pings + final result
  - GET  /download/{filename}  -> file bytes
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from worker import (
    generate_glb_from_image_bytes,
    get_ready_state,
    start_preload_in_background,
    _is_truthy,
)


APP_PORT = int(os.environ.get("PORT", "8000"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KEEPALIVE_INTERVAL = int(os.environ.get("KEEPALIVE_INTERVAL", "30"))

_STARTED_AT = time.time()
_LAST_REQUEST_AT = _STARTED_AT
_REQUEST_COUNT = 0
_EXECUTOR = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start server first so container is reachable; model download + load
    # runs in a background thread.
    start_preload_in_background()
    print(
        f"[server] listening on 0.0.0.0:{APP_PORT}; "
        f"preload (model download + pipeline load) running in background",
        flush=True,
    )
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return JSONResponse({"status": "OK"}, status_code=200)


@app.get("/ready")
def ready():
    st = get_ready_state()
    is_ready = st["status"] == "ready"
    return JSONResponse({
        "ready": is_ready,
        "status": st["status"],
        "detail": st.get("detail", ""),
    }, status_code=200)


@app.get("/idle")
def idle():
    idle_sec = time.time() - _LAST_REQUEST_AT
    return JSONResponse({
        "minutes_since_last_request": idle_sec / 60.0,
        "consecutive_failures": 0,
        "total_requests": _REQUEST_COUNT,
    })


@app.post("/generate")
async def generate(request: Request):
    global _LAST_REQUEST_AT, _REQUEST_COUNT
    _LAST_REQUEST_AT = time.time()
    _REQUEST_COUNT += 1

    try:
        form = await request.form()

        # Read image(s) — support both "image" (single) and "images" (multi)
        image_file = form.get("image")
        if image_file is None:
            return JSONResponse({"success": False, "error": "No image uploaded"}, status_code=400)
        raw = await image_file.read()

        # Parse flags — textures default to ON; caller can disable with
        # no_texture=1 or by explicitly passing texture=0 / false / off.
        _explicit_off = any(
            str(form.get(k, "")).strip().lower() in ("0", "false", "no", "off")
            for k in ("texture", "textures", "enable_texture", "embed_textures")
        )
        _explicit_no = _is_truthy(form.get("no_texture")) or _is_truthy(form.get("no_textures"))
        want_textures = not (_explicit_off or _explicit_no)
        preprocess_image = _is_truthy(form.get("preprocess_image"))

        # Parse optional seed
        seed = None
        seed_raw = form.get("seed")
        if seed_raw is not None:
            try:
                seed = int(str(seed_raw).strip())
            except (ValueError, TypeError):
                pass

        # Parse optional decimation target
        decimation_target = None
        dec_raw = form.get("decimation_target")
        if dec_raw is not None:
            try:
                decimation_target = int(str(dec_raw).strip())
            except (ValueError, TypeError):
                pass

        # Parse optional texture output size (1024, 2048, 4096)
        texture_output_size = None
        tex_size_raw = form.get("texture_output_size")
        if tex_size_raw is not None:
            try:
                texture_output_size = int(str(tex_size_raw).strip())
            except (ValueError, TypeError):
                pass

        # Log request info (matches API contract)
        expected_images = form.get("expected_images", "1")
        print(
            f"[worker] generate expected_images={expected_images} "
            f"textures={want_textures} preprocess={preprocess_image} "
            f"seed={seed} decimation_target={decimation_target} "
            f"texture_output_size={texture_output_size} bytes={len(raw)}",
            flush=True,
        )

        # Run generation in a thread so we can stream keepalive pings
        loop = asyncio.get_event_loop()
        gen_kwargs = dict(
            image_bytes=raw,
            out_dir=OUTPUT_DIR,
            want_textures=want_textures,
            preprocess_image=preprocess_image,
            seed=seed,
            decimation_target=decimation_target,
            texture_output_size=texture_output_size,
        )
        gen_future = loop.run_in_executor(_EXECUTOR, lambda: generate_glb_from_image_bytes(**gen_kwargs))

        async def ndjson_stream():
            t0 = time.time()
            try:
                while True:
                    try:
                        result = await asyncio.wait_for(asyncio.shield(gen_future), timeout=KEEPALIVE_INTERVAL)
                    except asyncio.TimeoutError:
                        # Generation still running — send keepalive
                        elapsed = int(time.time() - t0)
                        ping = json.dumps({"type": "keepalive", "elapsed_sec": elapsed})
                        yield ping + "\n"
                        continue

                    # Generation finished
                    elapsed = int(time.time() - t0)
                    glb_path = result["glb_path"]
                    preprocessed_path = result.get("preprocessed_image_path")
                    texture_status = result.get("texture_status", "unknown")
                    glb_size = glb_path.stat().st_size if Path(str(glb_path)).is_file() else 0

                    print(
                        f"[server] generate done: texture_status={texture_status} "
                        f"glb_size={glb_size} glb_path={glb_path} elapsed={elapsed}s",
                        flush=True,
                    )

                    response = {
                        "type": "result",
                        "success": True,
                        "glb_path": str(glb_path),
                        "texture_status": texture_status,
                    }

                    worker_export = {"texture_status": texture_status}
                    if preprocessed_path and preprocessed_path.is_file():
                        worker_export["preprocessed_images"] = [preprocessed_path.name]
                    response["worker_export"] = worker_export

                    yield json.dumps(response) + "\n"
                    return

            except Exception as e:
                tb = traceback.format_exc()
                if len(tb) > 20000:
                    tb = tb[:20000] + "\n...[truncated]...\n"
                error_resp = {"type": "result", "success": False, "error": tb or str(e)}
                yield json.dumps(error_resp) + "\n"

        return StreamingResponse(ndjson_stream(), media_type="application/x-ndjson")

    except Exception as e:
        tb = traceback.format_exc()
        if len(tb) > 20000:
            tb = tb[:20000] + "\n...[truncated]...\n"
        return JSONResponse({"success": False, "error": tb or str(e)}, status_code=200)


@app.get("/download/{filename}")
def download(filename: str):
    name = os.path.basename(filename or "")
    p = OUTPUT_DIR / name
    if not p.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)

    # Determine media type from extension
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    media_types = {
        "glb": "model/gltf-binary",
        "gltf": "model/gltf+json",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
    }
    media_type = media_types.get(ext, "application/octet-stream")
    return FileResponse(str(p), media_type=media_type, filename=name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=APP_PORT, log_level="info")
