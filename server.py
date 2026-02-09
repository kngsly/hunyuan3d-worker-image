#!/usr/bin/env python3
"""
FastAPI server for Vast smoke + simple worker orchestration.

Endpoints:
  - GET  /health -> {"status":"OK"}
  - POST /generate (multipart 'image') -> {"success": true, "glb_path": "/outputs/<file>.glb"} or {"success": false, "error": "..."}
  - GET  /download/{filename} -> returns file bytes
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from worker import generate_glb_from_image_bytes


APP_PORT = int(os.environ.get("PORT", "8000"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()


@app.get("/health")
def health():
    return JSONResponse({"status": "OK"}, status_code=200)


@app.post("/generate")
async def generate(image: UploadFile = File(...)):
    try:
        raw = await image.read()
        out_path = generate_glb_from_image_bytes(raw, out_dir=OUTPUT_DIR)
        # Keep the response shape that rent.py expects.
        return JSONResponse({"success": True, "glb_path": str(out_path)}, status_code=200)
    except Exception as e:
        tb = traceback.format_exc()
        # Keep the payload small-ish but useful.
        if len(tb) > 20000:
            tb = tb[:20000] + "\n...[truncated]...\n"
        return JSONResponse({"success": False, "error": tb or str(e)}, status_code=200)


@app.get("/download/{filename}")
def download(filename: str):
    # Tighten path handling: only serve direct children under OUTPUT_DIR.
    name = os.path.basename(filename or "")
    p = OUTPUT_DIR / name
    if not p.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(p), media_type="model/gltf-binary", filename=name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=APP_PORT, log_level="info")
