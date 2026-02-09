# hunyuan3d-worker-image

Source for a Docker image that runs a simple HTTP worker for Hunyuan3D-2.1 on Vast.ai:

- `GET /health` -> `{"status":"OK"}`
- `POST /generate` (multipart form field `image`) -> JSON `{success, glb_path}` or `{success:false, error}`
- `GET /download/{filename}` -> downloads the generated `.glb`

This repo exists because Docker Hub hosts images, not your Dockerfile. You can keep this source locally, or push it to any git remote later.

## Build + Push (Docker Hub)

```bash
cd services/hunyuan3d-worker-image

export IMG="kngsly/hunyuan3d-worker"
export TAG="dev-$(date +%Y%m%d-%H%M)"

docker build -t "${IMG}:${TAG}" .
docker push "${IMG}:${TAG}"

echo "${IMG}:${TAG}"
```

## Vast smoke (from `services/vast-ai`)

```bash
cd services/vast-ai
source .venv/bin/activate

python ./rent.py --smoke --stop-on-fail \
  --image kngsly/hunyuan3d-worker:YOUR_TAG \
  --test-image https://cdn.crateyard.com/images/assets/1/3/eec071f9-9eb.png \
  --smoke-output-dir /media/user/Zoomer/projects/crateyard-website/hunyuan_outputs \
  --startup-timeout-sec 1200 --generate-timeout-sec 7200 \
  --request-logs-on-fail --request-logs-tail 8000
```

## Notes

- This image intentionally **does not** depend on Blender `bpy`. It generates a shape-only `.glb` for reliability.
- If you later want textures, we can add a non-bpy OBJ->GLB path or a Blender-based sidecar approach.

