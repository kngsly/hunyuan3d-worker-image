# hunyuan3d-worker-image

Source for a Docker image that runs a simple HTTP worker for Hunyuan3D-2.1 on Vast.ai:

- `GET /health` -> `{"status":"OK"}`
- `POST /generate` (multipart form field `image`) -> JSON `{success, glb_path}` or `{success:false, error}`
- `GET /download/{filename}` -> downloads the generated `.glb`

This repo exists because Docker Hub hosts images, not your Dockerfile. You can keep this source locally, or push it to any git remote later.

## Target Behavior (Goal)

- Deploy worker quickly: `docker pull` + container start should be about **3 minutes or less** on a decent host (excluding cold-cache GPU host variability).
- Process many image -> 3D requests.
- Later (not implemented yet): shutdown after ~15 minutes of no requests.

## Repo + CI (Recommended Setup)

1. Create a GitHub repo that contains only this folder (Dockerfile + worker code).
2. Use GitHub Actions to build/push to Docker Hub on every tag or push to `main`.
3. Vast.ai deployments always use Docker Hub tags (fast, consistent), not runtime installs.

### Create the GitHub repo (from your machine)

From this folder:
```bash
cd services/hunyuan3d-worker-image
git init
git add .
git commit -m "Initial hunyuan3d worker image"

# Requires GitHub CLI login (recommended) or a PAT/SSH remote.
gh auth login
gh repo create kngsly/hunyuan3d-worker-image --public --source . --remote origin --push
```

### Configure Docker Hub secrets in GitHub

In the GitHub repo settings -> Secrets and variables -> Actions:
- `DOCKERHUB_USERNAME` = your Docker Hub username (e.g. `kngsly`)
- `DOCKERHUB_TOKEN` = a Docker Hub access token

Then add the workflow file below (already included in this folder under `.github/workflows/`).

## Build + Push (Manual)

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
