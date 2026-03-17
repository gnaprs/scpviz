# Deploy scpviz Dash app

This project includes deployment-ready files for both Render and Docker.

## 1) Render deployment (recommended quick path)

### Prerequisites
- Push this repo to GitHub.
- Create a Render account.

### Steps
1. In Render, create a new **Blueprint** service and connect your repo.
2. Render will read `render.yaml` and create:
   - `scpviz-redis` (Redis)
   - `scpviz-dash` (web app)
3. Deploy.
4. Open the generated web URL.

### Notes
- `REDIS_URL` is wired from the Redis service automatically.
- Gunicorn is configured with `--workers 1 --threads 4` to stay compatible with current in-process workflows.

## 2) Docker Compose deployment (local or VM)

### Run
```bash
docker compose up --build
```

App URL:
- `http://localhost:8050`

### Services
- `web`: Dash app + Gunicorn
- `redis`: Redis 7

## 3) Environment variables

- `REDIS_URL`  
  Example: `redis://redis:6379/0` or managed Redis URL in cloud.

- `SCPVIZ_SESSION_TTL_SECONDS`  
  Session expiry in seconds. Default: `21600` (6 hours).

- `SCPVIZ_REDIS_PREFIX`  
  Optional key prefix. Default: `scpviz`.

## 4) Post-deploy checks

1. Open app in two separate browsers/incognito sessions.
2. Upload different files in each session.
3. Confirm each user sees only their own data/results.
4. Run QC/DE and verify downloads work.
5. Restart service and verify app still responds.

## 5) Scaling note

Current Redis-backed session implementation stores session payloads server-side and supports multi-user use.
For very large datasets or high throughput, consider moving large analysis objects to persistent/object storage and keeping only references in Redis.
