# Deploy scpviz Dash Behind Cloudflare

This app is a Python Dash server and should run on a container/VM. Cloudflare sits in front via a Tunnel.

## 1) One-time Cloudflare setup

1. In Cloudflare Zero Trust, create a **Tunnel**.
2. Add a **Public Hostname** for your app domain/subdomain (for example `scpviz.example.com`).
3. Point that hostname to service URL: `http://web:8050`.
4. Copy the generated tunnel token.

## 2) Build and deploy with Docker Compose

Use the Cloudflare compose file in this repo.

### Build command

```bash
docker compose -f docker-compose.cloudflare.yml build web
```

### Deploy command

Linux/macOS:

```bash
export CLOUDFLARE_TUNNEL_TOKEN="<your_tunnel_token>"
docker compose -f docker-compose.cloudflare.yml up -d
```

Windows PowerShell:

```powershell
$env:CLOUDFLARE_TUNNEL_TOKEN="<your_tunnel_token>"
docker compose -f docker-compose.cloudflare.yml up -d
```

## 3) Verify deployment

```bash
docker compose -f docker-compose.cloudflare.yml ps
docker compose -f docker-compose.cloudflare.yml logs -f web
docker compose -f docker-compose.cloudflare.yml logs -f cloudflared
```

Then open your Cloudflare hostname in browser.

## 4) Update / restart

```bash
docker compose -f docker-compose.cloudflare.yml build web
docker compose -f docker-compose.cloudflare.yml up -d
```

## 5) Stop

```bash
docker compose -f docker-compose.cloudflare.yml down
```

## Notes

- The app uses Redis session storage in this profile for stable sessions.
- The web service is not published to the host (`expose` only). Access is via Cloudflare Tunnel.
- Keep your tunnel token secret.
