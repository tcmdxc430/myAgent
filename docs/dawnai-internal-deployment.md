# DawnAI Internal Deployment

This document describes the first-month deployment mode where DawnAI calls myAgent as an internal service.

## Topology

- Public traffic continues to enter DawnAI at `http://43.167.193.216/`.
- DawnAI backend calls myAgent through the Docker network, for example `http://myagent:8080`.
- myAgent does not publish port `8080` to the public host.
- Streamlit is not deployed for this mode.

## Server Files

1. Copy `.env.production.example` to `.env.production`.
2. Fill real values for:
   - `AUTH_SECRET`
   - `POSTGRES_PASSWORD`
   - LLM / embedding keys used by the agents
   - `BAIDU_OCR_API_KEY`
   - `BAIDU_OCR_SECRET_KEY`
3. Keep `.env.production` on the server only.

## Start myAgent

Run from the myAgent repository on the server:

```bash
docker compose --env-file .env.production -f compose.myagent.prod.yaml up -d --build
```

The compose file creates persistent volumes for:

- PostgreSQL data
- Chroma data
- Xiaohongshu browser profile and article assets

## DawnAI Backend Environment

Set these environment variables on the DawnAI backend service:

```bash
MYAGENT_BASE_URL=http://myagent:8080
MYAGENT_AUTH_SECRET=<same value as myAgent AUTH_SECRET>
MYAGENT_INGEST_TIMEOUT_MS=180000
```

If DawnAI backend runs in a separate Compose project, attach both projects to a shared external Docker network and keep the myAgent service alias as `myagent`.
The myAgent production compose file creates a named Docker network called `dawnai_internal` by default.
Attach the DawnAI backend container to that network, or set `MYAGENT_DOCKER_NETWORK` before starting myAgent if the server already uses another internal network name.

## Verification

From the server or DawnAI backend container:

```bash
curl http://myagent:8080/health
```

From outside the server, this should fail:

```bash
curl http://43.167.193.216:8080/health
```

From DawnAI frontend, use the Article Ingest page to submit a normal article URL. The request should go:

```text
DawnAI frontend -> DawnAI backend -> myAgent /ingest/article
```

The DawnAI backend writes audit log entries for import success and failure.
