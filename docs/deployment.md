# Deployment

## Python Version

mnemory requires **Python 3.11–3.13**. Python 3.14 is not yet supported.

The `fastembed` dependency (required for BM25 hybrid search) uses a Rust
extension (`py_rust_stemmers`) that segfaults on Python 3.14 due to a
PyO3/C API incompatibility. This is a known upstream issue
([qdrant/fastembed#576](https://github.com/qdrant/fastembed/issues/576)).
**Python 3.13 is recommended for production.** The Docker image uses
`python:3.13-slim`.

## Installation Methods

### Using uvx (recommended)

```bash
uvx mnemory
```

mnemory starts on `http://localhost:8050/mcp`, stores data in `~/.mnemory/`.

### Using pip

```bash
pip install mnemory
mnemory
```

### Using Docker

```bash
export OPENAI_API_KEY=sk-your-key
docker-compose up -d
```

## Production Setup

For production, use remote Qdrant for vectors and S3/MinIO for artifacts:

```bash
docker run -d \
  -p 8050:8050 \
  -e OPENAI_API_KEY=sk-your-key \
  -e LLM_BASE_URL=https://your-litellm-proxy/v1 \
  -e LLM_MODEL=gpt-5.4-mini \
  -e QDRANT_HOST=qdrant.example.com \
  -e ARTIFACT_BACKEND=s3 \
  -e S3_ENDPOINT=http://minio.example.com:9000 \
  -e S3_ACCESS_KEY=admin \
  -e S3_SECRET_KEY=secret \
  -e MCP_API_KEYS='{"your-api-key": "your-username"}' \
  -v mnemory-data:/data \
  ghcr.io/fpytloun/mnemory:latest
```

### Docker Build

```bash
# Build for linux/amd64 and linux/arm64 (for Kubernetes deployment)
docker buildx build --platform linux/amd64,linux/arm64 -t ghcr.io/fpytloun/mnemory:latest .

# Publish with buildx
docker buildx build --platform linux/amd64,linux/arm64 --push -t ghcr.io/fpytloun/mnemory:latest .
```

GitHub Actions publishes `latest` from `main` and `vX.Y.Z` for published,
non-prerelease server releases. Before first use, an owner must set the GHCR
package visibility to **Public** in GitHub Packages settings
(`fpytloun/mnemory`). Images can then be pulled without authentication:
`docker pull ghcr.io/fpytloun/mnemory:latest` or
`docker pull ghcr.io/fpytloun/mnemory:v1.0.0`.

## Authentication

mnemory can be deployed either as a standalone service (API keys) or behind Cognis (JWT service auth).

### Single API Key

For simple setups, set a single shared API key:

```bash
MCP_API_KEY=your-secret-key
```

All clients must include `Authorization: Bearer your-secret-key` in requests. No user binding — `user_id` must come from headers or tool parameters.

### Multi-User API Keys (`MCP_API_KEYS`)

Map API keys to user IDs so the LLM doesn't need to pass `user_id` in every tool call:

```bash
MCP_API_KEYS='{"mnm-key-for-filip": "filip", "mnm-shared-service-key": "*"}'
```

- `"key": "username"` — authenticates AND binds `user_id=username` to the session
- `"key": "*"` — authenticates only (wildcard), `user_id` must come from identity headers or tool parameter

See [Configuration](configuration.md#authentication) for full identity resolution details.

### Cognis JWT Service Auth

When Cognis is the client, configure Mnemory to trust Cognis-issued ES256 JWTs:

```bash
export MNEMORY_JWT_PUBLIC_KEY=/path/to/cognis-public.pem
# or
export MNEMORY_JWKS_URL=https://cognis.example.com/.well-known/jwks.json
```

Mnemory expects:

- `Authorization: Bearer <jwt>`
- `iss="cognis"`
- `aud` includes `"mnemory"`
- `sub` = user identity
- optional `agent_id` claim (falls back to `X-Agent-Id` header)

API keys remain supported for backward compatibility with standalone clients.

## Kubernetes

### Health Probes

Set `MGMT_PORT` to serve unauthenticated `/health` and `/metrics` on a separate port:

```bash
MGMT_PORT=9090          # /health and /metrics on port 9090, no auth
MGMT_HOST=127.0.0.1     # Optional: bind management to localhost only
```

This is the recommended setup for Kubernetes liveness/readiness probes and Prometheus scraping. The main port continues to expose authenticated `/health` and `/metrics` for service-to-service checks such as Cognis provider health. See [Configuration](configuration.md#management-port) for details.

### Session Persistence

By default, mnemory persists sessions to a local SQLite database (`~/.mnemory/sessions.db`). For clustered deployments with multiple replicas, use Redis:

```bash
SESSION_BACKEND=redis
REDIS_URL=redis://redis.example.com:6379/0
```

Redis is an optional dependency — install with `pip install mnemory[redis]`. If `REDIS_URL` is set and `SESSION_BACKEND` is not explicitly set, Redis is auto-selected.

Sessions use a write-through cache (in-memory + backend) with lazy loading on cache miss. Losing a session is harmless — the client gets a fresh one on next recall.

### Stateless HTTP

mnemory runs in stateless HTTP mode (`stateless_http=True`) for Kubernetes compatibility. No sticky sessions or WebSocket connections required.

## Monitoring

mnemory exposes Prometheus metrics at `/metrics`. See [Monitoring](monitoring.md) for available metrics, Prometheus scrape config, and a pre-built Grafana dashboard.
