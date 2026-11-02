# Development

## Setup

> **Python 3.14 is not supported.** fastembed's BM25 model uses `py_rust_stemmers`,
> a Rust extension that segfaults on Python 3.14 due to a PyO3/C API incompatibility
> ([qdrant/fastembed#576](https://github.com/qdrant/fastembed/issues/576)).
> Use **Python 3.11–3.13** (3.13 recommended).

```bash
# Install with all optional dependencies
pip install -e ".[all,dev]"

# Run with minimal config (uses OPENAI_API_KEY, data in ~/.mnemory/)
export LLM_API_KEY=sk-your-key
mnemory

# Or run the module directly
python -m mnemory.server
```

## Tests

### Unit Tests

Fast, no API key needed. This is the default `pytest` run:

```bash
pytest tests/
```

### End-to-End Tests

E2e tests exercise the full pipeline (extraction -> storage -> search) with real LLM calls and embedded Qdrant. They are **excluded from the default `pytest` run** via `addopts = "-m 'not e2e'"` in `pyproject.toml`.

```bash
# Requires LLM_API_KEY or OPENAI_API_KEY
pytest -m e2e -v
```

**When to run e2e tests:**
- After changing `prompts.py` (extraction prompts, dedup logic)
- After changing `memory.py` (business logic, add/search/update/delete)
- After changing `storage/vector.py` or `storage/artifact.py`
- After changing `sanitize.py` (anti-injection, boundary tags)
- Before releases

**Requirements:**
- `LLM_API_KEY` or `OPENAI_API_KEY` environment variable set (auto-skips if missing)
- ~5 minutes runtime (LLM calls + embedding generation)
- `pytest-timeout` installed (120s per test, 180s for `find_memories`)

### All Tests

```bash
pytest -m '' -v
```

## Linting

```bash
ruff check mnemory/
ruff format mnemory/
```

## UI Development

To modify the UI, edit files in `mnemory/ui/static/` (JS, HTML) or `mnemory/ui/src/input.css` (Tailwind). Rebuild CSS after changes:

```bash
# One-time: download Tailwind CLI (https://github.com/tailwindlabs/tailwindcss/releases)
# Then:
make ui-build    # Build minified CSS
make ui-watch    # Watch mode for development
```

The UI uses Alpine.js + Tailwind CSS + Chart.js + D3.js — all vendored as static files. Zero external requests at runtime.

## Docker Build

```bash
# Build for linux/amd64 and linux/arm64 (for Kubernetes deployment)
docker buildx build --platform linux/amd64,linux/arm64 -t ghcr.io/fpytloun/mnemory:latest .

# Publish with buildx
docker buildx build --platform linux/amd64,linux/arm64 --push -t ghcr.io/fpytloun/mnemory:latest .
```
