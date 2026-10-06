# mn-deployed — source of the running mnemory image

Branch `mn-deployed` = upstream tag `v1.15.0` + the exact delta running in production today.

- Image: `mnemory:master` == `mnemory:v1.15.0-patched7` (id `2754d56485ab`), container `mnemory` on the unraid homeserver.
- Delta vs `v1.15.0`: **7 files, 284 insertions / 32 deletions, 31 hunks** — `mnemory/consolidation.py`, `fsck.py`, `llm.py`, `maintenance.py`, `memory.py`, `prompts.py`, `storage/vector.py`.
- Authoritative diff: `deploy/deployed-vs-v1.15.0.diff` (applies clean on a pristine `v1.15.0` checkout; result verified byte-identical to the deployed tree by md5, all modules compile).
- `deploy/local-patches.diff` (374 lines, 5 files) is kept for history. It is **incomplete** — it predates the `fsck.py` and `maintenance.py` work and does not reproduce production. Use `deployed-vs-v1.15.0.diff`.

## What each patch does

### 1. Consolidation reliability (`llm.py`, `consolidation.py`, `storage/vector.py`)
Root cause: consolidation ran 10 notes per batch with `max_tokens=16384` against the small local extractor; the JSON reply was cut at the token limit, parsing raised `ValueError`, the session was stamped `consolidation_state = "failed"`, and the scan filter did not include `"failed"` — so those sessions were never re-scanned and notes silently stopped consolidating.
- `llm.py:366` `salvage_json_objects(text, key)` — recovers complete fact objects from a reply truncated at `max_tokens`.
- `consolidation.py` — queue-based batches, halve-and-retry on unparseable output, re-queue of raw memories no fact references (max 2 attempts, `attempt_count`), plus a normalized-text dedup guard before write (small models re-emit the same fact on each re-queued pass).
- `storage/vector.py` — scan filter `MatchAny(["idle","consolidating","failed"])` (~line 2614) and `claim_consolidation` accepting `"failed"` (~line 2289) so a once-failed session is eligible again.

### 2. fsck reliability (`fsck.py`, `prompts.py`, `memory.py`)
- `fsck.py:68` `_FSCK_MAX_TOKENS = 4096` at all fsck call sites; `_parse_llm_json()` salvages complete issues from a truncated dedup/quality reply and never raises (a bad batch is skipped, the scan continues).
- `prompts.py` fsck schemas: `memory_id` gets `"pattern": "^[0-9]+$"` + `"maxLength": 2`, and the id arrays get `minItems: 1` / `maxItems: 20` (3 schema sites: ~4072, ~4097, ~4307). llama.cpp structured output accepts `pattern`. Prevents the unresolvable-ID drops that were silently discarding ~10 fixes per auto run.
- `memory.py` — `remember` dedup candidate pool `limit=10` (was 5).
- `prompts.py` dedup decisions — SKIP bias, content-hygiene rules, empty/missing-text backfill from the Stage-1 fact (marked `LOCAL-PATCH`, ~3667 and ~3701).

### 3. Auto-fsck coverage (`maintenance.py`)
`include_raw=True` at both `run_check` call sites (line 188 scheduled full check, line 433 incremental auto run), so the periodic fsck sees raw memories instead of only consolidated ones. Auto run every 4h, auto-apply conf>=0.85 sev>=low.

## Build + deploy

`deploy/Dockerfile` is the deployed overlay (`FROM mnemory:base`, `COPY mnemory/`, `pip install --no-deps .`). `mnemory:base` (`5b1d7f279c7d`) is the unpatched v1.14.0 baseline — build base only, never run it. Host build dir: `/mnt/user/appdata/mnemory-patched-src/`. Deploy is Portainer stack #20 (`/data/compose/20/docker-compose.yml`, project name `mnemory`); the host has no compose plugin, so recreate with the compose v2 binary per netvault `services.mnemory.recreate`.

To reproduce production from this branch:
```
git checkout mn-deployed && git apply deploy/deployed-vs-v1.15.0.diff  # from v1.15.0
# then build with deploy/Dockerfile from a tree whose mnemory/ matches this branch
```

## Not deployed

Branch `mn-fork` (commit `3bb9753`) carries the Open WebUI **filter** provenance patch (`deployed/` dir: baseline + mn-fork filter, patch, test, deploy script, `PATCHES.md`, `build_fork.py`). It changes how recalled memories are labelled in the chat prompt — staged, tested locally, **not applied to the OWUI function DB** and not in the image.
