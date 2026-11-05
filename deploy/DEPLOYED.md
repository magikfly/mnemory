# mn-deployed — the patched mnemory tree that CI builds

Branch `mn-deployed` is an upstream release tag plus the local delta that goes into
`ghcr.io/magikfly/mnemory:latest`. It is **not** the fork's default branch. `main` exists to host
`.github/workflows/mnemory-ghcr.yml` so its `schedule:` trigger fires — GitHub only reads
schedules from the default branch — and the workflow then checks *this* branch out explicitly
with `ref: mn-deployed` and pushes its upstream-merge commits back here. So `main` always shows
many commits behind: by design, not a missed merge. **Never merge this branch into `main`.** It
carries upstream's `docker-publish.yml`, which triggers on `push: branches: [main]` and tags
`ghcr.io/fpytloun/mnemory`; a merge push would try to publish this patched tree into the upstream
author's package namespace.

**Scope of this file.** Why each local patch exists, upstream-merge friction, and the build
procedure — the material that has to travel with the code. Which image, tag or digest is running
on the homeserver is **deployment state and does not belong here**: read it from the netvault
(`services.mnemory`, keys `image`, `deploy`, `source_of_truth`), which is where those claims get
stamped and corrected. The previous header carried a "Currently running" bullet that went two
releases stale while telling the reader to update it.

## Current delta

- Base: upstream tag `v1.15.2` (an ancestor of this branch, merged as `9b251aa`). Branch tip `81e5e48`.
- Delta vs `v1.15.2`, `mnemory/` only — **15 files, 3133 insertions / 126 deletions**:
  `api/fsck.py`, `api/memories.py`, `api/schemas.py`, `categories.py`, `config.py`,
  `consolidation.py`, `fsck.py`, `llm.py`, `logsink.py`, `maintenance.py`, `memory.py`,
  `output_guards.py`, `prompts.py`, `server.py`, `storage/vector.py`.
- Whole-branch delta including `deploy/`, `docs/`, `tests/` and `benchmarks/`: 52 files, 14084
  insertions. Re-measure both with `git diff --stat v1.15.2..mn-deployed`.
- **Authoritative diff: `deploy/deployed-vs-v1.15.1.diff`** — applies clean on a pristine
  `v1.15.1` checkout, result verified byte-identical by md5 across its 11 files, all modules
  compile. **No diff against `v1.15.2` exists yet; generate one at the next upstream merge** and
  re-run the md5 parity check before trusting it.
- `deploy/deployed-vs-v1.15.0.diff` and `deploy/local-patches.diff` are **historical** — they
  describe pre-merge trees and will not apply to the current base.

## What each patch does

### 1. Consolidation reliability (`llm.py`, `consolidation.py`, `storage/vector.py`, `maintenance.py`)
Root cause: consolidation ran 10 notes per batch with `max_tokens=16384` against the small local
extractor; the JSON reply was cut at the token limit, parsing raised `ValueError`, the session was
stamped `consolidation_state = "failed"` — and a failed session was never re-scanned, so notes
silently stopped consolidating.
- `llm.py:366` `salvage_json_objects(text, key)` — recovers complete fact objects from a reply
  truncated at `max_tokens`.
- `consolidation.py` — queue-based batches, halve-and-retry on unparseable output, re-queue of raw
  memories no fact references (max 2 attempts, `attempt_count`), plus a normalized-text dedup guard
  before write (small models re-emit the same fact on each re-queued pass).
- `storage/vector.py` — `claim_consolidation` accepts `"failed"`, **and** the maintenance scan must
  *offer* failed rows. See §5: upstream v1.15.1 moved that scan and dropped the state again.
- `maintenance.py` — a failed session is due work, not a dead end.

### 2. fsck reliability (`fsck.py`, `prompts.py`, `memory.py`, `api/*`)
- `fsck.py` `_FSCK_MAX_TOKENS` / batch ceilings + per-pass ceilings, `_parse_llm_json()` salvages
  complete issues from a truncated dedup/quality reply and never raises (a bad batch is skipped, the
  scan continues). Truncation recovery is wired at all three salvage call sites.
- fsck quality/dedup detection: no-op findings (the model proposes the metadata it was just shown)
  and stale targets (deleted/changed since the scan snapshot) are suppressed, which is what stopped
  the permanent ~59-row "misclassified" pile.
- `prompts.py` fsck schemas: `memory_id` gets `"pattern": "^[0-9]+$"` + `maxLength`, id arrays get
  `minItems`/`maxItems`. llama.cpp structured output accepts `pattern`. Prevents unresolvable-ID
  drops that silently discarded ~10 fixes per auto run.
- `memory.py` — `remember` dedup candidate pool `limit=10` (was 5).
- `prompts.py` dedup decisions — SKIP bias, content-hygiene rules, empty/missing-text backfill from
  the Stage-1 fact (marked `LOCAL-PATCH`).

### 3. Auto-fsck coverage (`maintenance.py`)
`include_raw=True` so the periodic fsck sees raw memories, not only consolidated ones. Since
v1.15.1 the **incremental auto path runs inside upstream's `_run_fsck_worker`**, so the flag lives
there (one site), and the manual run-now site keeps `include_raw=True, incremental=False`. Both
sites now also log `skipped=` / `superseded=` — `apply_check` returns them and the old log line
swallowed them, which is how 71 stuck rows stayed invisible.

### 4. Write-time noise guards (`output_guards.py`, `categories.py`, `api/`)
`output_guards.py` (new, 485 lines) refuses junk at extraction: meta-observations ("user was
reminded to…"), greeting narration, meta-indexicals, date echoes, quote fragments, speech-act
laundering across roles, and ASCII/typographic punctuation blind spots. `categories.py` tolerates
invented categories the small model emits instead of dropping the whole fact. Log-only by default;
gated so it can be flipped to enforcing per deployment.

### 5. Merge notes — upstream v1.15.1 (2026-10-19, commit `38211f1`)
Upstream "improve memory lifecycle cleanup and maintenance reliability". Merged 2026-10-20; the
nightly CI had correctly refused (run 39372920022, conflict, old image kept serving).

**Conflict** — `maintenance.py`: upstream routed the auto fsck through a new `_run_fsck_worker`
(lifecycle telemetry even if the await is cancelled). Resolution keeps their wrapper and carries our
`include_raw=True` into it.

**Regression caught** — upstream replaced `scan_consolidation_candidates()` with the paged
`SessionSummaryStore.maintenance_page()`, whose due-work filter selects **only `"idle"`**. Failed
sessions are still stamped by `consolidation.py` (5 sites); `update_consolidation_state` neither
re-arms them nor stamps `maintenance_next_at`, and `diagnose_maintenance` "never changes lifecycle
state". Under the new scan a failed session is permanently invisible — i.e. §1's bug returns,
silently. Fix: `"failed"` added to the due + diagnostic filters and to `_run_consolidation`'s
post-hydration gate. `scan_consolidation_candidates` still exists but is now **dead code in the
product path** (only tests call it) — do not "clean it up", it documents the old contract.
Guard: `tests/test_fork_failed_session_eligibility.py`.

**Independent same-bug fixes, now from both sides** (keep ours; upstream's is also merged):
- Upstream fictionalised the few-shot examples that were real user data (Malibu/Stephen
  King/Kurilian Bobtail cat, the "2.5 fields" exchange, the 2015 Skoda Octavia, and the
  `project:domecek/…` category example). Same class as our `fix(prompts): stop few-shot examples
  leaking as user facts` and the metadata-leak fix.
- Upstream now skips fsck operations already `committed/skipped/superseded` at apply time and
  narrowed `allowed_statuses` to `("planned",)`. Ours suppresses the same pile at detection time.
  Complementary.
- Upstream added `include_vectors=False` on the fsck scroll, batched `get_dense_vectors` in 256-id
  chunks, and a `1e-7` float32 tie margin on duplicate search (both call sites take
  `search_by_vector_ids`, which is the overload that accepts `score_threshold`).

**Qualification** (deployed image, Python 3.13, embedded Qdrant, non-E2E):
`1735 passed, 20 skipped, 80 deselected, 4 failed`. All four failures pre-date the merge on a
parent — none were introduced here:
- `test_fsck::test_duplicate_prompt_includes_metadata_tags` (fork) — upstream test asserting the
  pre-patch prompt shape. Fork commit `5fe0193` intentionally stopped emitting `type: fact` tags in
  the duplicate prompt because that metadata leaked into stored memory text. Stale expectation, not
  a live bug; either update the test or re-add the tags without the leak.
- `test_fsck::test_apply_delete_skips_not_found` (fork) — failing pre-merge too, untriaged.
- `test_prompts::test_add_action` (fork) — `pinned=True` comes back `False` from
  `parse_extraction_response`; failing pre-merge too, untriaged.
- `test_resource_audit::test_threshold_keeps_all_ties` (**fails on pristine v1.15.1 here**) —
  embedded-vs-server Qdrant float32 threshold semantics; upstream qualified against server Qdrant.
Zero merge-induced regressions.

## Build + deploy

`deploy/Dockerfile` is the deployed overlay (`FROM mnemory:base`, `COPY mnemory/`,
`pip install --no-deps .`). The GHCR image is built by `.github/workflows/mnemory-ghcr.yml` on
branch `mn-deployed`: it merges the newest upstream tag if it is not already an ancestor, skips the
build when the version tag is already published (use `rebuild=true` to force), and publishes
`latest` + the version tag. Watchtower recreates the container on a newer `latest` digest. Scheduled at 18:30 and 20:30 UTC (two slots because GitHub delivers `schedule` events to this account up to ~5h late — measured 2026-10-21..27 and written into the workflow comments); both land before the 04:00 Europe/Amsterdam Watchtower poll, which reads `:latest` at 03:00 UTC.

To reproduce production from this branch:
```
git checkout v1.15.1 && git apply deploy/deployed-vs-v1.15.1.diff   # reproduces the v1.15.1-era state
# or simply: git checkout mn-deployed   (the branch *is* the current base + delta)
```

Manual build/publish when the schedule must not wait:
```
gh workflow run mnemory-ghcr.yml --repo magikfly/mnemory --ref mn-deployed \
  -f rebuild=true -f tag=v1.15.1
```

Rollback point for the v1.15.1 merge: tag `backup/pre-v1.15.1-merge` (= `b710b98`).

## Not deployed

Branch `mn-fork` (commit `3bb9753`) carries the Open WebUI **filter** provenance patch (`deployed/`
dir: baseline + mn-fork filter, patch, test, deploy script, `PATCHES.md`, `build_fork.py`). It
changes how recalled memories are labelled in the chat prompt — staged, tested locally, **not
applied to the OWUI function DB** and not in the image.

## 2026-10-27 - retrieval seam + TTL reaper (commit 2493941)

Patched files touched by this change, for upstream-merge conflict resolution:

- `mnemory/memory.py`
  - `find_memories(score_threshold=None)` - a caller threshold now wins over
    `SEARCH_SCORE_THRESHOLD`; that constant is vector-calibrated and was being
    applied to rerank rubric scores after the score field was overwritten.
  - the rerank stage keeps candidates the LLM omitted (score set to the floor,
    ranked last) and logs the omission, instead of deleting those memories.
  - `list_categories` scan cap 1000 -> 20000, with a warning if it ever clips.
  - `_get_available_categories` scan cap 1000 -> 20000, same warning.
  - `delete_all_memories` artifact-cleanup enumeration 1000 -> 20000, same.
- `mnemory/fsck.py` - new `_phase_ttl_reap()` (deterministic, consumes no LLM
  budget) called from `run_check` after the quality phase. It holds pinned
  rows, rows cited as evidence by a live memory, and fact/preference types.
- `mnemory/api/schemas.py` - `FindMemoriesRequest.score_threshold` and
  `SearchMemoriesResponse.stats`.
- `mnemory/api/memories.py` - the `/find` route forwards the threshold and
  returns the stats block.
- `tests/test_fork_rerank_seam.py` - new guard suite (10 tests).

If upstream rework touches `find_memories`, the rerank scoring loop, or the
fsck phase sequence, re-read this section first: those three behaviours
regressed silently and were only caught by instrumentation.
