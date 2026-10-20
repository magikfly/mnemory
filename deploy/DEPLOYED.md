# mn-deployed — source of the running mnemory image

Branch `mn-deployed` = upstream tag `v1.15.1` + the exact delta that goes into the image.

- **Currently running**: `ghcr.io/magikfly/mnemory:latest` == the `v1.15.0`-based build
  (`mnemory:v1.15.0-patched7`, id `2754d56485ab`), container `mnemory` on the unraid homeserver.
  The `v1.15.1` merge is committed here but **not yet published** — it lands when CI rebuilds and
  Watchtower recreates the container. Update this bullet once that happens.
- Delta vs `v1.15.1`: **11 files, 1601 insertions / 92 deletions, 80 hunks** —
  `mnemory/api/fsck.py`, `api/schemas.py`, `categories.py`, `consolidation.py`, `fsck.py`,
  `llm.py`, `maintenance.py`, `memory.py`, `output_guards.py`, `prompts.py`, `storage/vector.py`.
- **Authoritative diff: `deploy/deployed-vs-v1.15.1.diff`** (applies clean on a pristine `v1.15.1`
  checkout; result verified byte-identical to this branch by md5 across all 11 files, all modules
  compile).
- `deploy/deployed-vs-v1.15.0.diff` and `deploy/local-patches.diff` are **historical** — they
  describe the pre-merge tree and will not apply to `v1.15.1`.

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
`latest` + the version tag. Watchtower recreates the container on a newer `latest` digest.

To reproduce production from this branch:
```
git checkout v1.15.1 && git apply deploy/deployed-vs-v1.15.1.diff
# or simply: git checkout mn-deployed   (the branch *is* v1.15.1 + delta)
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
