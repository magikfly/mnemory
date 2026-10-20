# Resource audit optimizations

These changes preserve worker limits, retention, retry behavior, and search
quality. They do not change deployment configuration or promise a process-RSS
target.

## Implemented

- Fsck collects the existing eligible payload corpus without embeddings.
  Incremental selection and random caps precede dense-only retrieval in batches
  of 256. Unchanged/shared eligible neighbors remain available to cluster
  evaluation. Security checks, checked-at stamping, and phase budgets are unchanged.
- Duplicate queries pass a conservative margin below 0.75 to Qdrant, preserving
  exact ties even with strict float32 server comparisons.
  The corpus-sized limit is deliberately retained: a small top-k would lose
  qualifying edges. Python eligibility and threshold checks remain.
- TTL caches reclaim expired cold entries on subsequent cache activity using a
  timestamp-ordered queue, with synchronized reads/writes/invalidation. Replacing
  a key moves it to the newest position. Nonpositive TTLs store nothing.
- Session listing runs its unchanged synchronous operation in a worker thread.
  This protects event-loop responsiveness; it does not reduce scan allocations.
- Metrics filter active/legacy memory revisions and nonterminal fsck operations
  at storage. Existing Python guards, expiry parsing, atomic gauge replacement,
  freshness and singleflight behavior remain.

## Deliberate boundaries

- Payloads still scale with the eligible fsck corpus. Exact active-revision
  collapse, security checks and cluster evaluation need these records. This
  change removes corpus vector residency, not all corpus metadata.
- Duplicate eligibility is not pushed down beyond existing scope/revision
  filters. Adding an eligible-ID filter can change which neighbors survive the
  existing corpus-sized limit when excluded points rank higher, and changes
  approximate-search planning. That needs separate recall/parity qualification.
- Failed-session metric fields are already projected to consumed fields.
  `memory_ids` is needed for exact counts; the full `consolidation_plan` currently
  establishes Python truthiness. Replacing either with a stored summary needs a
  migration and parity proof for legacy/null/empty values.
- Session indexed pagination is deferred: current Python ordering, text search,
  access filtering and exact totals need a separate storage contract.
- No admission changes, queue rejection, worker reductions, model unloading,
  global pool replacement or speculative instrumentation.
- Cache expiration is activity-driven; an inactive cache may retain its last
  expired batch until the next operation. No per-cache background thread.
- Idle-session pre-hydration filtering is deferred. The existing cutoff is
  sampled *after* hydration; moving it before the scan could delay sessions that
  cross the threshold during scanning by one maintenance interval. A tested
  prototype showed why preserving this timing matters. Streaming also must
  preserve per-user recovery ordering.

## Reproducible allocation experiment

Run `uv run python -m benchmarks.fsck_allocations --corpus 10000 --selected 512`.
The synthetic storage response creates independent Python floats, like a remote
JSON response. It uses production scroll/dense retrieval helpers and tracemalloc.
It does not measure Qdrant execution, networking, RSS, or full fsck latency.

With Python 3.12, 1,536 dimensions and 512 selected memories:

| Eligible corpus | Eager peak bytes | Bounded peak bytes | Dense vectors transferred |
| ---: | ---: | ---: | ---: |
| 1,000 | 50,172,469 | 13,202,154 | 1,000 → 512 |
| 10,000 | 499,602,772 | 16,131,474 | 10,000 → 512 |

Peak retained dense vectors fall from corpus size to 256. Dense duplicate graphs
can still require corpus-sized result/union state. Warm/cold production RSS and
p95 latency require separate measurement; Python allocator
high-water behavior means allocation reductions need not map directly to RSS.

## Historical validation and limitations

- `uv sync --python 3.12 --extra all --extra dev`
- `uv run pytest tests/test_resource_audit.py tests/test_cache.py tests/test_fsck.py tests/test_fsck_journal.py tests/test_metrics.py tests/test_maintenance.py tests/test_consolidation.py tests/test_sessions_api.py -q`:
  **353 passed**, one existing Pydantic forward-reference warning.
- `uv run ruff check mnemory/ tests/test_resource_audit.py benchmarks/fsck_allocations.py`
  and `git diff --check`: passed.
- `uv run pytest -m e2e tests/test_e2e.py -q`: **76 passed, 4 failed**,
  using real-provider inference and the repository fixture's temporary
  embedded Qdrant/filesystem backends. The four failures are
  `TestAgentScoping.test_sub_agent_access`, `TestCRUD.test_update_memory_content`,
  `TestCRUD.test_delete_memory`, and `TestCRUD.test_delete_memory_with_artifacts`.
  All four also reproduced in their full test classes on the unchanged
  baseline. They were not weakened or skipped.
- Python 3.14's initial E2E attempt crashed in `py_rust_stemmers` during BM25
  initialization; validation therefore used supported Python 3.12.

This historical E2E run was **not green** and does not qualify any later
revision or waive release tests. Remote-Qdrant
threshold/approximate-search parity and live latency/RSS are not proven by the
embedded and synthetic checks. No production data or deployment was modified.
