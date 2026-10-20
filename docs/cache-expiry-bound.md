# Bounded foreground cache expiry

The audit cache reclamation change drained every expired key under a single
lock. Foreground expiry now removes at most 16 oldest entries per operation,
plus at most one explicitly addressed entry. The budget limits maintenance
work, not valid cache entries, TTL, request admission, or worker concurrency.

## Invariants

- Read monotonic time inside the same lock used to serialize writes. Replacement
  removes and reinserts the key, preserving timestamp order.
- `get` checks the requested entry even when older expired entries remain beyond
  the cleanup budget. Expiration retains the existing strict `age > TTL`
  boundary. Falsey values are preserved; nonpositive TTL stores nothing.
- Expired-prefix removals retain keys and values until after unlocking. Targeted
  removal/replacement retains the removed value until after unlocking. Slow or
  reentrant value finalizers cannot execute inside the cache lock.
- At most 16 maintenance entries and one targeted entry are retired per ordinary
  operation. This bounds entry visits/reference releases, not arbitrary Python
  finalizer execution time or the size of one value's object graph.
- Explicit `invalidate_prefix` remains an atomic full matching-key invalidation:
  O(total keys) scanning and O(matches) removals. `clear` atomically detaches the
  dictionary and destroys it outside the lock. These explicitly bulk APIs are
  not bounded-expiry operations; there is no hidden deferred disposal queue.
- This repository's `TTLCache` has no `incr` API, integer delta, or counter TTL
  contract. No speculative counter API was introduced.

## Progress and memory

One write admits at most one entry. While at least 16 expired entries remain,
each write reclaims 16 and adds at most one: total population decreases by at
least 15. Reads and invalidations add none and reclaim up to 16. A fixed expired
backlog B drains in at most ceil(B / 16) accesses without new writes. New arrivals
cannot starve older expired entries because the queue is timestamp ordered.

For steady unique writes at rate R and fixed TTL T, the live population is
approximately R*T (including the strict boundary), and the expiration service
capacity is 16*R entries/s versus R newly expiring entries/s. Bursts can leave
an expired backlog, but continued activity drains it. Without any activity,
expired entries remain resident but population does not grow. Arbitrarily many
valid writes within the TTL cannot have an absolute memory bound without dropping
valid data; this change deliberately does not do that.

Tests prove exact per-operation population deltas, eventual drain, and stable
population of 11 at one new key/second with TTL 10 over 2,000 additional writes.

## Historical synthetic measurements

Lazy expiry, unbounded expiry, and bounded expiry: five repetitions per
implementation per dataset, alternating order (60 cases), using a contention
harness. Controlled cache clock, 20/1k/10k/100k expired short category-list values
and one valid value. Three real operation wrappers use fixed local retrieve,
insert and embedding stubs with gated dependency release. No remote inference
or production access. Same outputs and eventual cleanup asserted.

Median sweep / concurrent valid-hit duration, milliseconds:

| Expired keys | Lazy expiry | Unbounded expiry | Bounded expiry |
|---:|---:|---:|---:|
| 20 | 0.016 / 0.006 | 0.039 / 0.475 | 0.043 / 0.450 |
| 1,000 | 0.014 / 0.006 | 0.922 / 1.456 | 0.049 / 0.371 |
| 10,000 | 0.016 / 0.006 | 7.136 / 7.364 | 0.054 / 0.519 |
| 100,000 | 0.015 / 0.005 | 79.413 / 79.745 | 0.042 / 0.456 |

Thread handoff contributes to valid-hit times. Largest corrected sweep across
all 20 samples: 0.091 ms (rounded upward). Lazy expiry leaves expired cold entries
resident, so its faster miss is not a safe alternative.

At 100k keys, retrieve/insert/embed completion after dependency release changed
from 5.382/5.437/5.466 ms with unbounded expiry to 0.220/0.191/0.244 ms with bounded expiry.
Maximum event-loop lag per run, median: 5.703 ms unbounded, 0.842 ms bounded.
These are local synthetic measurements, not production latency attribution.
Production cache entry counts must not be inferred from memory-record counts.

## Historical validation

```
uv run pytest tests/test_cache.py tests/test_cache_bounded.py \
  tests/test_resource_audit.py tests/test_metrics.py tests/test_maintenance.py -q
uv run ruff check mnemory/cache.py tests/test_cache_bounded.py tests/test_resource_audit.py
git diff --check
```

The original check recorded 81 passing tests; lint and diff checks passed.
These historical results do not qualify later prompt or fixture revisions.
Tests include beyond-budget expired
key access, strict time boundary, overwrite ordering, serialized clock reads,
concurrent mutation/prefix invalidation, reentrant and blocked destructors, and
disabled TTL. Release validation must independently run the required E2E suite.
