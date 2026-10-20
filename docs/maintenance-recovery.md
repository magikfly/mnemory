# Maintenance recovery boundaries

## Status is not permission

`mnemory_fsck_operations_nonterminal` retains its historical definition:
planned, applying, and failed journal rows. The aged-row gauge is not a count
of abandoned or safely recoverable jobs. Failed-session eligibility is a
preliminary metadata classification, not an automatic retry queue. The
explicit failed-session retry policy is unchanged.

New maintenance fsck journals carry `mnemory.maintenance.v1`. Existing issue
identity is resolved by check and issue before inspecting targets. Existing
nonterminal journals are diagnosed without reconstructing a new plan or
token, changing status, or replaying side effects. Terminal issue results are
not applied again. Duplicate journal identities fail closed.

## Why automatic takeover is blocked

The disposable remote-Qdrant experiment in `benchmarks/maintenance_fencing.py`
pauses a worker after its ownership read, transfers ownership in the journal,
then resumes its conditional target write. Qdrant 1.16.2 accepts the stale
target write even with strong write ordering. Filters and write ordering are
not a cross-collection transaction. Lease renewal or cancellation of an
asyncio await does not fence a thread's subsequent writes.

Automatic consolidation no longer invokes recovery on every consolidating
session. These records receive an `ownership_unknown` diagnosis; their token,
plan, state, and generation are untouched. No lease is invented for legacy
workers. Existing explicit retry APIs are not redesigned by this change and
must not be treated as proof of cross-collection takeover safety.

Safe takeover requires a target-side fencing primitive enforced atomically
with each mutation (including creation and every source transition), with a
provable handoff for all targets, or externally established writer quiescence
before an explicitly authorized recovery. Neither is established here.
Initial concurrent journal creation is also not made into a distributed
transaction by this change. Do not enable multi-writer replay on that basis.

## Bounded scheduling

Each consolidation tick requests one due page of at most 32 summaries and
one diagnostic page of at most 16 summaries. Separate point-ID cursors wrap
after exhaustion. New IDs behind the cursor are visited on the next sweep.
For a finite stable eligible set, all entries are visited within a full
sweep; continual restarts or unbounded arrivals do not have a finite latency
guarantee. Cursors are process-local hints, never durable completion state.

The due page filters idle state, idle timestamp, absence of ownership, and
next eligibility at the storage boundary. It includes attempt counts and
hydrates summaries and append events only for due work. Empty-input sessions
are deferred for one day; a normal session update clears this scheduling
hint. The diagnostic sweep includes missing/invalid timestamps and legacy
ownership, independent of eligibility. It never terminalizes records.

Bounds apply to scheduler rows/pages, not the number of linked memories or
append events in an individual session, full fsck corpus metadata, metrics
aggregation, or Qdrant's internal query cost. Date/state payload indexes are
ensured on first scheduler use without a data backfill.

## Rollout

Do not assume mixed-version workers obey these boundaries: old maintenance
workers can still reset/replay consolidating rows, and old writers do not
clear new scheduling hints. Drain old workers before relying on the new
behavior. No migration or production repair is performed by this work.
Diagnosis fields are additive observations, not authoritative lifecycle
states; existing journal plans and identities remain intact.

Lifecycle counters are emitted by the synchronous fsck worker. Cancelling
its awaiting coroutine does not emit a false worker completion. Diagnostic
counters count observations, not distinct jobs, and use bounded labels.
