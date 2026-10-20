# Bounded maintenance performance evidence

## Scope and verdict

The initial single full-fsck pair measured 54.28 s baseline / 75.62 s candidate.
Two additional matched pairs did **not reproduce** that regression. They do not
establish a runtime gain either: timing variance is large, and the measured fsck
phase source is identical between revisions. No evidence-based production
correction was justified. The broad end-to-end performance gate remains
**not established**; retain the safe partial changes without claiming a
general fsck speedup or deployment readiness.

The intended scheduler benefit is independently demonstrated as bounded
per-pass work, not faster processing of the entire queue in one pass.

## Full fsck repeated pairs

`benchmarks/maintenance_fsck.py` ran under the shared heavy-validation flock,
in fresh processes, in candidate/base then base/candidate order. Both use
4,096 synthetic memories, 1,536-dimensional decoded vectors, deterministic
model replies, two model threads, no warmup, and tracemalloc throughout the
check. Identical fixture setup precedes both timers. RSS is process high-water
including imports; Python allocation peak covers the check.

| Measurement | Base 1 | Candidate 1 | Base 2 | Candidate 2 |
|---|---:|---:|---:|---:|
| Wall seconds | 23.050 | 17.927 | 35.490 | 20.134 |
| Process CPU seconds | 23.069 | 17.942 | 35.469 | 20.161 |
| Duplicate phase wall seconds | 22.302 | 17.482 | 34.522 | 19.319 |
| Quality phase wall seconds | 0.526 | 0.342 | 0.662 | 0.525 |
| Metadata scroll wall seconds | 0.133 | 0.058 | 0.175 | 0.181 |
| Security regex wall seconds | 0.066 | 0.029 | 0.102 | 0.083 |
| Peak RSS bytes | 229482496 | 228667392 | 229195776 | 229171200 |
| Peak Python bytes | 20113267 | 20097103 | 20075619 | 20125077 |

Median wall time is 29.270 s baseline / 19.030 s candidate. This is descriptive,
not an attributed improvement. Only two repetitions per revision, shared-host
load, allocation instrumentation, and Python scheduling limit inference.
The earlier pair remains part of the evidence; it is not discarded.
Process CPU varies as well as wall time, so waiting alone does not explain
the variation. No particular environmental cause has been proven.

All four runs produce 410 issues and the same output SHA-256:
`b793b557a8364198efef2eef5ee9d0c52185a5159bb4eb3362b263238e793396`.
All measured phase source fingerprints are
`e3b180026c40a384bcac774a133a02fce926382f59d9183c23a76f4d3cba45c9`.

Every run performs 16 metadata scrolls, 16 vector retrievals, 4,096 neighbor
queries, and 3,276 metadata stamp writes. Maximum vector batch is 256;
4,096 vectors are decoded in total. Index operations and maintenance scans
are zero.

These are **synthetic backend calls**, not real Qdrant requests. The full
check benchmark does not execute apply/recovery, scheduler diagnosis, network
search, or durable stamp writes. Its call counts cannot establish end-to-end
durability performance. Detailed phase wall/CPU measurements are retained
in the JSON evidence, including validation, security re-evaluation and stamps.

## Actual remote-Qdrant scheduler measurement

`benchmarks/maintenance_scan.py` used a disposable Qdrant 1.16.2 container
under the same lock. Each revision got a newly created collection with the
same deterministic fixture and no warmup:

- 1,000 recent idle sessions (ineligible).
- 70 due idle sessions.
- 40 legacy consolidating sessions with missing timestamps.

Both setup times are reported separately: base 0.480 s, candidate 0.518 s;
each setup performs one collection creation and one fixture upsert.
Candidate index creation remains **inside** its first-pass timer.
The server process is shared sequentially; OS/server caches are not forcibly
flushed, so timing is illustrative. Request and row bounds are the primary
evidence.

| Measurement | Previous full pass | Candidate pass |
|---|---:|---:|
| Selection scrolls | 5 | 2 |
| Summary rows returned | 1,110 | At most 48: 32 due + 16 diagnostic |
| Due rows returned | 70 | At most 32 |
| Ineligible rows returned | 1,040 | At most 16 in diagnostic page; none in due page |
| Additional hydration requests | 1,110 collection-list calls | At most 32 retrieves + 32 collection-list calls |
| Index requests | 0 | 5 on first pass, 0 thereafter |
| Diagnosis writes | 0 | Observed 0–4 per pass; bounded by 16 |
| Measured Python peak bytes | 11,705,631 | At most 1,234,602 |
| Wall seconds | 4.556 | Cold 3.230; steady 0.040–1.172, median 0.417 |

The candidate's observed steady total request count was 14–68 per pass,
including hydration and diagnosis. Its first pass made 71 requests.
The theoretical steady bound for this fixture without append events is
82 requests (2 scrolls, 64 hydration calls, 16 diagnosis writes).
This is not a bound on append-event pages in a real individual session.
The baseline scans append links for every session before idle filtering.

All 70 due sessions are visited by pass 3; all 1,110 diagnostic rows by pass
70, including the 40 missing-timestamp legacy records. The eligible-ID digest
matches between revisions. Cursors wrap without hiding ineligible data.
The benchmark deliberately does not consume due work, so later passes revisit
it; it is not a throughput comparison of completed consolidations.

Row counts describe records returned to and examined by Python, not internal
Qdrant index-node visits. The candidate count excludes duplicate full payloads
returned by hydration retrieves; those retrieves are separately counted.
Source/result consolidation writes, arbitrary append-event histories,
crash/restart fairness, and sustained infinite-arrival workloads are not
performance-certified by this experiment.

## Reproduction and evidence

Use the same candidate benchmark scripts with baseline and candidate
`PYTHONPATH`, the same locked environment, and separate processes.
The measurements describe the historical baseline/candidate comparison, not
qualification of the current release. Runtime behavior was unchanged during
that measurement follow-up.

Save evidence in a local directory of your choice outside the public repository.
Historical evidence is not bundled; regenerate these outputs when reproducing:

- `repeat-{base,candidate}-{1,2}.json`: full phase and request measurements.
- `repeat-summary.json`: distributions and scheduler bounds.
- `scan-{base,candidate}.json`: actual remote request counts and every pass.
- `fsck-{baseline,candidate}.json`: original pair, retained unchanged.
- `fencing.log`: target-fencing failure, unchanged and still blocking takeover.
