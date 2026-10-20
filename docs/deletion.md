# Deletion and best-effort erasure

Normal `delete_memory` retracts a memory and retains immutable history and
artifacts. Existing online `privacy_erase=true` and the REST privacy endpoint
attempt physical cleanup of the selected lineage and unshared artifacts.
They remain available without a maintenance-only requirement.

Known permission, storage and cleanup errors are reported, not converted into
success. A successful result means the cleanup work discovered by this operation
completed. It does **not** prove that content can never reappear.

Qdrant, filesystem/S3 artifacts, session stores and operation records do not form
one transaction. Concurrent jobs, cached work, recovery and new writes may retain
or recreate content. Session summaries and derived outputs can retain information
without complete lineage provenance. Backups, snapshots, historical S3 versions,
provider logs and external systems are outside the online cleanup scope.
This is not verified erasure, a forensic wipe or a compliance guarantee.

For edge cases requiring stronger assurance, an operator should stop all writers,
recovery workers and queued ingestion producers before deliberate cleanup, inspect
dependent session/replay state, and verify the intended stores before restarting
fresh processes. Do not resume stale cached work or blindly delete unrelated
session content. Remote writer exclusion must be verified operationally; no flag
or online result establishes it. Actual production cleanup needs separate
authorization and is not a release smoke test.

Repeated finalization of an existing erase plan is supported. Once its lineage
and audit records are gone, a new request can return not found; no personal
tombstone is retained solely to turn every later retry into success.
