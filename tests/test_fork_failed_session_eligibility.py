"""Fork guard: a session stamped ``failed`` must stay eligible for consolidation.

Why this file exists
--------------------
Upstream v1.15.1 replaced ``scan_consolidation_candidates`` (the path this fork
patched) with the new paged ``SessionSummaryStore.maintenance_page``, whose
due-work filter selected only ``"idle"``. Meanwhile:

  * ``consolidation.py`` still stamps ``consolidation_state = "failed"`` on an
    unexpected error (5 sites),
  * ``update_consolidation_state`` never re-arms that state and never stamps
    ``maintenance_next_at``, and
  * ``diagnose_maintenance`` deliberately "never changes lifecycle state".

So under the new scan a failed session would become permanently invisible —
exactly the silent consolidation stall this fork fixed on 2026-10-05
(deploy/DEPLOYED.md §1): notes stop consolidating and nothing reports it.
``claim_consolidation`` already accepts ``["idle", "failed"]`` (also a fork
patch), so the only missing piece was offering the row.

These tests fail if a future merge drops either half of that fix.
"""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from qdrant_client import QdrantClient, models

from mnemory.storage.vector import SessionSummaryStore


def _store_with(state: str, *, revision: int = 1, stale: bool = True):
    """One synthetic session in `state`, idle for hours, no ownership tokens."""
    client = QdrantClient(":memory:")
    store = SessionSummaryStore(client)
    client.create_collection(
        store.COLLECTION,
        vectors_config=models.VectorParams(size=1, distance=models.Distance.DOT),
    )
    point_id = str(uuid4())
    now = datetime.now(timezone.utc)
    payload = {
        "session_id": point_id,
        "user_id": "synthetic",
        "consolidation_state": state,
        "session_revision": revision,
        "attempt_count": 0,
        "memory_ids": ["m-1", "m-2"],
        "updated_at": (now - timedelta(hours=5 if stale else 0)).isoformat(),
    }
    client.upsert(
        store.COLLECTION,
        [models.PointStruct(id=point_id, vector=[1.0], payload=payload)],
    )
    return client, store, point_id


def test_failed_session_is_offered_as_due_work():
    now = datetime.now(timezone.utc)
    _, store, point_id = _store_with("failed")

    page, _ = store.maintenance_page(cutoff=now - timedelta(hours=1), now=now)

    assert point_id in {item["_point_id"] for item in page}, (
        "a 'failed' session is never re-tried: maintenance_page regressed to "
        "upstream's idle-only due filter (see this module's docstring)"
    )


def test_failed_session_is_visible_to_the_diagnostic_sweep():
    now = datetime.now(timezone.utc)
    _, store, point_id = _store_with("failed")

    page, _ = store.maintenance_page(
        cutoff=now - timedelta(hours=1), now=now, diagnostic=True
    )

    assert point_id in {item["_point_id"] for item in page}


def test_due_filter_still_respects_the_idle_cutoff():
    """The fork patch must not turn every failed row into hot work."""
    now = datetime.now(timezone.utc)
    _, store, point_id = _store_with("failed", stale=False)

    page, _ = store.maintenance_page(cutoff=now - timedelta(hours=1), now=now)

    assert point_id not in {item["_point_id"] for item in page}


def test_failed_session_can_be_claimed():
    """The other half of the fix: claiming a failed revision is allowed."""
    client, store, point_id = _store_with("failed")

    claimed = store.claim_consolidation(
        point_id,
        expected_revision=1,
        token="tok-1",
        attempt_count=1,
        point_id=point_id,
    )

    assert claimed is True
    stored = client.retrieve(store.COLLECTION, ids=[point_id], with_payload=True)
    assert stored[0].payload["consolidation_state"] == "consolidating"
    assert stored[0].payload["consolidation_token"] == "tok-1"


def test_consolidation_gate_agrees_with_the_store_filter():
    """maintenance.py's post-hydration gate must allow 'failed' too.

    Source-level on purpose: the runtime path needs a full MaintenanceService,
    and the failure mode being guarded is a merge silently taking upstream's
    `!= "idle"` line back.
    """
    import inspect

    from mnemory.maintenance import MaintenanceService

    src = inspect.getsource(MaintenanceService._run_consolidation)
    assert '"idle", "failed"' in src, (
        "_run_consolidation drops hydrated 'failed' sessions again"
    )
