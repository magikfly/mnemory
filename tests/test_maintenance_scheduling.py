from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from uuid import uuid4

from qdrant_client import QdrantClient, models

from mnemory.storage.vector import SessionSummaryStore


def test_bounded_keyset_fairness_and_legacy_visibility():
    client = QdrantClient(":memory:")
    store = SessionSummaryStore(client)
    client.create_collection(
        store.COLLECTION,
        vectors_config=models.VectorParams(size=1, distance=models.Distance.DOT),
    )
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=1)
    eligible = set()
    legacy = str(uuid4())
    points = []
    for index in range(1071):
        point_id = str(uuid4()) if index != 1070 else legacy
        payload = {
            "session_id": point_id,
            "user_id": "synthetic",
            "consolidation_state": "idle",
            "attempt_count": 7,
            "updated_at": (
                now if index < 1000 else cutoff - timedelta(hours=1)
            ).isoformat(),
        }
        if 1000 <= index < 1070:
            eligible.add(point_id)
        if index == 1070:
            payload.pop("updated_at")
            payload["consolidation_state"] = "consolidating"
        points.append(models.PointStruct(id=point_id, vector=[1.0], payload=payload))
    client.upsert(store.COLLECTION, points)
    seen = set()
    cursor = None
    with patch.object(client, "scroll", wraps=client.scroll) as scroll:
        for _ in range(3):
            page, cursor = store.maintenance_page(cutoff=cutoff, now=now, offset=cursor)
            assert len(page) <= 32
            assert all(item["attempt_count"] == 7 for item in page)
            seen.update(item["_point_id"] for item in page)
        assert cursor is None
        assert seen == eligible
        assert scroll.call_count == 3
        assert all(
            call.kwargs["with_vectors"] is False for call in scroll.call_args_list
        )
    cursor = None
    audited = set()
    for _ in range(67):
        page, cursor = store.maintenance_page(
            cutoff=cutoff, now=now, offset=cursor, diagnostic=True, limit=16
        )
        assert len(page) <= 16
        audited.update(item["_point_id"] for item in page)
    assert legacy in audited
    assert cursor is None
    client.close()


def test_diagnosis_is_generation_conditional_and_does_not_terminalize():
    client = QdrantClient(":memory:")
    store = SessionSummaryStore(client)
    client.create_collection(
        store.COLLECTION,
        vectors_config=models.VectorParams(size=1, distance=models.Distance.DOT),
    )
    point_id = str(uuid4())
    record = {
        "_point_id": point_id,
        "session_revision": 2,
        "consolidation_state": "consolidating",
        "updated_at": "2025-01-01T00:00:00Z",
    }
    client.upsert(
        store.COLLECTION,
        [models.PointStruct(id=point_id, vector=[1.0], payload=record)],
    )
    store.diagnose_maintenance({**record, "session_revision": 1}, "ownership_unknown")
    assert (
        "maintenance_diagnosis"
        not in client.retrieve(store.COLLECTION, [point_id])[0].payload
    )
    store.diagnose_maintenance(record, "ownership_unknown")
    result = client.retrieve(store.COLLECTION, [point_id])[0].payload
    assert result["maintenance_diagnosis"] == "ownership_unknown"
    assert result["consolidation_state"] == "consolidating"
    client.close()
