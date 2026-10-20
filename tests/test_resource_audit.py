"""Resource optimizations preserve storage, scope, and scheduling semantics."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient, models

from mnemory.cache import TTLCache
from mnemory.storage.vector import VectorStore
from tests.test_fsck import _make_fsck_service, _make_memory
from tests.test_metrics import _make_collector


def test_cache_cold_key_churn_and_replacement(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("mnemory.cache.time.monotonic", lambda: clock[0])
    cache = TTLCache(ttl_seconds=10)
    for i in range(10000):
        cache.set(i, i)
    clock[0] = 5
    cache.set(0, "replacement")
    clock[0] = 11
    assert cache.get("absent") is None
    # Expiry is deliberately bounded per access; continued activity drains it.
    for _ in range(625):
        cache.get("absent")
    assert list(cache._cache) == [0]
    assert cache.get(0) == "replacement"
    clock[0] = 15
    assert cache.get(0) == "replacement"  # Existing strict > expiry boundary.
    clock[0] = 16
    cache.set("new", 1)
    assert list(cache._cache) == ["new"]
    disabled = TTLCache(ttl_seconds=0)
    disabled.set("key", "value")
    assert not disabled._cache


def test_cache_concurrent_mutation_and_prefix_invalidation():
    cache = TTLCache(ttl_seconds=60)

    def churn(worker):
        for i in range(500):
            cache.set((str(worker), i), i)
            cache.get((str(worker), i))
            cache.invalidate_prefix(str(worker))
            cache.invalidate((str(worker), i))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(churn, range(8)))
    assert not cache._cache


@pytest.fixture
def vector():
    client = QdrantClient(":memory:")
    client.create_collection(
        "audit",
        vectors_config={"": models.VectorParams(size=2, distance=models.Distance.DOT)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )
    store = VectorStore.__new__(VectorStore)
    store._client = client
    store._config = SimpleNamespace(vector=SimpleNamespace(collection_name="audit"))
    yield store
    client.close()


def insert(vector, score, **metadata):
    mid = str(uuid4())
    payload = {"user_id": "alice", "data": "fact", **metadata}
    vector._client.upsert(
        "audit",
        [
            models.PointStruct(
                id=mid,
                vector={
                    "": [score, 1.0],
                    "sparse": models.SparseVector(indices=[1], values=[1.0]),
                },
                payload=payload,
            )
        ],
    )
    return mid


def test_metadata_scan_and_dense_retrieval_preserve_scope(vector):
    kept = insert(vector, 1.0, agent_id="agent")
    insert(vector, 1.0, agent_id="other")
    insert(vector, 1.0, agent_id="agent", memory_layer="raw")
    insert(vector, 1.0, agent_id="agent", revision_state="superseded")
    insert(vector, 1.0, agent_id="agent", user_id="bob")
    insert(vector, 1.0, agent_id="agent", expires_at="2000-01-01T00:00:00Z")
    kwargs = dict(
        user_id="alice", agent_id="agent", exclude_layers=["raw"], exclude_expired=True
    )
    old = vector.scroll_with_vectors(**kwargs)
    new = vector.scroll_with_vectors(**kwargs, include_vectors=False)
    assert [m["id"] for m in new] == [kept]
    assert all("vector" not in m for m in new)
    assert new == [{k: v for k, v in m.items() if k != "vector"} for m in old]
    assert vector.get_dense_vectors([kept]) == {kept: old[0]["vector"]}


def test_threshold_keeps_all_ties_and_more_than_small_topk(vector):
    ids = {insert(vector, 0.75) for _ in range(40)}
    insert(vector, 0.7499)
    insert(vector, 0.99, user_id="bob")
    old = vector.search_by_vector_ids([1.0, 0.0], user_id="alice", limit=100)
    new = vector.search_by_vector_ids(
        [1.0, 0.0], user_id="alice", limit=100, score_threshold=0.75
    )
    assert new == [m for m in old if m["score"] >= 0.75]
    assert {m["id"] for m in new} == ids


def test_legacy_unnamed_dense_vectors(vector):
    client = vector._client
    client.create_collection(
        "legacy",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    mid = str(uuid4())
    client.upsert(
        "legacy",
        [
            models.PointStruct(
                id=mid,
                vector=[1.0, 0.0],
                payload={"user_id": "alice", "data": "legacy"},
            )
        ],
    )
    vector._config.vector.collection_name = "legacy"
    assert vector.scroll_with_vectors(user_id="alice")[0]["vector"] == [1.0, 0.0]
    assert vector.get_dense_vectors([mid]) == {mid: [1.0, 0.0]}


def test_missing_dense_vectors_and_threshold_margin_do_not_add_edges(vector):
    source = insert(vector, 1.0, agent_id="agent")
    tie = insert(vector, 0.75)
    below = insert(vector, 0.75 - 5e-8)
    missing = insert(vector, 1.0, agent_id="agent")
    sparse_only = insert(vector, 1.0, agent_id="agent")
    vector._client.update_vectors(
        "audit", [models.PointVectors(id=source, vector={"": [1.0, 0.0]})]
    )
    corpus = vector.scroll_with_vectors(user_id="alice", include_vectors=False)
    # Deletion between metadata selection and batch retrieval must be harmless.
    vector._client.delete("audit", points_selector=[missing])
    vector._client.delete_vectors("audit", vectors=[""], points=[sparse_only])
    hits = vector.search_by_vector_ids(
        [1.0, 0.0],
        user_id="alice",
        shared_only=True,
        score_threshold=0.75 - 1e-7,
        limit=100,
    )
    assert {hit["id"] for hit in hits} == {tie, below}
    assert next(hit["score"] for hit in hits if hit["id"] == below) < 0.75
    service = _make_fsck_service()
    service._vector = vector
    service._evaluate_duplicate_cluster = MagicMock(return_value=[])
    check = service._store.create(user_id="alice", agent_id="agent")
    working = [m for m in corpus if m["id"] in {source, missing, sparse_only}]
    _, clustered = service._phase_duplicate_detection(
        working,
        {m["id"]: m for m in working},
        check,
        full_corpus_by_id={m["id"]: m for m in corpus},
    )
    assert clustered == {source, tie}
    assert check.progress.percent == 55


def test_duplicate_clusters_match_eager_with_unchanged_shared_neighbors(vector):
    changed = insert(vector, 1.0, agent_id="agent")
    shared = insert(vector, 1.0)
    insert(vector, 1.0, memory_layer="raw")
    insert(vector, 1.0, agent_id="other")
    insert(vector, 1.0, expires_at="2000-01-01T00:00:00Z")
    outcomes = []
    for eager in (True, False):
        corpus = vector.scroll_with_vectors(
            user_id="alice",
            agent_id="agent",
            include_vectors=eager,
            exclude_layers=["raw"],
            exclude_expired=True,
        ) + vector.scroll_with_vectors(
            user_id="alice",
            shared_only=True,
            include_vectors=eager,
            exclude_layers=["raw"],
            exclude_expired=True,
        )
        service = _make_fsck_service()
        service._vector = vector
        service._evaluate_duplicate_cluster = MagicMock(return_value=[])
        check = service._store.create(user_id="alice", agent_id="agent")
        working = [m for m in corpus if m["id"] == changed]
        issues, clustered = service._phase_duplicate_detection(
            working,
            {m["id"]: m for m in working},
            check,
            full_corpus_by_id={m["id"]: m for m in corpus},
        )
        outcomes.append((issues, clustered))
        assert clustered == {changed, shared}
        evaluated = service._evaluate_duplicate_cluster.call_args.args[0]
        assert {m["id"] for m in evaluated} == {changed, shared}
    assert outcomes[0] == outcomes[1]


def test_fsck_dense_batches_do_not_attach_vectors_to_corpus():
    memories = [_make_memory(mid=f"m{i}") for i in range(600)]
    for memory in memories:
        memory.pop("vector")
    service = _make_fsck_service(memories=memories)
    batches = []

    def retrieve(ids):
        batches.append(len(ids))
        return {mid: [1.0, 0.0] for mid in ids}

    service._vector.get_dense_vectors.side_effect = retrieve
    check = service._store.create(user_id="alice")
    service._phase_duplicate_detection(memories, {m["id"]: m for m in memories}, check)
    assert batches == [256, 256, 88]
    assert all("vector" not in m for m in memories)
    assert all(
        c.kwargs["score_threshold"] == 0.75 - 1e-7
        for c in service._vector.search_by_vector_ids.call_args_list
    )


def test_fsck_incremental_cap_fetches_only_selected_vectors():
    memories = [_make_memory(mid=f"m{i}") for i in range(100)]
    for i, memory in enumerate(memories):
        memory.pop("vector")
        if i < 90:
            memory["metadata"]["checked_at"] = "2025-01-01"
            memory["metadata"]["updated_at_utc"] = "2024-01-01"
    service = _make_fsck_service(memories=memories)
    service._config.memory.fsck_max_memories = 3
    service._vector.get_dense_vectors.side_effect = lambda ids: {
        mid: [1.0] for mid in ids
    }
    check = service._store.create(user_id="alice")
    service.run_check(check.check_id, incremental=True)
    assert check.status == "completed"
    assert check.progress.truncated
    selected = service._vector.get_dense_vectors.call_args.args[0]
    assert len(selected) == 3
    assert all(int(mid[1:]) >= 90 for mid in selected)
    assert (
        service._vector.scroll_with_vectors.call_args.kwargs["include_vectors"] is False
    )


def test_metrics_filters_with_real_qdrant(vector):
    for state in ["active", "superseded", "pending", None, ""]:
        insert(vector, 1.0, revision_state=state)
    insert(vector, 1.0)  # Legacy missing revision state is active.
    client = vector._client
    client.create_collection("_mnemory_operations", vectors_config={})
    client.upsert(
        "_mnemory_operations",
        [
            models.PointStruct(
                id=str(uuid4()),
                vector={},
                payload={"operation_kind": "fsck", "status": status},
            )
            for status in ["planned", "applying", "failed", "completed"]
        ],
    )
    collector = _make_collector()
    collector._vector_store = vector
    collector._refresh_gauges_from_qdrant()
    assert sum(m._value.get() for m in collector._memories_total._metrics.values()) == 2
    assert collector._fsck_operations_nonterminal._value.get() == 3


def test_session_listing_does_not_block_event_loop(monkeypatch):
    from mnemory.api import sessions

    started = threading.Event()
    release = threading.Event()
    expected = {"sessions": [], "total": 0}

    def listing(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return expected

    store = SimpleNamespace(list_for_user=listing)
    monkeypatch.setattr(
        sessions, "_get_service", lambda: SimpleNamespace(_session_summary_store=store)
    )

    async def run():
        task = asyncio.create_task(
            sessions.list_sessions(
                offset=0,
                limit=50,
                consolidation_state=None,
                q=None,
                sort_by="updated_at",
                sort_dir="desc",
                ctx=SimpleNamespace(user_id="alice", owner_id="alice", agent_id=None),
            )
        )
        try:
            for _ in range(100):
                if started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert started.is_set()
            assert not task.done()
        finally:
            release.set()
        assert await task == expected

    asyncio.run(run())
