"""LOCAL-PATCH tests 2026-11-04: owner/subject scoping of core memories.

Failure class measured on the live store: MemoryService.get_core_memories issued
five storage reads without the owner/subject scoping that list, search and
browse apply. _build_owner_scope_condition(user_id, None) degrades to a bare
user_id match, so rows whose owner_id belongs to a different identity were
formatted into the requesting user's prompt - while every enumeration endpoint
filtered them out by owner, leaving the operator no way to see or delete them.
About 166 such rows in this store; 9 were still being injected after a full
drain of everything the API would show.
"""
import inspect
import types

from mnemory.api.memories import browse_memories
from mnemory.memory import MemoryService
from mnemory.storage.vector import VectorStore, _is_scope_mismatch

ME = "chairilw@hotmail.com"
OTHER = "someone-else@example.com"


class _Cache:
    def get(self, key):
        return None

    def set(self, key, value):
        return None

    def invalidate_prefix(self, prefix):
        return None


class _RecordingVector:
    """Records every read so the scoping of each call can be asserted."""

    def __init__(self):
        self.calls = []

    def _record(self, name, kwargs):
        self.calls.append((name, kwargs))
        return []

    def get_pinned_memories(self, **kwargs):
        return self._record("pinned", kwargs)

    def get_all(self, **kwargs):
        self._record("get_all", kwargs)
        return {"results": []}

    def get_recent_memories(self, **kwargs):
        return self._record("recent", kwargs)


def _svc():
    from mnemory.config import Config

    svc = object.__new__(MemoryService)
    svc._config = Config()
    svc._core_cache = _Cache()
    svc.vector = _RecordingVector()
    return svc


class TestCoreScoping:
    def test_every_core_read_is_owner_and_subject_scoped(self):
        """The leak was structural: reads that omit owner_id silently widen scope."""
        svc = _svc()
        svc.get_core_memories(user_id=ME, owner_id=ME, agent_id="open-webui",
                             track_access=False)
        vec = svc.vector
        assert vec.calls, "core builder performed no storage reads"
        for name, kw in vec.calls:
            assert kw.get("owner_id") == ME, "%s called without owner scope" % name
            assert kw.get("subject_user_id") == ME, "%s called without subject scope" % name

    def test_pinned_read_carries_the_scoping(self):
        svc = _svc()
        svc.get_core_memories(user_id=ME, owner_id=ME, agent_id="open-webui",
                             track_access=False)
        pinned = [kw for name, kw in svc.vector.calls if name == "pinned"]
        assert len(pinned) == 1
        assert pinned[0]["owner_id"] == ME
        assert pinned[0]["subject_user_id"] == ME

    def test_scoping_survives_a_shared_agent_call(self):
        """Grantee path already scoped correctly; keep it that way."""
        svc = _svc()
        svc.get_core_memories(user_id=OTHER, owner_id=ME, agent_id="open-webui",
                             track_access=False)
        for name, kw in svc.vector.calls:
            assert kw.get("owner_id") == ME, "%s lost the owner dimension" % name


class TestScopeMismatchHelper:
    def test_owner_matching_subject_is_clean(self):
        assert _is_scope_mismatch({"owner_id": ME, "user_id": ME}, ME) is False

    def test_owner_differing_from_subject_is_reported(self):
        assert _is_scope_mismatch({"owner_id": OTHER, "user_id": ME}, ME) is True

    def test_legacy_rows_without_owner_are_not_reported(self):
        assert _is_scope_mismatch({"user_id": ME}, ME) is False

    def test_nested_metadata_form_is_understood(self):
        row = {"metadata": {"owner_id": OTHER, "user_id": ME}}
        assert _is_scope_mismatch(row, ME) is True


class _Point:
    def __init__(self, pid, payload):
        self.id = pid
        self.payload = payload


class _FakeClient:
    """Minimal stand-in for the qdrant client's offset-cursor scroll."""

    def __init__(self, points):
        self._points = points

    def scroll(self, **kw):
        off = kw.get("offset") or 0
        lim = kw.get("limit", 50)
        chunk = self._points[off:off + lim]
        nxt = off + lim if off + lim < len(self._points) else None
        return chunk, nxt


def _store(points):
    store = object.__new__(VectorStore)
    store._client = _FakeClient(points)
    # collection_name is a read-only property over config.vector, so the stub
    # has to satisfy the config rather than assign the attribute.
    store._config = types.SimpleNamespace(
        vector=types.SimpleNamespace(collection_name="test"))
    return store


POINTS = [
    _Point("own", {"data": "mine", "user_id": ME, "owner_id": ME}),
    _Point("stray", {"data": "about me, owned elsewhere", "user_id": ME, "owner_id": OTHER}),
    _Point("legacy", {"data": "predates owner dimension", "user_id": ME}),
]


class TestBrowseScopeMismatch:
    def test_mismatch_view_returns_only_the_stray_rows(self):
        res = _store(list(POINTS)).browse_active(
            user_id=ME, owner_id=ME, session_agent_id=None,
            scope_mismatch=True, include_decayed=True, limit=50)
        assert [r["id"] for r in res["results"]] == ["stray"]

    def test_normal_browse_still_shows_everything(self):
        res = _store(list(POINTS)).browse_active(
            user_id=ME, owner_id=ME, session_agent_id=None,
            include_decayed=True, limit=50)
        assert {r["id"] for r in res["results"]} == {"own", "stray", "legacy"}

    def test_mismatch_view_forces_storage_order(self):
        """The predicate is applied post-scroll, so it cannot ride the
        creation-ordered path; a newest request must not silently ignore it."""
        seen = _store(list(POINTS))
        seen._browse_by_creation = lambda **kw: (_ for _ in ()).throw(
            AssertionError("creation-ordered path used for scope_mismatch"))
        res = seen.browse_active(
            user_id=ME, owner_id=ME, session_agent_id=None,
            scope_mismatch=True, include_decayed=True, sort="newest", limit=50)
        assert [r["id"] for r in res["results"]] == ["stray"]

    def test_api_param_is_wired(self):
        params = inspect.signature(browse_memories).parameters
        assert "scope_mismatch" in params
        target = inspect.signature(VectorStore.browse_active).parameters
        assert "scope_mismatch" in target
