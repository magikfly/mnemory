"""Bounded expiry, exact TTL semantics and destructor/locking invariants."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import mnemory.cache as module
from mnemory.cache import _EXPIRY_BUDGET, TTLCache


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    return now


@pytest.mark.parametrize("count", [20, 1000, 10000, 100000])
def test_bound_requested_expiry_and_progress(clock, count):
    cache = TTLCache(10)
    for i in range(count):
        cache.set(i, i)
    clock[0] = 9
    cache.set("live", False)
    clock[0] = 11
    assert cache.get(count - 1) is None
    assert len(cache._cache) == count - _EXPIRY_BUDGET
    assert cache.get("live") is False
    remaining = sum(isinstance(k, int) for k in cache._cache)
    for _ in range((remaining + _EXPIRY_BUDGET - 1) // _EXPIRY_BUDGET):
        assert cache.get("absent") is None
    assert list(cache._cache) == ["live"]
    clock[0] = 19
    assert cache.get("live") is False
    clock[0] = 19.00001
    assert cache.get("live") is None


def test_steady_churn_backlog_contract(clock):
    cache = TTLCache(10)
    for i in range(1000):
        cache.set(i, i)
    clock[0] = 11
    for i in range(1000, 1100):
        before = len(cache._cache)
        expired = sum(timestamp == 0 for _, timestamp in cache._cache.values())
        cache.set(i, i)
        assert len(cache._cache) == before - min(expired, _EXPIRY_BUDGET) + 1
    assert len(cache._cache) == 100
    # One unique write per second: <= 11 live entries, no growth with runtime.
    for i in range(1100, 3100):
        clock[0] += 1
        cache.set(i, i)
    assert len(cache._cache) == 11


@pytest.mark.parametrize("action", ["get", "set", "invalidate", "prefix", "clear"])
def test_destructors_run_unlocked_and_can_reenter(clock, action):
    cache = TTLCache(10)
    destroyed = []

    class Value:
        def __del__(self):
            acquired = cache._lock.acquire(blocking=False)
            destroyed.append(acquired)
            if acquired:
                cache._lock.release()
                cache.get("missing")

    cache.set("key", Value())
    if action == "get":
        clock[0] = 11
        cache.get("missing")
    elif action == "set":
        cache.set("key", 2)
    elif action == "invalidate":
        cache.invalidate("key")
    elif action == "prefix":
        cache.invalidate_prefix("key")
    else:
        cache.clear()
    assert destroyed == [True]


def test_slow_destructor_does_not_block_valid_hit(clock):
    cache = TTLCache(10)
    entered, release = threading.Event(), threading.Event()

    class Slow:
        def __del__(self):
            entered.set()
            release.wait(5)

    cache.set("old", Slow())
    clock[0] = 9
    cache.set("live", 0)
    clock[0] = 11
    with ThreadPoolExecutor(2) as pool:
        sweep = pool.submit(cache.get, "absent")
        try:
            assert entered.wait(2)
            assert pool.submit(cache.get, "live").result(timeout=1) == 0
        finally:
            release.set()
        assert sweep.result(timeout=2) is None


def test_bounded_destructor_count_without_reentrancy(clock):
    cache = TTLCache(10)
    destroyed = []

    class Value:
        def __del__(self):
            destroyed.append(1)

    for i in range(10000):
        cache.set(i, Value())
    clock[0] = 11
    cache.get("absent")
    assert len(destroyed) == _EXPIRY_BUDGET
    cache.get(9999)
    assert len(destroyed) == 2 * _EXPIRY_BUDGET + 1


def test_serialized_clock_reads_and_overwrite_order(monkeypatch):
    cache = TTLCache(10)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def now():
        assert cache._lock.locked()
        calls.append(len(calls))
        if len(calls) == 1:
            entered.set()
            assert release.wait(2)
        return float(len(calls))

    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=now))
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(cache.set, "a", 1)
        assert entered.wait(1)
        second = pool.submit(cache.set, "b", 2)
        assert calls == [0]
        release.set()
        first.result(timeout=2)
        second.result(timeout=2)
    cache.set("a", 3)
    assert list(cache._cache) == ["b", "a"]
    assert [entry[1] for entry in cache._cache.values()] == [2.0, 3.0]
    cache.invalidate("a")
    assert cache.get("a") is None
    cache.set("a", 0)
    assert cache.get("a") == 0


@pytest.mark.parametrize("ttl", [0, -1])
def test_disabled_never_stores(clock, ttl):
    cache = TTLCache(ttl)
    cache.set("a", False)
    assert not cache._cache
    assert cache.get("a") is None
