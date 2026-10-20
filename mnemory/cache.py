"""Generic TTL-based in-memory cache.

Used by MemoryService for caching category lists (auto-classification)
and core memory responses (get_core_memories).
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")

# Each write adds at most one entry; reclaiming up to 16 gives backlog progress
# under steady churn without making foreground work proportional to cache size.
_EXPIRY_BUDGET = 16


class TTLCache(Generic[K, V]):
    """Simple TTL-based in-memory cache.

    Entries expire after ``ttl_seconds``. Expired entries are evicted
    lazily on access. Setting ``ttl_seconds=0`` effectively disables
    caching (every ``get`` returns ``None``).
    """

    def __init__(self, ttl_seconds: int = 300):
        self._ttl = ttl_seconds
        self._cache: OrderedDict[K, tuple[V, float]] = OrderedDict()
        self._lock = threading.Lock()

    def _expire(self, now: float, discarded: list) -> None:
        # Writes are timestamp ordered, including replacements. Each expired
        # entry is visited once, rather than scanning every live key per write.
        for _ in range(_EXPIRY_BUDGET):
            if not self._cache:
                break
            timestamp = next(iter(self._cache.values()))[1]
            if now - timestamp <= self._ttl:
                break
            # Keep both key and value alive until the caller releases the lock.
            discarded.append(self._cache.popitem(last=False))

    def get(self, key: K) -> V | None:
        """Return cached value or ``None`` if missing/expired."""
        discarded: list = []
        with self._lock:
            now = time.monotonic()
            self._expire(now, discarded)
            entry = self._cache.get(key)
            # A requested expired key may be beyond the cleanup budget.
            if entry is not None and now - entry[1] > self._ttl:
                discarded.append(self._cache.pop(key))
                entry = None
            return entry[0] if entry is not None else None

    def set(self, key: K, value: V) -> None:
        """Store a value with the current timestamp."""
        discarded: list = []
        with self._lock:
            now = time.monotonic()
            self._expire(now, discarded)
            discarded.append(self._cache.pop(key, None))
            if self._ttl > 0:
                self._cache[key] = (value, now)

    def invalidate(self, key: K) -> None:
        """Remove a single entry."""
        discarded: list = []
        with self._lock:
            self._expire(time.monotonic(), discarded)
            discarded.append(self._cache.pop(key, None))

    def invalidate_prefix(self, prefix: str) -> None:
        """Remove all entries whose key starts with *prefix*.

        Works for string keys and tuple keys (matches the first element).
        This is useful for invalidating all cache entries for a given
        ``user_id`` regardless of other key components.
        """
        discarded: list = []
        with self._lock:
            self._expire(time.monotonic(), discarded)
            to_delete = []
            for k in self._cache:
                if isinstance(k, str) and k.startswith(prefix):
                    to_delete.append(k)
                elif isinstance(k, tuple) and k and str(k[0]) == prefix:
                    to_delete.append(k)
            for k in to_delete:
                discarded.append(self._cache.pop(k))

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            discarded = self._cache
            self._cache = OrderedDict()
        # Explicit bulk clear is unbounded, but never destroys values under lock.
        discarded.clear()
