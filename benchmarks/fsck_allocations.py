"""Synthetic decoded-vector allocation benchmark (no external backends).

Run: uv run python -m benchmarks.fsck_allocations --corpus 10000 --selected 512
Measures Python allocations at the storage response boundary, not Qdrant RSS.
"""

import argparse
import gc
import json
import tracemalloc
from types import SimpleNamespace

from mnemory.storage.vector import VectorStore


class SyntheticClient:
    """Generate independent float objects like a decoded remote response."""

    def __init__(self, count: int, dimensions: int):
        self.count = count
        self.dimensions = dimensions
        self.transferred_vectors = 0

    def point(self, index: int, vectors: bool) -> SimpleNamespace:
        self.transferred_vectors += int(vectors)
        return SimpleNamespace(
            id=str(index),
            payload={"data": "Synthetic fact", "user_id": "alice"},
            vector={"": [float(i) for i in range(self.dimensions)]}
            if vectors
            else None,
        )

    def scroll(self, *, offset, limit, with_vectors, **kwargs):
        start = offset or 0
        end = min(start + limit, self.count)
        return (
            [self.point(i, bool(with_vectors)) for i in range(start, end)],
            end if end < self.count else None,
        )

    def retrieve(self, *, ids, **kwargs):
        return [self.point(int(mid), True) for mid in ids]


def measure(count: int, selected: int, dimensions: int, eager: bool) -> dict:
    client = SyntheticClient(count, dimensions)
    store = VectorStore.__new__(VectorStore)
    store._client = client
    store._config = SimpleNamespace(vector=SimpleNamespace(collection_name="synthetic"))
    gc.collect()
    tracemalloc.start()
    corpus = store.scroll_with_vectors(user_id="alice", include_vectors=eager)
    candidates = corpus[:selected]
    if not eager:
        for offset in range(0, len(candidates), 256):
            batch = store.get_dense_vectors(
                [m["id"] for m in candidates[offset : offset + 256]]
            )
            assert all(len(vector) == dimensions for vector in batch.values())
            del batch
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "mode": "eager" if eager else "bounded",
        "corpus": count,
        "selected": selected,
        "dimensions": dimensions,
        "peak_bytes": peak,
        "transferred_vectors": client.transferred_vectors,
        "peak_retained_vectors": count if eager else min(256, selected, count),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=int, default=10000)
    parser.add_argument("--selected", type=int, default=512)
    parser.add_argument("--dimensions", type=int, default=1536)
    args = parser.parse_args()
    for eager in (True, False):
        print(json.dumps(measure(args.corpus, args.selected, args.dimensions, eager)))
