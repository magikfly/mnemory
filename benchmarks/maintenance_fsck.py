"""Full fsck synthetic pipeline benchmark, no external models or real memories.

Run in separate processes against baseline and candidate PYTHONPATHs.
The backend models decoded remote responses, not Qdrant search performance.
"""

import argparse
import hashlib
import inspect
import json
import resource
import time
import tracemalloc
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

from mnemory.fsck import FsckService, FsckStore
from mnemory.storage.vector import VectorStore


class Client:
    def __init__(self, count, dimensions):
        self.count, self.dimensions = count, dimensions
        self.pages = self.vectors = self.max_batch = self.searches = 0
        self.retrieves = self.writes = 0
        self.stamped = set()

    def point(self, index, vectors=False):
        self.vectors += int(vectors)
        return SimpleNamespace(
            id=str(index),
            payload={
                "data": f"Synthetic equipment fact {index // 2}",
                "user_id": "synthetic",
                "memory_type": "fact",
                "importance": "normal",
                "categories": [],
            },
            vector=(
                [float(index)] + [float(i) / 1536 for i in range(self.dimensions - 1)]
            )
            if vectors
            else None,
        )

    def scroll(self, *, offset, limit, with_vectors, **kwargs):
        self.pages += 1
        start = int(offset or 0)
        end = min(start + limit, self.count)
        return (
            [self.point(i, bool(with_vectors)) for i in range(start, end)],
            end if end < self.count else None,
        )

    def retrieve(self, *, ids, **kwargs):
        self.retrieves += 1
        self.max_batch = max(self.max_batch, len(ids))
        return [self.point(int(mid), True) for mid in ids]

    def stamp(self, mid, metadata):
        self.writes += 1
        self.stamped.add(mid)

    def search(self, vector, **kwargs):
        self.searches += 1
        index = int(vector[0])
        # Exact threshold duplicates plus below-threshold nonduplicates.
        peer = index ^ 1
        return (
            [{"id": str(peer), "score": 0.75}]
            if index % 10 < 2 and peer < self.count
            else []
        )


def run(count, dimensions):
    client = Client(count, dimensions)
    vector = VectorStore.__new__(VectorStore)
    vector._client = client
    vector._config = SimpleNamespace(
        vector=SimpleNamespace(collection_name="synthetic")
    )
    vector.search_by_vector_ids = client.search
    vector.update_metadata = client.stamp
    vector.get_categories = lambda **kwargs: []
    config = SimpleNamespace(
        memory=SimpleNamespace(
            max_memory_length=1000,
            fsck_reasoning_effort=None,
            fsck_max_memories=0,
            fsck_max_llm_calls=0,
            fsck_llm_concurrency=2,
        ),
    )
    llm = Mock()

    def generate(*args, operation, **kwargs):
        if operation == "fsck_dedup":
            return json.dumps(
                {
                    "issues": [
                        {
                            "type": "duplicate",
                            "severity": "medium",
                            "reasoning": "synthetic duplicate",
                            "affected_memory_ids": ["0", "1"],
                            "confidence": 1.0,
                            "actions": [],
                        }
                    ]
                }
            )
        return '{"issues": []}'

    llm.generate.side_effect = generate
    service = FsckService(config, vector, llm, FsckStore())
    phases = {}
    phase_names = (
        "_phase_validation_projection",
        "_phase_security_scan_regex",
        "_phase_security_reeval",
        "_phase_duplicate_detection",
        "_phase_quality_check",
        "_stamp_clean_memories",
    )

    def instrument(name, method):
        def measured(*args, **kwargs):
            wall, cpu = time.perf_counter(), time.process_time()
            try:
                return method(*args, **kwargs)
            finally:
                phases[name] = {
                    "wall_seconds": time.perf_counter() - wall,
                    "cpu_seconds": time.process_time() - cpu,
                }

        return measured

    code = "".join(
        inspect.getsource(getattr(service, name))
        for name in (
            "run_check",
            *phase_names,
        )
    )
    for name in phase_names:
        setattr(service, name, instrument(name, getattr(service, name)))
    vector.scroll_with_vectors = instrument(
        "metadata_scroll", vector.scroll_with_vectors
    )
    check = service.start_check(user_id="synthetic")
    tracemalloc.start()
    started = time.perf_counter()
    cpu_started = time.process_time()
    service.run_check(check.check_id)
    cpu_elapsed = time.process_time() - cpu_started
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert check.status == "completed", check
    issues = []
    for issue in check.issues:
        item = asdict(issue)
        item.pop("issue_id")
        item["affected_memories"].sort(key=lambda value: value["id"])
        issues.append(item)
    output = {
        "issues": sorted(issues, key=lambda item: json.dumps(item, sort_keys=True)),
        "summary": asdict(check.summary),
        "stamped": sorted(client.stamped),
        "llm_calls": check.llm_calls,
    }
    return {
        "corpus": count,
        "dimensions": dimensions,
        "runtime_seconds": elapsed,
        "process_cpu_seconds": cpu_elapsed,
        "phases": phases,
        "phase_source_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "backend": "synthetic decoded responses; no Qdrant server",
        "warmup": "none; fresh process and fixture for each run",
        "synthetic_requests": {
            "scroll": client.pages,
            "retrieve": client.retrieves,
            "search": client.searches,
            "write": client.writes,
            "index": 0,
        },
        "maintenance_scan_requests": 0,
        "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "python_peak_bytes": peak,
        "scroll_pages": client.pages,
        "vector_count": client.vectors,
        "max_vector_batch": client.max_batch,
        "neighbor_queries": client.searches,
        "issue_count": len(issues),
        "output_sha256": hashlib.sha256(
            json.dumps(output, sort_keys=True).encode()
        ).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=int, default=4096)
    parser.add_argument("--dimensions", type=int, default=1536)
    args = parser.parse_args()
    print(json.dumps(run(args.corpus, args.dimensions), sort_keys=True))
