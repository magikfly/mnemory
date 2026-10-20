"""Matched session scan experiment. Use ONLY a disposable Qdrant server.

Creates and deletes _mnemory_sessions; refuses an existing collection.
Both versions get identical fixtures and cold indexes, with setup reported
separately for both. Measures scheduler selection, not consolidation execution.
"""

import argparse
import hashlib
import json
import resource
import time
import tracemalloc
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone

from qdrant_client import QdrantClient, models

from mnemory.storage.vector import SessionSummaryStore


class CountingClient:
    def __init__(self, client):
        self.client = client
        self.calls = Counter()
        self.rows = []

    def __getattr__(self, name):
        method = getattr(self.client, name)

        def call(*args, **kwargs):
            self.calls[name] += 1
            result = method(*args, **kwargs)
            if name == "scroll":
                self.rows.extend(point.payload or {} for point in result[0])
            return result

        return call

    def reset(self):
        self.calls.clear()
        self.rows.clear()


def run(url, candidate):
    client = QdrantClient(url=url)
    collection = SessionSummaryStore.COLLECTION
    if client.collection_exists(collection):
        raise RuntimeError("Refusing to touch an existing session collection")
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    cutoff = now - timedelta(hours=1)
    eligible = set()
    setup = time.perf_counter()
    created = False
    try:
        client.create_collection(
            collection,
            vectors_config=models.VectorParams(size=1, distance=models.Distance.DOT),
        )
        created = True
        points = []
        for index in range(1110):
            sid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"synthetic-session-{index}"))
            # 1,000 recent/ineligible idle rows, 70 due idle, 40 legacy owned.
            state = "idle" if index < 1070 else "consolidating"
            if 1000 <= index < 1070:
                eligible.add(sid)
            payload = {
                "session_id": sid,
                "user_id": "synthetic",
                "owner_id": "synthetic",
                "consolidation_state": state,
                "updated_at": (
                    now if index < 1000 else now - timedelta(days=1)
                ).isoformat(),
                "session_revision": 1,
                "attempt_count": 2,
                "memory_ids": ["synthetic-input"],
            }
            if index >= 1070:
                # Deliberately missing timestamp: legacy sweep must not hide it.
                payload.pop("updated_at")
                payload["consolidation_token"] = "synthetic-owner"
            points.append(
                models.PointStruct(
                    id=SessionSummaryStore.get_point_id(
                        sid,
                        user_id="synthetic",
                        owner_id="synthetic",
                    ),
                    vector=[1.0],
                    payload=payload,
                )
            )
        client.upsert(collection, points, wait=True)
        setup_seconds = time.perf_counter() - setup
        counted = CountingClient(client)
        store = SessionSummaryStore(counted)
        passes, seen, audited = [], set(), set()
        due_cursor = diagnostic_cursor = None
        # 70 diagnostic pages cover all 1,110 rows; work stays bounded per pass.
        for iteration in range(70 if candidate else 1):
            counted.reset()
            tracemalloc.start()
            wall, cpu = time.perf_counter(), time.process_time()
            if candidate:
                due, due_cursor = store.maintenance_page(
                    now=now,
                    cutoff=cutoff,
                    offset=due_cursor,
                    limit=32,
                )
                diagnostics, diagnostic_cursor = store.maintenance_page(
                    now=now,
                    cutoff=cutoff,
                    offset=diagnostic_cursor,
                    diagnostic=True,
                    limit=16,
                )
                audited.update(row["session_id"] for row in diagnostics)
                for row in diagnostics:
                    if row["consolidation_state"] == "consolidating":
                        store.diagnose_maintenance(row, "ownership_unknown")
                # Include the actual candidate hydration cost, not only its
                # cheap projection. Baseline scan hydrates append IDs itself.
                for row in due:
                    store.get(row["session_id"], point_id=row["_point_id"])
            else:
                rows = store.scan_consolidation_candidates()
                due = [row for row in rows if row["session_id"] in eligible]
                diagnostics = rows
                audited.update(row["session_id"] for row in rows)
            elapsed, cpu_elapsed = time.perf_counter() - wall, time.process_time() - cpu
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            seen.update(row["session_id"] for row in due)
            passes.append(
                {
                    "pass": iteration + 1,
                    "wall_seconds": elapsed,
                    "cpu_seconds": cpu_elapsed,
                    "python_peak_bytes": peak,
                    "requests": dict(counted.calls),
                    "rows_returned": len(counted.rows),
                    "due_returned": len(due),
                    "diagnostic_returned": len(diagnostics),
                    "ineligible_rows_returned": sum(
                        row.get("session_id") not in eligible for row in counted.rows
                    ),
                    "eligible_seen": len(seen),
                    "diagnostic_seen": len(audited),
                }
            )
        assert seen == eligible
        assert len(audited) == 1110
        return {
            "candidate": candidate,
            "server": client.info().version,
            "setup_seconds": setup_seconds,
            "setup_requests": {"create_collection": 1, "upsert": 1},
            "cache": "new collection; no warmup; indexes included in first pass",
            "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "eligible_sha256": hashlib.sha256(
                json.dumps(sorted(seen)).encode()
            ).hexdigest(),
            "passes": passes,
            "limitation": "Rows returned/examined by Python; not Qdrant internal index nodes examined. Includes candidate hydration and diagnosis writes, not consolidation execution.",
        }
    finally:
        if created:
            client.delete_collection(collection)
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--candidate", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.url, args.candidate), sort_keys=True))
