"""Disposable remote-Qdrant cross-collection fencing counterexample.

Run only against a disposable server: python -m benchmarks.maintenance_fencing URL
No application collections or data are used.
"""

import json
import sys
import threading
import uuid

from qdrant_client import QdrantClient, models


def demonstrate(url: str) -> dict:
    client = QdrantClient(url=url)
    worker = QdrantClient(url=url)
    prefix = f"maintenance_fencing_{uuid.uuid4().hex}"
    journal, targets = f"{prefix}_journal", f"{prefix}_targets"
    point_id = str(uuid.uuid4())
    checked, resume = threading.Event(), threading.Event()
    errors = []
    try:
        for collection in (journal, targets):
            client.create_collection(
                collection,
                vectors_config=models.VectorParams(
                    size=1, distance=models.Distance.DOT
                ),
            )
            client.upsert(
                collection,
                [models.PointStruct(id=point_id, vector=[1.0], payload={"owner": "A"})],
                wait=True,
                ordering=models.WriteOrdering.STRONG,
            )

        def stale_worker():
            try:
                assert worker.retrieve(journal, [point_id])[0].payload["owner"] == "A"
                checked.set()
                assert resume.wait(10)
                # Target CAS does not consult the journal collection.
                worker.set_payload(
                    targets,
                    {"written_by": "A"},
                    models.Filter(
                        must=[
                            models.HasIdCondition(has_id=[point_id]),
                            models.FieldCondition(
                                key="owner", match=models.MatchValue(value="A")
                            ),
                        ]
                    ),
                    wait=True,
                    ordering=models.WriteOrdering.STRONG,
                )
            except BaseException as exc:
                errors.append(type(exc).__name__)
                checked.set()

        thread = threading.Thread(target=stale_worker)
        thread.start()
        assert checked.wait(10)
        client.set_payload(
            journal,
            {"owner": "B"},
            [point_id],
            wait=True,
            ordering=models.WriteOrdering.STRONG,
        )
        assert client.retrieve(journal, [point_id])[0].payload["owner"] == "B"
        resume.set()
        thread.join(15)
        assert not thread.is_alive() and not errors, errors
        stale_write = client.retrieve(targets, [point_id])[0].payload.get("written_by")
        assert stale_write == "A"
        return {
            "gate_passed": False,
            "stale_target_write_after_journal_takeover": True,
            "ordering": "strong",
            "server_version": client.info().version,
        }
    finally:
        resume.set()
        for collection in (journal, targets):
            if client.collection_exists(collection):
                client.delete_collection(collection)
        worker.close()
        client.close()


if __name__ == "__main__":
    print(json.dumps(demonstrate(sys.argv[1])))
