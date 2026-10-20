"""Content-free maintenance diagnosis. Diagnosis never authorizes replay."""

from datetime import datetime, timezone
from typing import Any

MAINTENANCE_PROTOCOL = "mnemory.maintenance.v1"
DIAGNOSES = frozenset(
    {
        "active_lease",
        "ownership_unknown",
        "replay_blocked",
        "incompatible_protocol",
        "malformed_checkpoint",
        "retry_exhausted",
        "terminal",
        "idle",
        "invalid_timestamp",
        "empty_inputs",
    }
)


def parse_timestamp(value: Any) -> datetime | None:
    """Return an aware timestamp, or None for missing/invalid lifecycle data."""
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
    except (TypeError, ValueError):
        return None


def diagnose_operation(record: dict, now: datetime, max_attempts: int = 3) -> str:
    """Classify metadata without treating age as permission to resume."""
    if record.get("status") in {"committed", "skipped", "superseded"}:
        return "terminal"
    if record.get("maintenance_protocol") not in {None, MAINTENANCE_PROTOCOL}:
        return "incompatible_protocol"
    deadline = parse_timestamp(record.get("lease_expires_at"))
    if record.get("recovery_token"):
        if deadline is None:
            return "ownership_unknown"
        if deadline > now:
            return "active_lease"
    attempts = record.get("recovery_attempt_count", 0)
    if not isinstance(attempts, int) or attempts < 0:
        return "malformed_checkpoint"
    if attempts >= max_attempts:
        return "retry_exhausted"
    # Expiry does not fence already-running writes in another collection.
    return "replay_blocked"
