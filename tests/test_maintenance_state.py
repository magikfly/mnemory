from datetime import datetime, timedelta, timezone

from mnemory.maintenance_state import diagnose_operation


def test_cancelled_await_does_not_report_worker_completion():
    import asyncio
    import threading
    from types import SimpleNamespace
    from unittest.mock import Mock

    from mnemory.maintenance import MaintenanceService

    entered, release = threading.Event(), threading.Event()
    fsck = Mock()

    def run(*args, **kwargs):
        entered.set()
        assert release.wait(10)

    fsck.run_check.side_effect = run
    fsck.get_check.return_value = SimpleNamespace(status="completed")
    collector = Mock()
    service = MaintenanceService(Mock(), fsck, collector)

    async def scenario():
        task = asyncio.create_task(asyncio.to_thread(service._run_fsck_worker, "check"))
        await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert collector.record_maintenance_lifecycle.call_count == 1
        release.set()

    try:
        asyncio.run(scenario())
    finally:
        release.set()
    calls = collector.record_maintenance_lifecycle.call_args_list
    assert calls[0].args == ("fsck", "started")
    assert calls[1].args[:2] == ("fsck", "completed")


def test_age_never_authorizes_recovery():
    now = datetime.now(timezone.utc)
    for status in ("planned", "applying", "failed"):
        record = {"status": status, "updated_at_utc": "2000-01-01T00:00:00Z"}
        assert diagnose_operation(record, now) == "replay_blocked"
        assert record["status"] == status


def test_live_expired_unknown_and_exhausted_ownership():
    now = datetime.now(timezone.utc)
    record = {"status": "applying", "recovery_token": "worker"}
    assert diagnose_operation(record, now) == "ownership_unknown"
    record["lease_expires_at"] = (now + timedelta(seconds=10)).isoformat()
    assert diagnose_operation(record, now) == "active_lease"
    record["lease_expires_at"] = (now - timedelta(seconds=10)).isoformat()
    assert diagnose_operation(record, now) == "replay_blocked"
    record["recovery_attempt_count"] = 3
    assert diagnose_operation(record, now) == "retry_exhausted"
    assert record["status"] == "applying"


def test_future_protocol_is_not_interpreted():
    assert (
        diagnose_operation(
            {"maintenance_protocol": "future"}, datetime.now(timezone.utc)
        )
        == "incompatible_protocol"
    )
