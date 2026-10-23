"""Deterministic contract for the durable log file (mnemory/logsink.py).

Why this exists: guard verdicts are the only evidence of an irreversible write
decision, and stderr ties them to one container's lifetime. Two audit corpora
were lost to recreates before the behaviour was pinned here. A claim about a
file gets a test about a file -- no service, no qdrant, no model.

These deliberately drive the real root logger, because that is what server.py
attaches to and what mnemory.memory logs through; a private logger would prove
the helper works in a shape production never uses.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from logging.handlers import RotatingFileHandler

import pytest

from mnemory.logsink import LOG_FILE_ENV, attach_file_handler


@pytest.fixture
def root() -> Iterator[logging.Logger]:
    """The root logger, with only this test's handlers removed afterwards.

    pytest's logging plugin keeps its own capture handlers on root, so every
    assertion below counts RotatingFileHandler instances rather than handlers,
    and teardown closes only what was added during the test.
    """
    log = logging.getLogger()
    original = list(log.handlers)
    previous_level = log.level
    log.setLevel(logging.INFO)
    yield log
    for handler in list(log.handlers):
        if handler not in original:
            handler.close()
            log.removeHandler(handler)
    for handler in original:
        if handler not in log.handlers:
            log.addHandler(handler)
    log.setLevel(previous_level)


def _file_handlers(log: logging.Logger) -> list[RotatingFileHandler]:
    return [h for h in log.handlers if isinstance(h, RotatingFileHandler)]


def test_unset_is_a_noop(monkeypatch: pytest.MonkeyPatch, root: logging.Logger) -> None:
    monkeypatch.delenv(LOG_FILE_ENV, raising=False)
    assert attach_file_handler() is None
    assert _file_handlers(root) == [], "unset env must not attach anything"


def test_blank_is_a_noop(monkeypatch: pytest.MonkeyPatch, root: logging.Logger) -> None:
    monkeypatch.setenv(LOG_FILE_ENV, "   ")
    assert attach_file_handler() is None
    assert _file_handlers(root) == []


def test_verdict_line_reaches_the_file(
    monkeypatch: pytest.MonkeyPatch, root: logging.Logger, tmp_path
) -> None:
    target = tmp_path / "logs" / "mnemory.log"  # parent does not exist yet
    monkeypatch.setenv(LOG_FILE_ENV, str(target))
    handler = attach_file_handler()
    assert isinstance(handler, RotatingFileHandler)
    assert target.exists(), "the log dir must be created, not assumed"

    # The real call site and the real message shape, from mnemory/memory.py.
    logging.getLogger("mnemory.memory").info(
        "Remember noise guard: dropped (date_echo): Today\u2019s date is Wednesday, 7 October 2026."
    )
    handler.flush()
    text = target.read_text(encoding="utf-8")
    assert "date_echo" in text, f"verdict missing from durable log: {text!r}"
    # Same shape as stderr, so existing grep habits keep working.
    assert " INFO mnemory.memory: " in text, text


def test_second_call_does_not_duplicate_lines(
    monkeypatch: pytest.MonkeyPatch, root: logging.Logger, tmp_path
) -> None:
    target = tmp_path / "mnemory.log"
    monkeypatch.setenv(LOG_FILE_ENV, str(target))
    first = attach_file_handler()
    second = attach_file_handler()
    assert second is first, "re-attach must return the existing handler"
    assert len(_file_handlers(root)) == 1, "a duplicated handler double-writes evidence"

    logging.getLogger("mnemory.memory").info("durability-marker")
    first.flush()
    assert target.read_text(encoding="utf-8").count("durability-marker") == 1


def test_unusable_path_degrades_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch, root: logging.Logger, tmp_path
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv(LOG_FILE_ENV, str(blocker / "sub" / "mnemory.log"))
    assert attach_file_handler() is None, "a bad log path must not crash the service"
    assert _file_handlers(root) == []
