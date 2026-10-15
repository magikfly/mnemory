"""Tests for the write-side noise guard.

Content is synthetic on purpose: this repository is public, and the shapes
below are transcriptions of the *form* of rows observed in a private store on
2026-10-15, not their text. MUST_KEEP matters more than SHOULD_DROP: a false
positive silently destroys a real fact, and there is no un-retract endpoint.
"""

import pytest

from mnemory.output_guards import noise_enforcing, noise_memory

SHOULD_DROP = [
    "User is asking who the assistant is",
    "User was asking about the log output",
    "User inquired about their identity; the assistant answered",
    "User wondered whether the sync had finished",
    "User asked how proficient they are in a legacy language.",
    "User asked how to make a small model respect the prompt language.",
    "Assistant explained that small models rely on balanced examples.",
    "Assistant recommended using a helper tool for this.",
    "Assistant offered to adapt the template for another use.",
    "Assistant noted the presence of a second unrelated entry.",
    "Assistant will dump the tokenizer keys on request.",
    "Assistant reported the upstream maintainer's statement.",
    "User wants to have a specific name stored for the assistant.",
    "User's real name is stored as a pinned fact at row 12ab34cd.",
    "The store contains 408 rows classified as delete_safe.",
    "User executed a purge: 442 rows retracted, 104 stripped.",
    "Dashboard cleaned up diagnostic points to reduce bucket noise",
    "Memory store housekeeping left no findings open",
    "- Assistant explained that the flag defaults to off",
    "Assistant identified four retrieval misses where queries did not contain",
    "Assistant identified that 3 remaining misses are not ranking problems",
    "Assistant noted the pinned row still claims the old filter version",
    "User initiated a greeting with 'hey there'",
    "User greeted the assistant in Dutch",
]

MUST_KEEP = [
    "User's maximum heart rate is 188 BPM.",
    "User has Type 2 Diabetes and is on insulin or a sulfonylurea.",
    "User does not wear his watch during weightlifting sessions",
    "User's store grows at a rate of 40-60 rows per day.",
    "The fsck dedup call site echoes cluster content and needs its own ceiling.",
    "Assistant's name is Atlas.",
    "Assistant fixed the prefill indicator by anchoring the spinner.",
    "Assistant identified the root cause and patched the migration path",
    "User set hard limits: never reboot or restart containers.",
    "User considers a zero-load day without a watch as not real rest.",
    "User is available for freelance work at a minimum rate of EUR 110.",
    "User decided to keep the old config file for rollback readability.",
    "User lives in a suburb of a large city in the Netherlands",
    "User merged the upstream release tag without conflicts",
    "Assistant identified the root cause and patched the migration path",
    "Assistant identified cross-role leak bug where claims were laundered",
    "User's memory of the run includes pace and heart rate per kilometre",
    "User stores the vault file at mode 0600 with git history",
]


@pytest.mark.parametrize("text", SHOULD_DROP)
def test_noise_is_flagged(text: str) -> None:
    assert noise_memory(text) is not None, f"expected a reason code for {text!r}"


@pytest.mark.parametrize("text", MUST_KEEP)
def test_real_facts_survive(text: str) -> None:
    assert noise_memory(text) is None, f"false positive, would drop {text!r}"


def test_reason_codes_are_stable() -> None:
    assert noise_memory("User is asking who the assistant is") == "question_in_progress"
    assert noise_memory("Assistant offered to adapt it") == "assistant_speech_act"
    assert (noise_memory("User wants to have a specific name stored for it")
            == "pointer_not_fact")
    assert noise_memory("Store holds 408 rows classified as delete_safe.") == "store_housekeeping"


def test_empty_and_none_are_not_noise() -> None:
    assert noise_memory(None) is None
    assert noise_memory("") is None
    assert noise_memory("- ") == "empty_after_strip"


def test_default_mode_is_log_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMORY_NOISE_GUARD", raising=False)
    assert noise_enforcing() is False, "dropping by default could eat facts silently"
    monkeypatch.setenv("MEMORY_NOISE_GUARD", "drop")
    assert noise_enforcing() is True
    monkeypatch.setenv("MEMORY_NOISE_GUARD", "LOG")
    assert noise_enforcing() is False
