"""Tests for the guard rules added alongside the extraction chokepoint.

Sanitized: the mnemory repository is public, so every fixture here is invented.
The real-world rows that motivated these rules (a "Fact N" meta cluster, a
"Today's date is Wednesday" row stored as type=fact, and a 'Speaker: "quote"'
fragment) are described in commit text, not reproduced here.

Covers the three new noise_memory() reason codes and pin_allowed(), which is
the part that changes behaviour on every write: the extraction schema makes
`pinned` a required field, so without an allowlist the model assigns permanent
pins to ordinary facts.
"""

from __future__ import annotations

from mnemory.output_guards import noise_memory, pin_allowed

# --- meta indexical: reference into a numbered list that will not survive ----
META_INDEXICAL = [
    "Fact 3 describes a specific row that exists in the raw layer but not in "
    "the consolidated layer, unlike Fact 9.",
    "Item 7 identifies a data corruption issue in the sample database.",
    "Point 2 proposes reclassification targets for the current run.",
]

# --- date echo: the session clock stored as durable knowledge ---------------
DATE_ECHO = [
    "Today's date is Wednesday, 7 October 2026.",
    "The user indicated today's date is 2026-08-24 and used it as the run "
    "date.",
]

# Deliberately NOT flagged, and a guard against widening the date rule: a date
# statement can be the fact itself.
DATE_NOT_ECHO = [
    "The date is August 13, her birthday.",
    "User's birthday is August 13, 1975.",
]

# --- speaker quote: a bare utterance with no proposition --------------------
SPEAKER_QUOTE = [
    'Assistant: "That settles it, lock in 512"',
    'User: "sounds good to me"',
]

# --- must never be flagged --------------------------------------------------
KEEP = [
    "User's name is Ada Lovelace",
    "User is a solutions architect at Example Corp",
    "User lives in Amsterdam, Netherlands.",
    "User takes metformin 1000 mg twice daily",
    "User switched from Sublime to Neovim last month",
    "Caroline was promoted to senior engineer at Example Corp",
    "The service was restarted at 09:14 and the queue drained",
    "The store grows by 40-60 rows per day under normal chat traffic",
    "Assistant identified the root cause and patched the migration path that "
    "dropped revision rows",
    "Upstream released v9.9.9 which deleted the apply_settings function, "
    "failing the CI merge step",
]

PIN_WARRANTED = [
    "User's name is Ada Lovelace",
    "Assistant's name is Atlas",
    "User was born in Paramaribo",
    "User's birthday is August 13",
    "User takes metformin, semaglutide and empagliflozin",
    "User is allergic to penicillin",
    "User lives in Amsterdam, Netherlands.",
    "User is a solutions architect at Example Corp",
    "User works at Example Corp",
]

NOT_PIN_WARRANTED = [
    "The Planck constant quantifies energy per photon via E equals h nu",
    "User's resting heart rate was 75 on Tuesday and 61 on Monday",
    "The bearer token appears in 22 files and needs rotation",
    "Assistant verified both monitoring agents are up on version 2.45.2",
    "Upstream released v9.9.9 which deleted the apply_settings function",
    "User requested a summary of the quarterly results",
]


def test_meta_indexical_rejected():
    for text in META_INDEXICAL:
        assert noise_memory(text) == "meta_indexical", text


def test_date_echo_rejected():
    for text in DATE_ECHO:
        assert noise_memory(text) == "date_echo", text


def test_date_statements_that_are_the_fact_survive():
    for text in DATE_NOT_ECHO:
        assert noise_memory(text) is None, text


def test_speaker_quote_rejected():
    for text in SPEAKER_QUOTE:
        assert noise_memory(text) == "speaker_quote", text


def test_substantive_rows_survive():
    for text in KEEP:
        assert noise_memory(text) is None, text


def test_pin_allowlist_grants_identity_only():
    for text in PIN_WARRANTED:
        assert pin_allowed(text) is True, text


def test_pin_allowlist_withholds_everything_else():
    for text in NOT_PIN_WARRANTED:
        assert pin_allowed(text) is False, text


def test_pin_allowed_is_total():
    assert pin_allowed("") is False
    assert pin_allowed(None) is False
