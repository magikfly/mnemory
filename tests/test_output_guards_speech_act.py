"""Regression tests for the anchored narration rule and rewrite-not-drop.

Every string in MUST_KEEP / MUST_REWRITE / MUST_DROP was taken from the live
``would drop`` log of 2026-10-18 (http://192.168.2.1:8050, 24h window, 47
assistant_speech_act hits) or is a deliberate near-neighbour of one. The corpus
is real output, not invented examples -- invented examples test the regex you
already had in mind.

The NEGATIVE CONTROL at the bottom matters more than the assertions above it:
it re-creates the pre-fix pattern and requires it to still flag the ordinary
facts, proving these assertions can fail. A guard test that passes against both
the fixed and the broken rule is decoration.
"""

import re

import pytest

from mnemory.output_guards import (MIN_REWRITE_CLAUSE, _ASSISTANT_ACT_RE,
                                   noise_memory, triage_memory)

_APOS = r"['\u2019]"

# ── ordinary facts that the unanchored rule flagged as narration ────────────
MUST_KEEP = [
    "User's wife Lisandra agreed to the move to Zoetermeer",
    "User's note about knee pain says it started in August",
    "User keeps a training log and reports it weekly to his coach",
    "The knee offer of service is expired, replace the brace",
    "User explained the irrigation wiring to the contractor on site",
    "User's maximum heart rate is 188 BPM",
    # engineering findings about the store are this user's actual work; the
    # guard must not eat them
    "Assistant fixed the anchor precedence in output_guards and shipped it",
    "User's memory filter drops metadata blocks before embedding",
]

# ── narration wrapping a real statement: unwrap, do not delete ──────────────
MUST_REWRITE = {
    "Assistant recommended keeping Thursday 21/10 as a rest day":
        "Advice: Keeping Thursday 21/10 as a rest day",
    "Assistant advised drinking water with salt today instead of plain water":
        "Advice: Drinking water with salt today instead of plain water",
    "Assistant recommended not running on an empty stomach":
        "Advice: Not running on an empty stomach",
    "Assistant suggested short steps of 160-170 per minute on the hills":
        "Advice: Short steps of 160-170 per minute on the hills",
    "Assistant told the user to take carbohydrate before runs over 75 minutes":
        "Advice: Take carbohydrate before runs over 75 minutes",
    # explicit that-clause: kept verbatim, no label invented
    "Assistant noted that icu_apply_sport_settings will overwrite zones":
        "icu_apply_sport_settings will overwrite zones",
    "Assistant explained that the consolidator runs every 300 seconds":
        "The consolidator runs every 300 seconds",
}

# ── the act IS the content: still refuse ───────────────────────────────────
MUST_DROP = [
    "Assistant offered to unpin memories 28, 35, 45 via PUT",
    "Assistant will dump the tokenizer keys later",
    "Assistant agreed to the plan",
    "Assistant declined to speculate further",
    "Assistant mentioned it in passing during the review",
]


@pytest.mark.parametrize("text", MUST_KEEP)
def test_ordinary_facts_survive(text):
    stored, reason, rewrote = triage_memory(text)
    assert reason is None, f"{text!r} flagged as {reason}"
    assert not rewrote
    assert stored == text


@pytest.mark.parametrize("text,expected", sorted(MUST_REWRITE.items()))
def test_substantive_advice_is_unwrapped_not_dropped(text, expected):
    stored, reason, rewrote = triage_memory(text)
    assert reason is None, f"{text!r} still dropped as {reason}"
    assert rewrote
    assert stored == expected


@pytest.mark.parametrize("text", MUST_DROP)
def test_pure_speech_acts_still_refused(text):
    stored, reason, rewrote = triage_memory(text)
    assert reason == "assistant_speech_act", f"{text!r} -> {reason}"
    assert not rewrote
    assert stored == text


def test_rewrite_is_idempotent():
    """A rewritten clause must not be re-flagged on its next pass."""
    for text in MUST_REWRITE:
        once, reason, _ = triage_memory(text)
        assert reason is None
        twice, reason2, rewrote2 = triage_memory(once)
        assert reason2 is None and not rewrote2, (once, reason2)
        assert twice == once


def test_no_identifier_is_reworded_by_capitalisation():
    """Version strings and code names come through byte-identical."""
    for text in MUST_REWRITE:
        stored, _, _ = triage_memory(text)
        for token in ("icu_apply_sport_settings",):
            if token in text:
                assert token in stored, (text, stored)


def test_short_clause_is_not_rescued():
    """'Assistant explained more' has no clause to keep -- stay a drop."""
    stored, reason, rewrote = triage_memory("Assistant explained more")
    assert not rewrote and reason == "assistant_speech_act"
    assert MIN_REWRITE_CLAUSE >= 12


def test_typographic_apostrophe_still_matches():
    """The extractor writes U+2019; the head must accept it."""
    assert noise_memory("Assistant\u2019s recommended resting Thursday") in (
        "assistant_speech_act", None)
    assert _ASSISTANT_ACT_RE.search(
        "Assistant\u2019s offered to rerun the sweep")


# ── NEGATIVE CONTROL ───────────────────────────────────────────────────────
# The pre-fix pattern, verbatim. It MUST still flag the ordinary facts above:
# that is the bug. If this assertion ever stops holding, the control is no
# longer reproducing anything and the tests above lose their meaning.
_OLD_BROKEN_RE = re.compile(
    r"^\s*assistant(?:" + _APOS + r"s)?\s+"
    r"(?:is\s+|was\s+|has\s+|have\s+|will\s+|can\s+|"
    r"would\s+)*"
    r"(?:will|shall)\s+[a-z]+|"
    r"(?:offer(?:s|ed)?|explain(?:s|ed)?|not(?:e|es|ed)|"
    r"mention(?:s|ed)?|clarif(?:y|ies|ied)|agree[sd]?|declin(?:e|es|ed)|"
    r"demonstrat(?:e|es|ed)|recommend(?:s|ed)?|report(?:s|ed)?|"
    r"stat(?:e|es|ed)\s+that|"
    r"confirm(?:s|ed)?\s+that)\b",
    re.IGNORECASE,
)


def test_negative_control_old_rule_broke_on_plain_facts():
    wrongly_flagged = [t for t in MUST_KEEP if _OLD_BROKEN_RE.search(t)]
    assert len(wrongly_flagged) >= 5, (
        "control no longer reproduces the precedence bug; the MUST_KEEP "
        "assertions above are not proving anything")

    newly_safe = [t for t in MUST_KEEP if _ASSISTANT_ACT_RE.search(t)]
    assert newly_safe == [], f"new rule still flags: {newly_safe}"
