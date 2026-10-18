"""Structural invariant: no content guard may assume ASCII-only punctuation.

Why a test instead of a fix: the first version of these guards matched an ASCII
apostrophe, while the extractor writes U+2019 into its facts as a matter of
course (so do em dashes and curly quotes throughout the production store).
ASCII-only unit fixtures passed, CI passed, and the bug reached a published
image -- where one of the two failures was in the destructive direction,
withholding a legitimate identity pin from "User's name is ...".

Sweeping the module for that one instance would have left the next one in place.
This test makes the whole class fail loudly at the point of definition, for
every pattern in the module, including patterns added later.

Outcome of the sweep, measured rather than assumed: 20 patterns, one offender
(_ASSISTANT_ACT_RE), and fixing it changed **zero** verdicts -- that rule needs
"assistant is/was/has ... will/shall <verb>" or a bare "<verb> that", so it
never matched the shape in question with either apostrophe. The sweep found no
second exploitable bug. It stays because the class of defect is real (it did
ship once, in _PIN_ALLOW_RE and _DATE_ECHO_RE) and because the negative control
below proves the sweep can see one.
"""

from __future__ import annotations

import re

from mnemory import output_guards as og

#: ASCII character -> the typographic equivalents the extractor emits.
#:
#: The right single quote is an apostrophe AND a closing single quote; the
#: left form covers quoted material. Double quotes pair the same way.
PAIRS: dict[str, tuple[str, ...]] = {
    "'": ("\u2019", "\u2018"),   # ' -> ’ ‘
    '"': ("\u201c", "\u201d"),   # " -> “ ”
}

#: Hyphen is deliberately NOT paired with en/em dash.  Tried first: it produced
#: five of six reported offenders and every one was a false positive, because
#: the hyphens sit inside identifiers ("glp-1", "n-gpu-layers", a boundary-tag
#: character class) where "glp\u2014 1" is not a thing the extractor writes.  A rule
#: that needs an exemption list per pattern is encoding the wrong rule.
EXEMPT: dict[str, set[str]] = {}


def _patterns() -> list[tuple[str, re.Pattern[str]]]:
    out = []
    for name in dir(og):
        value = getattr(og, name)
        if isinstance(value, re.Pattern):
            out.append((name, value))
    return out


def test_module_exposes_patterns():
    """Guard against the sweep silently checking nothing after a rename."""
    assert len(_patterns()) >= 10, (
        f"only {len(_patterns())} compiled patterns found in output_guards; "
        "if the module was refactored, update this test rather than lowering "
        "the floor"
    )


def test_every_guard_accepts_typographic_punctuation():
    offenders = []
    for name, pattern in _patterns():
        text = pattern.pattern
        exempt = EXEMPT.get(name, set())
        for ascii_char, equivalents in PAIRS.items():
            if ascii_char not in text or ascii_char in exempt:
                continue
            # Acceptable if the class/pattern also carries at least one
            # typographic form, or escapes the char so it is not prose.
            if any(eq in text or f"\\u{ord(eq):04x}" in text for eq in equivalents):
                continue
            offenders.append(f"{name}: ASCII {ascii_char!r} with no typographic twin")
    assert not offenders, "guard patterns will miss curly-punctuation rows:\n  " + \
        "\n  ".join(offenders)


def test_sweep_actually_detects_an_offender():
    """Negative control.

    A sweep that can never fail is decoration. This plants a known-bad pattern
    on the module and asserts the same scan reports it, so the invariant above
    is proven to have teeth rather than merely to be passing.
    """
    planted = re.compile("^user" + chr(39) + "s favourite colour")
    setattr(og, "_PLANTED_OFFENDER_RE", planted)
    try:
        offenders = []
        for name, pattern in _patterns():
            text = pattern.pattern
            for ascii_char, equivalents in PAIRS.items():
                if ascii_char in text or ascii_char in EXEMPT.get(name, set()):
                    continue
                if any(eq in text or f"\\u{ord(eq):04x}" in text
                       for eq in equivalents):
                    continue
                offenders.append(name)
        assert "_PLANTED_OFFENDER_RE" in offenders, (
            "the punctuation sweep did not flag an ASCII-only pattern it was "
            "given; it is not actually checking anything"
        )
    finally:
        delattr(og, "_PLANTED_OFFENDER_RE")


def test_apostrophe_forms_behave_identically():
    """The behavioural version of the invariant, on both punctuation forms."""
    ascii_rows = [
        ("Today's date is Wednesday, 7 October 2026.", "date_echo"),
        ("User's name is Ada Lovelace", None),
    ]
    curly_rows = [
        ("Today\u2019s date is Wednesday, 7 October 2026.", "date_echo"),
        ("User\u2019s name is Ada Lovelace", None),
    ]
    for (a_text, a_want), (c_text, c_want) in zip(ascii_rows, curly_rows):
        assert og.noise_memory(a_text) == og.noise_memory(c_text) == a_want
        assert a_want == c_want
    # ...and the pin direction must agree too, since that is the side that
    # damages a row by withholding something.
    for text in ("User's name is Ada Lovelace", "User\u2019s name is Ada Lovelace"):
        assert og.pin_allowed(text) is True, text
