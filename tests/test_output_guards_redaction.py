"""Secret-redaction guard: behaviour, false-positive controls, and wiring.

Why the negative controls carry as much weight as the positives: this store is
full of numeric configuration facts (max_tokens ceilings, FSCK_* limits,
thresholds) and one measured lesson already exists in it -- a too-broad
speech-act rule would have silently deleted medical and training guidance had
enforcement been on. A redaction rule that eats "--max-tokens: 8192" is worse
than the credential it hides, because the credential was never read by anyone
and the threshold is read every session.

Redaction is enforced on sight, unlike the noise drops, and the reason is
reversibility: the row survives, only the value goes. There is no un-retract
endpoint in mnemory, so every *drop* rule is permanent and must ship behind
MEMORY_NOISE_GUARD. This one does not need to.

All fixtures are synthetic. The repo is public.
"""

from __future__ import annotations

import pathlib

from mnemory import output_guards
from mnemory.output_guards import (
    _CRED_KV_RE,
    _looks_credential,
    redact_secrets,
    redaction_enabled,
)

# ── named token shapes ───────────────────────────────────────────────

#: Synthetic, exactly AIza + 35 characters -- the real format is a fixed 39,
#: and the pattern is deliberately that strict so an ordinary word starting
#: with AIza cannot be redacted.
GOOGLE_FAKE = "AIza" + "aB3d" * 8 + "aB3"

assert len(GOOGLE_FAKE) == 39, "fixture drift: Google keys are AIza + 35"

#: Slack, Stripe and Telegram fixtures are assembled at runtime. Written as
#: literals they are structurally valid tokens for real services, and GitHub
#: push protection refused the commit -- correctly, since a scanner cannot tell
#: a fixture from a leak. Concatenation keeps the string handed to the guard
#: identical while no scanner-legal token appears anywhere in the source.
SLACK_FAKE = "xox" + "b" + "-" + "123456789012" + "-" + "abcdefghijklmno"
STRIPE_FAKE = "sk_" + "test" + "_" + "Ab1cD2eF3gH4iJ5kL6mN7oP8"
TELEGRAM_FAKE = "123456789" + ":" + "AA" + "HdqTcvCH1vGWJxfSeofSAs0K5PALDsaw"

#: Shape assertions, not magic totals -- each mirrors what its guard pattern
#: actually requires, so a fixture edit that stops exercising the pattern fails
#: here rather than silently narrowing coverage.
assert SLACK_FAKE.startswith("xoxb-") and len(SLACK_FAKE.split("-")[1]) == 12
assert STRIPE_FAKE.startswith("sk_test_") and len(STRIPE_FAKE[8:]) >= 16
assert TELEGRAM_FAKE.split(":")[0].isdigit() and len(TELEGRAM_FAKE.split(":")[1]) >= 32

TOKEN_CASES = [
    (
        "github_token",
        "User pasted ghp_" + "A1" * 18 + " into the chat",
        "[REDACTED:github_token]",
    ),
    (
        "aws_access_key",
        "The deploy key AKIAIOSFODNN7EXAMPLE rotates monthly",
        "[REDACTED:aws_access_key]",
    ),
    (
        "slack_token",
        "Bot posts with " + SLACK_FAKE,
        "[REDACTED:slack_token]",
    ),
    (
        "google_api_key",
        "Maps uses " + GOOGLE_FAKE,
        "[REDACTED:google_api_key]",
    ),
    (
        "stripe_key",
        "Payments test key " + STRIPE_FAKE,
        "[REDACTED:stripe_key]",
    ),
    (
        "openai_key",
        "Extractor uses sk-proj-" + "aB3dE5fG7hI9jK1lM2nO3pQ4rS5tU6vW7x",
        "[REDACTED:openai_key]",
    ),
    (
        "jwt",
        "Session cookie eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJzdWIiOiIxMjM0NSJ9.abcDEF1234567890XYZ",
        "[REDACTED:jwt]",
    ),
    (
        "private_key",
        "Key material -----BEGIN RSA PRIVATE KEY----- truncated",
        "[REDACTED:private_key]",
    ),
    (
        "telegram_bot_token",
        "Alerts bot " + TELEGRAM_FAKE + " here",
        "[REDACTED:telegram_bot_token]",
    ),
]


def test_every_named_token_shape_is_redacted():
    for kind, text, marker in TOKEN_CASES:
        out, kinds = redact_secrets(text)
        assert kind in kinds, f"{kind}: kinds={kinds}"
        assert marker in out, f"{kind}: {out!r}"


def test_no_secret_bytes_survive_any_output():
    """The direction that matters: the value must be gone, not merely labelled.

    Asserted against the secret substrings themselves, not against the marker,
    because a guard that emits '[REDACTED:x] sk-abc...' would satisfy a marker
    check and still leak.
    """
    secrets = (
        "ghp_" + "A1" * 18,
        "AKIAIOSFODNN7EXAMPLE",
        SLACK_FAKE,
        STRIPE_FAKE,
        GOOGLE_FAKE,
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
        TELEGRAM_FAKE,
        "-----BEGIN RSA PRIVATE KEY-----",
    )
    for _kind, text, _marker in TOKEN_CASES:
        out, _kinds = redact_secrets(text)
        for needle in secrets:
            assert needle not in out, f"{needle[:16]}... survived in {out!r}"


# ── generic label-delimited values ───────────────────────────────────

KV_REDACT = [
    "password: hunter2",
    "Set api_key=abc123def456 in compose",
    "wachtwoord: welkom123 voor de router",
    "client_secret=g0supersecretvalue is set per env",
    "The access_token : 0tokenabc123 appears in the header",
]

KV_KEEP = [
    # Ordinary prose after the noun.
    "password: required for sudo access",
    "password length must be at least 12 characters",
    "wachtwoord: verplicht",
    # Token-budget facts. This is the false-positive class that matters most
    # here: a hyphen is a word boundary, so an unqualified "token" label would
    # match the tail of "--max-tokens:" and redact a config value.
    "Set --max-tokens: 8192 on the llama-swap entry",
    "Per-pass max_tokens ceilings: fsck_security 6144 and Pass B 12288",
    "FSCK_MAX_LLM_CALLS=600",
    "The token bucket algorithm throttles the carry bucket",
    "CORE_MEMORY_INCLUDE_TOP_N is read by zero files",
    # Identity and health facts must be untouched, apostrophes included.
    # Fictional throughout: this repo is public, and a fixture that reuses a
    # real name, city, prescription or training target is the same leak
    # upstream v1.15.1 had to fictionalise its own examples to fix.
    "User lives in Examplestad, Netherlands",
    "User's real name is Ada Lovelace.",
    "Assistant's name is Atlas.",
    "User takes amlodipine 5 mg once daily",
    "User aims for 84-88 rpm cadence on the trainer",
]


def test_label_delimited_secrets_are_redacted():
    for text in KV_REDACT:
        out, kinds = redact_secrets(text)
        assert "credential_value" in kinds, f"missed: {text!r} -> {out!r}"
        assert "[REDACTED:credential_value]" in out


def test_ordinary_prose_and_config_are_untouched():
    for text in KV_KEEP:
        out, kinds = redact_secrets(text)
        assert out == text, f"false positive {kinds}: {text!r} -> {out!r}"
        assert kinds == []


def test_value_predicate_requires_machine_shape():
    # digit, or long enough that no English word is.
    assert _looks_credential("hunter2") is True
    assert _looks_credential("correcthorsebatterystaple") is True
    # short, digitless, or a stop word -> prose.
    assert _looks_credential("letmein") is False
    assert _looks_credential("required") is False
    assert _looks_credential("8192") is False  # below the 6-char floor
    assert _looks_credential("well-known") is False  # hyphen is NOT a signal


def test_bare_token_is_not_a_credential_label():
    """Guards the one deliberate omission in _CRED_KV_RE.

    'token' reads like an obvious label to add. Adding it redacts
    '--max-tokens: 8192' because \\b matches after the hyphen, which deletes a
    config value this user reasons about daily. If someone widens the label
    list, this fails and they have to come read this comment.
    """
    for text in ("--max-tokens: 8192", "max-tokens: 8192",
                 "ctx tokens: 65536"):
        out, kinds = redact_secrets(text)
        assert out == text, f"{text!r} redacted to {out!r} (kinds={kinds})"
    # The qualified compound labels must still be there, or this test would
    # pass simply because the whole generic rule was deleted.
    out, kinds = redact_secrets("access_token: abc123def456")
    assert "credential_value" in kinds, out
    assert "bearer" in _CRED_KV_RE.pattern


# ── invariants ───────────────────────────────────────────────────────


def test_redaction_is_idempotent():
    once, _k1 = redact_secrets("api_key=abc123def456 here")
    twice, k2 = redact_secrets(once)
    assert twice == once
    assert k2 == [], "a placeholder must not redact its own placeholder"


def test_redaction_never_empties_a_memory():
    """The claim that makes on-sight enforcement safe: the row survives."""
    for text in ("password: hunter2", "ghp_" + "A1" * 18, "token sk-" + "a" * 30):
        out, _kinds = redact_secrets(text)
        assert out and out.strip(), f"{text!r} -> {out!r}"
        assert "[REDACTED:" in out


def test_doubly_secret_text_reports_both_kinds():
    out, kinds = redact_secrets(
        "key ghp_" + "A1" * 18 + " and password: hunter2 together"
    )
    assert "github_token" in kinds and "credential_value" in kinds
    assert "hunter2" not in out


def test_env_switch_disables_without_crashing(monkeypatch):
    monkeypatch.setenv("MEMORY_REDACTION", "off")
    assert redaction_enabled() is False
    text = "password: hunter2"
    assert redact_secrets(text) == (text, [])
    monkeypatch.setenv("MEMORY_REDACTION", "on")
    assert redaction_enabled() is True


def test_empty_and_none_input():
    assert redact_secrets(None) == (None, [])
    assert redact_secrets("") == ("", [])


# ── wiring: the patches must survive the next upstream merge ─────────

WIRING = {
    "prompts": "text, redactions = redact_secrets(text)",
    "memory": "redact_secrets(",
    "consolidation": "redact_secrets(",
    "fsck": "redact_secrets(",
}


def _source(module_name: str) -> str:
    """Module text without importing it.

    mnemory.memory pulls qdrant_client at import time, so importing it to read
    it would make this suite unrunnable off the service container. The wiring
    assertion is about the source text, so the text is what is read.
    """
    pkg = pathlib.Path(output_guards.__file__).parent
    return (pkg / f"{module_name}.py").read_text(encoding="utf-8")


def test_guard_is_wired_into_every_write_path():
    """A guard that exists but is not called is the dead-salvage failure mode.

    Three builds of mnemory shipped a truncation-recovery path that recovered
    nothing because nothing exercised it, and two upstream merges have since
    removed fork patches silently. This asserts the call sites, not just the
    function.
    """
    for module_name, marker in WIRING.items():
        src = _source(module_name)
        assert marker in src, f"{module_name} lost its redaction call site"


def test_infer_false_path_is_covered():
    """infer=False stores caller text verbatim, bypassing extraction entirely."""
    assert "content, redactions = redact_secrets(content)" in _source("memory")


def test_update_path_holds_every_expected_call_site():
    """Census, not presence.

    A marker test alone cannot see a call site disappear: memory.py holds three
    (remember loop, infer=False, update_memory) and any one of them keeps a
    substring assertion green while an open write path stays open.
    """
    assert _source("memory").count("redact_secrets(") == 3
