"""Deterministic content guards for LLM-written memory text.

Why this exists: structured output IS enforced against the extraction endpoint
(verified against llama-swap build b1-847f447: a json_schema request returns
the exact schema shape, three times running). Grammar cannot help therefore, because the
artifacts below are perfectly valid strings inside an allowed ``text`` field.
Two independent reproductions in a production store on 2026-10-08:

  "- id=0: Assistant recommended explicitly disabling the healthcheck ..."
  "- [type: fact | categories: technical, project:mnemory | importance: high |
     role: assistant] Assistant decided to enable automatic merge ..."

Those are the consolidation prompt's own alias and metadata-tag formats leaking
into stored text, and the second reproduced leak is a user-subject fact emitted
by the assistant pass, which then becomes ``role=assistant`` and lands in the
agent-identity section of every future prompt. A prompt rule was tried first and
does not hold a 4B model, so the fix is here: strip what is formatting, drop what
is not the writer's role to write.

No third-party imports, so this module is unit-testable without the service.
"""

from __future__ import annotations

import os
import re

__all__ = [
    "clean_memory_text",
    "noise_enforcing",
    "noise_memory",
    "pin_allowed",
    "redact_secrets",
    "redaction_enabled",
    "role_mismatch",
    "triage_memory",
]

#: Below this length, cleaned text carries no usable statement.
MIN_CLEAN_LENGTH = 12

#: Apostrophe, either form.  The extractor writes typographic punctuation into
#: its facts as a matter of course, so an ASCII-only apostrophe in a guard
#: pattern silently misses most real rows.  Defined here, above every pattern
#: that uses it.  tests/test_output_guards_punctuation.py enforces the twin for
#: all patterns in this module, so the next one added cannot repeat the mistake.
_APOS = r"['\u2019]"

# Leading markdown/list noise, including the doubled "- - " form.
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*\u2022]\s+)+")

# The consolidation prompt numbers its sources S0..Sn and its fsck/dedup prompts
# number memories id=0, id=1. Neither belongs in user-facing memory text.
# The optional " (pinned)" group matches the flag build_fsck_duplicate_prompt has
# emitted since 2026-10-09; without it, "- id=0 (pinned): " survives into stored
# text and the guard that exists to catch prompt echoes would miss this one.
_ALIAS_PREFIX_RE = re.compile(
    r"^\s*(?:\(\s*)?id\s*=\s*\d+\s*(?:\))?\s*(?:\(\s*pinned\s*\))?\s*[:\-\u2013]\s*",
    re.IGNORECASE,
)
# Punctuation is required: without it, real text such as "S3 lifecycle rules
# were configured" would be mistaken for a source alias and have its subject cut.
_SOURCE_PREFIX_RE = re.compile(r"^\s*(?:\(\s*)?\[?\s*S\d{1,3}\s*\)?\s*\]?\s*[:\-]\s+")

# A metadata tag block. Two flavours reproduced in the production store:
#   "[type: fact | categories: ... | importance: high]"      consolidation
#   "[scope: open-webui | type: ... | created: 2026-10-08]"  fsck dedup
# Measured 2026-10-09 over 116 stored blocks: 75 open with "scope:", 41 with
# "type:". Keying on "type" alone - the original form of this regex - missed
# 65% of them, including the whole fsck flavour. Both keys are specific enough
# that ordinary brackets ("[S1]", "[10]", "[[nested]]") are still left alone.
_METADATA_BLOCK_RE = re.compile(r"\[\s*(?:scope|type)\s*:[^\]]{0,400}?\]", re.IGNORECASE)

# mnemory's boundary tags (U+27E8 / U+27E9) used by wrap_with_boundary(). The
# whole tag goes, not just the brackets: a test showed stripping only the
# brackets left "memory_item" embedded in the sentence.
_BOUNDARY_TAG_RE = re.compile(r"\u27e8/?[a-z_]*\u27e9")

# Prompt scaffolding that must never survive into stored text.
_SCHEMA_WORD_RE = re.compile(
    r"^\s*(?:store_artifact|source_ids|memory_type|event_date)\s*[:=].*$", re.IGNORECASE
)

_WHITESPACE_RE = re.compile(r"\s{2,}")


def clean_memory_text(text: str | None, min_length: int = MIN_CLEAN_LENGTH) -> str | None:
    """Strip prompt/format artifacts from a memory text.

    Returns the cleaned text, or ``None`` when nothing usable survives. Never
    rewrites wording: it only removes framing, so a real fact cannot be
    silently rephrased into a different claim.
    """
    if not text:
        return None
    cleaned = text
    for pattern in (_LIST_MARKER_RE, _ALIAS_PREFIX_RE):
        cleaned = pattern.sub("", cleaned, count=1)
    cleaned = _METADATA_BLOCK_RE.sub(" ", cleaned)
    cleaned = _BOUNDARY_TAG_RE.sub("", cleaned)
    if _SCHEMA_WORD_RE.match(cleaned.strip()):
        return None
    # Retry the source alias prefix after list markers were removed.
    cleaned = _LIST_MARKER_RE.sub("", cleaned, count=1)
    cleaned = _SOURCE_PREFIX_RE.sub("", cleaned, count=1)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip().strip("-—").strip()
    if len(cleaned) < min_length:
        return None
    return cleaned


def role_mismatch(text: str | None, role: str) -> str | None:
    """Return a machine-readable reason when a memory is in the wrong pass.

    Only one direction is actionable. The assistant pass reads a session summary
    that also describes the user's decisions, so it re-emits them with an
    assistant subject; those are safe to drop, because the same summary drives
    the user pass and the underlying raw memories stay until superseded. The
    reverse direction is reported but never dropped: an assistant-subject fact
    in the user pass has no second writer, so removing it would lose information.
    """
    if not text:
        return None
    probe = _LIST_MARKER_RE.sub("", text, count=1).strip()
    if role == "assistant" and re.match(r"^(?:User|The user)\b", probe, re.IGNORECASE):
        return "assistant_pass_user_subject"
    if role == "user" and re.match(r"^Assistant\b", probe, re.IGNORECASE):
        return "user_pass_assistant_subject"
    return None


# ── write-side noise guard ────────────────────────────────────────────────
# Why: the extractor and the consolidator both store the *act* of asking or
# answering as though it were a durable statement, and both store housekeeping
# counts about this very store. None of those carry information a future turn
# can use, yet each one occupies a retrieval slot for the life of the store.
# Measured 2026-10-15 on a 2705-row store: 15 such rows in one probe pass, and
# every probe ("user asked", "assistant offered") surfaced more of the species.
# Two of them were injected into a live turn within minutes of being written,
# which is how this became worth blocking at the writer instead of purging.
#
# Deliberately narrow. The failure mode that matters is dropping a real fact,
# so the patterns key on a speech-act or question *head* (subject + verb in
# clause-initial position), never on a topic word: "User's maximum heart rate
# is 188 BPM" and "The fsck dedup call site echoes cluster content" must both
# survive, and both are covered by MUST_KEEP in tests/test_output_guards_noise.py.
# Bare domain words ("mnemory", "dedup", "knee") are never sufficient on their
# own, because this user's real work and health facts contain them.

#: "User is asking who the assistant is" / "User asked how old they are"
_QUESTION_PROGRESS_RE = re.compile(
    r"^\s*(?:the\s+)?user\s+(?:is|was|has\s+been|keeps)\s+"
    r"(?:asked|asking|inquir(?:e|ed|ing)|wonder(?:ed|ing)|wants\s+to\s+know)"
    r"\b",
    re.IGNORECASE,
)

#: "User asked how proficient I am in COBOL." - a question, stored as a fact.
_QUESTION_PAST_RE = re.compile(
    r"^\s*(?:the\s+)?user\s+(?:inquired|wondered)\b|"
    r"^\s*(?:the\s+)?user\s+asked\s+(?:how|what|why|whether|if|who|when|"
    r"where)\b",
    re.IGNORECASE,
)

# ── assistant narration: anchored head, verb list, rewriter split ───────────
#
# PRECEDENCE BUG CLOSED 2026-10-18. The rule used to read:
#
#     r"^\s*assistant(?:'s)?\s+(?:is\s+|...)*(?:will|shall)\s+[a-z]+|"
#     r"(?:offer|explain|note|agree|report|advis|recommend|...)\b"
#
# The `|` between those alternatives is TOP LEVEL, so the second branch was
# never anchored to "^\s*assistant". Any sentence containing note / agree /
# report / offer / explain / recommend anywhere at all was classified as
# assistant narration -- and under MEMORY_NOISE_GUARD=drop, deleted. Measured
# against five ordinary user facts, all five dropped:
#
#   "User's wife Lisandra agreed to the move to Zoetermeer"
#   "User's note about knee pain says it started in August"
#   "User keeps a training log and reports it weekly to his coach"
#
# It shipped 2026-10-15 and logged 47 hits in the following 24h, which read as
# a writer producing a great deal of narration. It was mostly the word "note"
# appearing in prose. Both alternatives now share one anchored head, built once
# so the noise rule and triage_memory() below cannot drift apart.

#: Clause-initial narration head: subject + optional perfect/progressive modal.
_ACT_HEAD = (
    r"^\s*(?:the\s+)?assistant(?:" + _APOS + r"s)?\s+"
    r"(?:(?:is|was|has|have|had)\s+)*"
)

#: Reporting verbs that introduce narration rather than a proposition. Outcome
#: verbs (fixed, added, changed, removed, patched) stay out on purpose, and
#: "identified" belongs to _ASSISTANT_META_OBJECT_RE, not here.
_ACT_VERBS = (
    r"offer(?:s|ed)?|explain(?:s|ed)?|not(?:e|es|ed)|mention(?:s|ed)?|"
    r"clarif(?:y|ies|ied)|agree[sd]?|declin(?:e|es|ed)|demonstrat(?:e|es|ed)|"
    r"advis(?:e|es|ed)|warn(?:s|ed)?|recommend(?:s|ed)?|suggest(?:s|ed)?|"
    r"report(?:s|ed)?|stat(?:e|es|ed)|confirm(?:s|ed)?|said|told|urge[sd]?"
)

#: "Assistant explained that ..." / "Assistant recommended using ..." /
#: "Assistant will dump the tokenizer keys".
_ASSISTANT_ACT_RE = re.compile(
    # _APOS keeps the module-wide punctuation invariant intact
    # (tests/test_output_guards_punctuation.py): the extractor writes U+2019.
    _ACT_HEAD + r"(?:will|shall|would|can|could)\s+[a-z]+"
    r"|" + _ACT_HEAD + r"(?:" + _ACT_VERBS + r")\b",
    re.IGNORECASE,
)

#: Head + reporting verb + everything after it, so triage_memory() can recover
#: the proposition the narration was wrapped around.
_ACT_SPLIT_RE = re.compile(
    _ACT_HEAD + r"(?P<verb>" + _ACT_VERBS + r")\b(?P<rest>.*)\Z",
    re.IGNORECASE | re.DOTALL,
)

#: Verbs where the act IS the content; what follows says nothing a later reader
#: can use ("Assistant offered to adapt it", "agreed to the plan").
_ACT_ONLY_VERBS = ("offer", "agree", "decline", "apolog")

#: A clause after one of these is an instruction, so when it has no subject of
#: its own it is labelled as advice instead of being left a floating fragment.
_ADVICE_VERBS = ("advis", "recommend", "suggest", "urge", "warn", "told",
                 "tell", "said")

#: A recovered clause shorter than this is not worth keeping.
MIN_REWRITE_CLAUSE = 20



#: A pointer to where a fact lives, instead of the fact. Retrieval returns the
#: pointer and the caller still has to find the real row.
_POINTER_RE = re.compile(
    r"^\s*(?:the\s+)?user\s+wants\s+to\s+have\s+.{0,48}\bstored\b|"
    r"\b(?:real\s+)?name\s+is\s+stored\s+(?:as|at|in)\b",
    re.IGNORECASE,
)

#: Housekeeping state of the store itself, e.g. "store contains 408 rows
#: classified as delete_safe" or "purge: 442 rows retracted, 104 stripped".
#: Needs a count AND a housekeeping verb, so "grows at 40-60 rows per day"
#: (a real capacity metric) is not caught.
_STORE_HOUSEKEEPING_RE = re.compile(
    r"\b\d[\d,.]*\s*(?:rows?|memories|records|entries)\b"
    r"[^.]{0,80}\b(?:retract|delet|strip|dedup|purge|classif|archiv|merge)"
    r"[a-z]*",
    re.IGNORECASE,
)

#: Indexical reference into a numbered list the extractor was reading, e.g.
#: "Fact 5 identifies a specific data corruption issue".  Unresolvable once the
#: producing chat is gone, so it is information-free to every later reader.
#: Observed as a cluster of six such rows written on 2026-10-07/08.
_META_INDEXICAL_RE = re.compile(
    r"^\s*(?:fact|item|point|entry|row|note|issue|number)\s+\d+\b",
    re.IGNORECASE,
)

#: The session clock stored as durable knowledge: "Today's date is Wednesday, 7
#: October 2026.", and the "user indicated today's date is ..." framing.  Both
#: were sitting in the store classified as type=fact, i.e. never decaying.
_DATE_ECHO_RE = re.compile(
    r"^(?:today|yesterday|tomorrow)(?:" + _APOS + r"s)?\s+date\s+is\b"
    r"|\btoday(?:" + _APOS + r"s|s)\s+date\b"
    # Deliberately narrow: only the two shapes actually observed in the store.
    # A general "the date is ..." rule would drop legitimate facts such as "The
    # date is August 13, my birthday", and there is no evidence broadening it.
    r"|^(?:the\s+current\s+)?date\s+(?:is|was)\s+\d"
    # 2026-10-23: the injection-template family.  The client filter prepends a
    # rendered "Current date context: <date>" line and the extractor narrates it
    # back as a user fact; five such rows were purged from the live store, and
    # two of them had stored the placeholder UNEXPANDED -- proof the extractor
    # is fed the template, not the conversation.  An unexpanded brace pair is
    # never a durable statement, so that alternative is safe on its own; the
    # phrase rule is the deliberate part.  Accepted collateral, pinned by
    # test_known_accepted_collateral: a meta-note that quotes the placeholder
    # is dropped too -- write those to the ops journal, not the memory store.
    r"|\bcurrent\s+date\s+context\b"
    r"|\{\s*current[_\s-]*date\s*\}",
    re.IGNORECASE,
)

#: Literal transcript form -- a bare quoted utterance with no proposition
#: attached: 'Assistant: "That settles it — lock in 512"'.  _ASSISTANT_ACT_RE
#: matches narration of the "Assistant explained that ..." kind; it does not
#: match this quoting shape, which is the same defect wearing different clothes.
_SPEAKER_QUOTE_RE = re.compile(
    r'^\s*(?:assistant|user|ai|model)\s*:\s*["\u201c\u2018].*["\u201d\u2019]\s*$',
    re.IGNORECASE | re.DOTALL,
)

#: The only subjects a *newly extracted* memory may earn a permanent pin for:
#: identity, medical, location, role.  The extraction schema makes ``pinned`` a
#: REQUIRED field, so the model must emit a value on every single fact; without
#: a server-side allowlist it guesses, and the guesses land in core context.
#: Measured cost of guessing, from the live store: a Planck-constant
#: explanation, a resting-HR reading and two credential-audit counts all arrived
#: pinned=true.  Explicit add_memory/update calls bypass this -- a user asking
#: to pin something is a different act from a model guessing.
_PIN_ALLOW_RE = re.compile(
    r"^(?:user|assistant)(?:" + _APOS + r"s|s)?\s+(?:full\s+)?name\s+is\b"
    r"|\b(?:born|birthday|date of birth)\b"
    r"|\b(?:metformin|semaglutide|ozempic|empagliflozin|jardiance|insulin|"
    r"diabetes|t2d|allerg(?:y|ic)|anaphylaxis|coeliac|celiac|asthma|"
    r"sulfonylurea|glp-1|sglt2)\b"
    r"|\b(?:lives|located|resides|based)\s+in\b"
    r"|^(?:user|assistant)(?:" + _APOS + r"s|s)?\s+(?:job|profession|occupation|role)\s+is\b"
    r"|\bis\s+an?\s+(?:[a-z]+\s+){0,3}(?:engineer|architect|developer|"
    r"designer|manager|consultant|analyst|freelancer|professional|student)\b"
    r"|\bworks\s+(?:at|for|as)\b",
    re.IGNORECASE,
)


def pin_allowed(text: str | None) -> bool:
    """True when an extracted memory's text justifies ``pinned=True``.

    Used to *withhold* pins, never to grant them: a caller keeps its own pin
    only when this returns True.  Withholding a pin from a row that has not been
    written yet loses nothing durable -- the fact stays fully searchable --
    which is why this can be enforced on sight, unlike the drop-on-noise
    behaviour behind ``noise_enforcing``.
    """
    if not text:
        return False
    return bool(_PIN_ALLOW_RE.search(_LIST_MARKER_RE.sub("", text, count=1)))

# ── Secret redaction ─────────────────────────────────────────────────
#
# Why this guard applies on sight while the noise drops ship closed: redaction
# keeps the row. A memory that said "the API key is sk-abc..." still records
# that a key exists and what it is for; only the value is removed. Under
# MEMORY_NOISE_GUARD=drop every false positive is permanent, because mnemory
# has no un-retract endpoint. That constraint does not apply here, so this
# belongs in the clean_memory_text tier: it rewrites stored text
# unconditionally and behind no destructive flag.
#
# ORDERING CONTRACT -- every call site runs this BEFORE logging any text. Guard
# verdicts go to /data/logs/mnemory.log and to `docker logs`; a guard that
# logs the secret it just caught has copied the credential out of the vector
# store into an unencrypted log file, which is a worse leak than the one being
# prevented. Every call site in prompts.py, memory.py and consolidation.py
# assigns the redacted text before its first logger call for that fact.
_REDACTED_FMT = "[REDACTED:{kind}]"

#: Matches an already-emitted marker, so re-saving a redacted memory cannot
#: redact its own placeholder or report a second hit for one secret.
_REDACTED_ANY_RE = re.compile(r"\[REDACTED:[a-z0-9_]+\]")

# Named token shapes. Deliberately free of ASCII quote and apostrophe
# characters: these patterns run over machine identifiers, not prose, and the
# sweep in tests/test_output_guards_punctuation.py asserts a curly-punctuation
# twin for every quote character a pattern contains. Adding one here would
# either fail that sweep or force a per-pattern exemption, and that test's own
# docstring rejects an exemption list as encoding the wrong rule. A GitHub
# token is never written with curly quotes.
_GH_TOKEN_RE = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,99}\b")
_AWS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA|AIDA|AROA|AGPA|ANPA|APKA)[0-9A-Z]{16}\b")
_SLACK_TOKEN_RE = re.compile(r"\bxox[baprs](?:-[A-Za-z0-9]{8,})+\b")
_GOOGLE_API_RE = re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")
_STRIPE_KEY_RE = re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}\b")
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")
_JWT_RE = re.compile(
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
)
_PRIVATE_KEY_RE = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
_TELEGRAM_BOT_RE = re.compile(r"\b\d{8,10}:AA[A-Za-z0-9_-]{30,}\b")

#: Applied in this order: specific shapes first, so the log names what was
#: actually found instead of reporting the generic catch-all below.
_TOKEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_key", _PRIVATE_KEY_RE),
    ("github_token", _GH_TOKEN_RE),
    ("aws_access_key", _AWS_KEY_RE),
    ("slack_token", _SLACK_TOKEN_RE),
    ("google_api_key", _GOOGLE_API_RE),
    ("stripe_key", _STRIPE_KEY_RE),
    ("telegram_bot_token", _TELEGRAM_BOT_RE),
    ("jwt", _JWT_RE),
    ("openai_key", _OPENAI_KEY_RE),
)

#: Generic label-delimited credential. The value class is \S rather than an
#: explicit "not a quote" class on purpose -- see the sweep note above. \S
#: swallows a wrapping quote along with the value, which costs one cosmetic
#: character and cannot leak anything.
#:
#: Bare "token" is NOT a label here, and that is a measured decision, not an
#: omission: this store holds a lot of token-budget facts, and \btoken\b\s*[:=]
#: matches the tail of "--max-tokens: 8192" because a hyphen is a word
#: boundary. Qualifier-less labels only.
_CRED_KV_RE = re.compile(
    r"(?i)\b("
    r"api[_ -]?keys?|api[_ -]?secrets?|access[_ -]?tokens?|"
    r"refresh[_ -]?tokens?|auth[_ -]?tokens?|id[_ -]?tokens?|"
    r"client[_ -]?secrets?|secret[_ -]?keys?|private[_ -]?tokens?|"
    r"deploy[_ -]?keys?|ssh[_ -]?keys?|bearer|authorization|"
    r"authorisation|passwords?|passphrase|passwd|pwd|"
    r"wachtwoord|paswoord|geheim"
    r")\b(\s*[:=]\s*)(\S{6,})"
)
# Dutch labels are listed because the store holds Dutch-language facts and the
# extractor preserves input language.

#: Words that read like a credential value but are ordinary prose sitting after
#: the noun. Without this, "password: required" loses a word for nothing.
_CRED_VALUE_STOPWORDS = frozenset(
    {
        "required", "requested", "mandatory", "changed", "rotate", "rotated",
        "expired", "reset", "unknown", "missing", "needed", "stored",
        "hidden", "never", "always", "yours", "mine", "here", "this", "that",
        "none", "username", "user", "account", "login", "email", "e-mail",
        "password", "passwd", "pwd", "api", "key", "secret", "token",
        "and", "the", "for", "with", "again", "today", "tomorrow",
        "verplicht", "ingesteld", "onbekend",
    }
)

_REDACTION_ENV = "MEMORY_REDACTION"


def redaction_enabled() -> bool:
    """True unless MEMORY_REDACTION=off. On by default.

    ``off`` exists for one job only: reproducing a suspected false positive
    against a live row. It is not a safe steady state -- the alternative is a
    usable credential in the vector store and in every prompt that recalls it.
    """
    return os.environ.get(_REDACTION_ENV, "on").strip().lower() not in {
        "off",
        "0",
        "false",
        "no",
    }


def _looks_credential(value: str) -> bool:
    """Decide whether a label-delimited value is machine-made rather than prose.

    Requires six characters minimum and then either a digit anywhere (hunter2,
    sk-ab12cd) or a length no ordinary word reaches. Deliberately NOT accepting
    "contains an underscore or hyphen": that redacted "well-known" and
    "high-value" in review, which is the same false-positive class that made
    the assistant_speech_act rule unsafe to enforce.
    """
    # Raw, before the strip below: the strip removes [ ] and :, which are the
    # very characters that identify an already-emitted marker. Checked in the
    # wrong order, a re-saved row redacts its own placeholder and reports a
    # fresh hit for a secret that is no longer there.
    if _REDACTED_ANY_RE.search(value):
        return False
    v = value.strip("()[]{}<>.,;:!?\u201c\u201d\u2018\u2019\"'")
    if len(v) < 6 or v.lower() in _CRED_VALUE_STOPWORDS:
        return False
    return any(ch.isdigit() for ch in v) or len(v) >= 16


def redact_secrets(text: str | None) -> tuple[str | None, list[str]]:
    """Replace secret values with ``[REDACTED:<kind>]``; never drop a memory.

    Returns ``(text, kinds)`` where ``kinds`` lists what was replaced and is
    empty when the text was clean. Honours ``redaction_enabled()`` internally
    so a call site cannot forget the switch. See the ordering contract above:
    run this before the text reaches a log line.
    """
    if not text or not redaction_enabled():
        return text, []
    out = text
    kinds: list[str] = []

    def _mark(kind: str) -> str:
        if kind not in kinds:
            kinds.append(kind)
        return _REDACTED_FMT.format(kind=kind)

    for kind, pattern in _TOKEN_PATTERNS:
        out = pattern.sub(lambda m, k=kind: _mark(k), out)

    def _kv_sub(m: "re.Match[str]") -> str:
        if not _looks_credential(m.group(3)):
            return m.group(0)
        return f"{m.group(1)}{m.group(2)}{_mark('credential_value')}"

    out = _CRED_KV_RE.sub(_kv_sub, out)
    return out, kinds


_META_HEAD_RE = re.compile(
    r"^\s*(?:the\s+)?(?:dashboard|memory\s+(?:store|ui|system|filter))\b",
    re.IGNORECASE,
)
_META_PHRASE_RE = re.compile(
    r"\b(?:bucket\s+noise|data\s+quality\s+issues?|retrieval\s+miss(es)?|"
    r"self-reference\s+loop|memory\s+(?:quality|noise)\s+(?:issues?|improve))\b",
    re.IGNORECASE,
)

#: Narration verb followed by an object from the memory subsystem. The guard
#: deliberately excludes "identified" from _ASSISTANT_ACT_RE, because "Assistant
#: identified the root cause and patched the migration path" is a real outcome.
#: But the writer keeps producing "Assistant identified four retrieval misses"
#: and "Assistant identified that 3 remaining misses are not ranking problems",
#: both transcribed from a single assistant turn seconds after they were said.
#: Requiring a memory-subsystem object narrows the rule without re-widening the
#: verb list, so the engineering-record phrasing survives.
_ASSISTANT_META_OBJECT_RE = re.compile(
    r"^\s*assistant\s+(?:is\s+|was\s+|has\s+)?(?:identif(?:y|ies|ied)|"
    r"not(?:e|es|ed)|explain(?:s|ed)?|report(?:s|ed)?|observ(?:e|es|ed)|"
    r"confirm(?:s|ed)?|note(?:s|d)?|note?d?)\b[^.]{0,60}\b"
    r"(?:memor(?:y|ies)|retrieval|recall|miss(?:es)?|rows?|store|noise|filter|"
    r"retract|pinned|dedup|fsck|pointer|bucket|entry|carried)\b",
    re.IGNORECASE,
)

#: "User initiated a greeting with 'hey there'" / "User greeted the assistant".
#: A greeting is not a fact about anyone, and the recall-side trivia filter
#: already refuses these once stored; refusing them at the writer is cheaper.
_GREETING_ACT_RE = re.compile(
    r"^\s*(?:the\s+)?(?:user|assistant)\s+(?:greet(?:ed|s)|initiated\s+a\s+"
    r"greeting|opened\s+with\s+(?:a\s+)?(?:greeting|hi|hello|hey)|"
    r"said\s+(?:hi|hello|hey)|exchanged\s+greetings)\b",
    re.IGNORECASE,
)

#: log = report only (default), drop = actually refuse the row. There is no
#: un-retract endpoint, so refusing to write must be a conscious opt-in.
# ── user-side speech acts ────────────────────────────────────────────────
#: LOCAL-PATCH 2026-10-28 (gates/agent-voice).
#: The assistant side is policed (_ASSISTANT_ACT_RE). The user side was not,
#: and the 2,312-row census of 2026-10-27 measured the cost: of 1,166
#: narration-shaped rows, 434 carry role="user" -- "User requested to bump
#: max-file to 5 in docker-compose.yml", "User expressed frustration with
#: memory's current state". Policing only the assistant shape fixes 63% of the
#: surface and teaches the extractor to keep laundering the rest under the user
#: role.
#:
#: Deliberately absent from the verb list: decided, prefers, wants, needs,
#: plans, agreed. Those are the durable classes this store exists for (a
#: decision, a preference, an open loop). "agreed" is the recorded false
#: positive of 2026-10-18 -- a sentence of the form "<spouse> agreed to the
#: move" must survive, so it stays out. The fixture pinning that lives in
#: tests/test_output_guards_agent_voice.py.
#:
#: NOTE for whoever flips MEMORY_NOISE_USER_ACTS=drop: unlike the assistant
#: case, there is no that-clause rescue wired for these heads yet. Run in
#: shadow first and read the would-drop list; if substance is wrapping itself
#: in a user speech act, extend triage_memory() with a user-head split before
#: enforcing.
_USER_ACT_HEAD = (
    r"^\s*(?:the\s+)?user(?:" + _APOS + r")?\s+"
    r"(?:(?:is|was|has|have|had)\s+)*"
)
_USER_ACT_RE = re.compile(
    _USER_ACT_HEAD
    + r"(?:request(?:s|ed)?|ask(?:s|ed)|inquir(?:e|es|ed|ing)"
    r"|express(?:es|ed)?|complain(?:s|ed)?|thank(?:s|ed)?"
    r"|apologis(?:e|ed|ing)|reiterat(?:es|ed)?|clarif(?:y|ies|ied)"
    r"|acknowledg(?:es|ed)?|volunte(?:er|ered|ers)|mention(?:s|ed)?"
    r"|stat(?:e|es|ed)|inform(?:s|ed)?|report(?:s|ed)?)\b",
    re.IGNORECASE,
)

#: Facts the extractor attributes to the assistant. This is NOT the assistant
#: identity: identity rows are role=assistant AND memory_type in (fact,
#: preference) AND usually pinned -- "Assistant's name is Atlas" is exactly
#: that shape and must survive. Measured on the live store: of 1,070
#: role=assistant rows, 999 are episodic or context (transcribed utterances)
#: and 71 are fact/preference/procedural or pinned. Refusing the episodic and
#: context ones removes 93% of the agent voice while leaving the persona and
#: the durable capability notes intact.
AGENT_VOICE_TYPES = frozenset({"episodic", "context"})

#: Per-rule enforcement modes. Both new rules ship CLOSED and in shadow, even
#: though MEMORY_NOISE_GUARD is already set to drop in production: a new
#: pattern added to a predicate that is already enforcing starts deleting on
#: the deploy that adds it, and there is no un-retract endpoint.
_USER_ACT_MODE_ENV = "MEMORY_NOISE_USER_ACTS"
_AGENT_VOICE_MODE_ENV = "MEMORY_AGENT_VOICE"


def _mode(env: str) -> str:
    return os.getenv(env, "shadow").strip().lower()


def user_act_enforcing() -> bool:
    """True when MEMORY_NOISE_USER_ACTS=drop."""
    return _mode(_USER_ACT_MODE_ENV) == "drop"


def agent_voice_enforcing() -> bool:
    """True when MEMORY_AGENT_VOICE=drop."""
    return _mode(_AGENT_VOICE_MODE_ENV) == "drop"


def noise_enforcing_for(reason: str | None) -> bool:
    """Whether a given noise reason should actually refuse the row.

    The single global noise_enforcing() cannot be reused for new rules: it is
    already `drop` on this deployment, so any pattern added to noise_memory()
    would start deleting the moment it merged. Each new reason gets its own
    switch; every legacy reason keeps inheriting MEMORY_NOISE_GUARD so no
    existing behaviour changes.
    """
    if reason == "user_speech_act":
        return user_act_enforcing()
    return noise_enforcing()


def agent_voice(role: str | None, memory_type: str | None, pinned: bool,
                importance: str | None = None) -> str | None:
    """Return "assistant_agent_voice" for a transcribed assistant utterance.

    Pure and deterministic, like the rest of this module: no model call, no
    store access. `role` is the per-fact field the extractor already emits
    (prompts.py "Each extracted fact must include a role field"), which means
    the provenance we needed was already being produced and thrown away.
    """
    if (role or "").strip().lower() != "assistant":
        return None
    if pinned:
        return None
    if (memory_type or "").strip().lower() not in AGENT_VOICE_TYPES:
        return None
    if str(importance or "").strip().lower() in ("high", "critical"):
        return None
    return "assistant_agent_voice"


_NOISE_MODE_ENV = "MEMORY_NOISE_GUARD"


def noise_enforcing() -> bool:
    """True when the noise guard should drop rows rather than only log them."""
    return os.getenv(_NOISE_MODE_ENV, "log").strip().lower() == "drop"


def noise_memory(text: str | None) -> str | None:
    """Return a reason code when a memory carries no durable statement.

    ``None`` means "keep it". Like ``role_mismatch`` this is pure and
    deterministic: no model call, no store access, unit-testable in isolation.
    """
    if not text:
        return None
    probe = _LIST_MARKER_RE.sub("", text, count=1).strip()
    if not probe:
        return "empty_after_strip"
    for regex, reason in (
        (_GREETING_ACT_RE, "greeting_narration"),
        (_ASSISTANT_META_OBJECT_RE, "assistant_meta_observation"),
        (_QUESTION_PROGRESS_RE, "question_in_progress"),
        (_QUESTION_PAST_RE, "question_recorded_as_fact"),
        (_ASSISTANT_ACT_RE, "assistant_speech_act"),
        (_POINTER_RE, "pointer_not_fact"),
        (_STORE_HOUSEKEEPING_RE, "store_housekeeping"),
        (_META_INDEXICAL_RE, "meta_indexical"),
        (_DATE_ECHO_RE, "date_echo"),
        (_SPEAKER_QUOTE_RE, "speaker_quote"),
        (_USER_ACT_RE, "user_speech_act"),
    ):
        if regex.search(probe):
            return reason
    if _META_HEAD_RE.search(probe) or _META_PHRASE_RE.search(probe):
        return "about_the_memory_system"
    return None


def triage_memory(text: str | None) -> tuple[str, str | None, bool]:
    """Decide what to store: ``(text_to_store, drop_reason, was_rewritten)``.

    Narration is not always noise. "Assistant recommended not running on an
    empty stomach" wraps a real instruction inside a speech-act head, and
    dropping the row loses the instruction. Measured 2026-10-18 on 24h of
    ``would drop`` logs, ``assistant_speech_act`` was firing on substantive
    guidance: the euDKA fueling rule, sodium intake with empagliflozin, a
    cadence cue, a scheduled rest day. Enforcing the drop as written would have
    silently deleted medical advice.

    So the head is stripped and the clause kept, in exactly two provable shapes:

      * an explicit ``that``-clause, kept verbatim -- it already has a subject,
        so nothing has to be inferred;
      * a subject-less clause (gerund, or the ``told the user to ...``
        indirect-object form), given a short provenance label rather than being
        rephrased into a bald assertion.

    Everything else keeps the original drop verdict. This does not breach
    ``clean_memory_text``'s "never rewrite wording" invariant: that function
    removes formatting only, whereas this one is a separate, logged operation
    whose whole job is de-narration, and it removes precisely the reporting head
    it matched and nothing else.
    """
    reason = noise_memory(text)
    if reason != "assistant_speech_act" or not text:
        return text, reason, False

    probe = _LIST_MARKER_RE.sub("", text, count=1).strip()
    m = _ACT_SPLIT_RE.match(probe)
    if m is None:
        return text, reason, False        # future-modal form: pure act
    verb = m.group("verb").lower()
    if verb.startswith(_ACT_ONLY_VERBS):
        return text, reason, False

    rest = (m.group("rest") or "").strip()
    rest = re.sub(r"^[\s,;:.\u2013\u2014-]+", "", rest)
    had_that = bool(re.match(r"^(?:the\s+fact\s+)?that\b", rest, re.IGNORECASE))
    rest = re.sub(r"^(?:the\s+fact\s+)?that\s+", "", rest, flags=re.IGNORECASE)
    # "told/advised the user to drink water" -> "drink water"
    rest = re.sub(r"^(?:the\s+)?user\s+(?:to|that)\s+", "", rest,
                  flags=re.IGNORECASE)
    # Live 2026-10-19: the extractor writes "recommended AGAINST running",
    # not "recommended NOT running". The negation belongs to the reporting
    # verb, so a bare head-strip left "Advice: Against running on an empty
    # stomach" -- a prepositional fragment that loses the "do not" force
    # altogether. Fold it into the clause instead.
    rest = re.sub(r"^against\s+", "not ", rest, flags=re.IGNORECASE)
    # "advised to sleep more" -> "sleep more"; the infinitive marker is part
    # of the head, same as "that".
    rest = re.sub(r"^to\s+", "", rest, flags=re.IGNORECASE)
    rest = _WHITESPACE_RE.sub(" ", rest).strip(" .;,:")

    if len(rest) < MIN_REWRITE_CLAUSE:
        return text, reason, False

    # Only capitalise an ordinary leading word. "Assistant noted that
    # qdrant-client 1.19.1 is pinned" must not become "Qdrant-client ...":
    # re-shaping a version string or identifier is exactly the silent
    # rewording the guard has to avoid.
    capitalised = rest
    if re.match(r"^[a-z][a-z]+\b", rest):
        capitalised = rest[0].upper() + rest[1:]

    # Advice verbs always keep a label, including before a that-clause. Dropping
    # the head from "Assistant recommended that the user sleep more" and storing
    # "The user sleep more" would turn advice into a broken assertion about a
    # third party; the label preserves the modality that the verb carried.
    if verb.startswith(_ADVICE_VERBS):
        return f"Advice: {capitalised}", None, True

    if had_that:
        return capitalised, None, True

    subjectless = (
        re.match(r"^(?:not\s+)?[a-z]+ing\b", rest, re.IGNORECASE)
        or re.match(
            r"^(?:drink|run|eat|keep|avoid|take|sleep|hydrate|stretch|rest|"
            r"raise|reduce|add|drop|swap|try|aim|hold|push|slow|speed|"
            r"weigh|log|record|prioritise|prioritize|expect|monitor)\b",
            rest, re.IGNORECASE,
        )
    )
    if subjectless:
        label = "Advice" if verb.startswith(_ADVICE_VERBS) else "Note"
        return f"{label}: {capitalised}", None, True

    return text, reason, False
