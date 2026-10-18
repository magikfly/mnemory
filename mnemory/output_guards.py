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
    r"|^(?:the\s+current\s+)?date\s+(?:is|was)\s+\d",
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
