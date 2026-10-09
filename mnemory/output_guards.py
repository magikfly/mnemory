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

import re

__all__ = ["clean_memory_text", "role_mismatch"]

#: Below this length, cleaned text carries no usable statement.
MIN_CLEAN_LENGTH = 12

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
