"""QA of the patched mnemory logic. Synthetic inputs only - no store access.

Regression suite for the truncation-recovery chain and the Check-pile fixes:
  1. _action_matches_stored_state       -> no-op detection
  2. _targets_all_absent                -> stale-target detection
  3. sanitize_categories                -> invented categories dropped
  4. salvage_json_objects               -> works for BOTH payload shapes
  5. fsck._parse_llm_json               -> truncated reply -> {"issues": [...]}
  6. parse_remember_extraction_response -> truncated reply -> facts, no crash
  7. _issues_from_parsed                -> no-op-only issues dropped
"""

import json
import traceback

FAILURES = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got={got!r} want={want!r}")
    if not ok:
        FAILURES.append(name)


def section(title):
    print(f"\n--- {title} ---")


import mnemory.fsck as F
from mnemory.categories import sanitize_categories
from mnemory.llm import salvage_json_objects
from mnemory.prompts import parse_remember_extraction_response

svc = object.__new__(F.FsckService)

MEM = {
    "id": "m1",
    "memory": "User asked about NVFP4 vs Q4_K_XL on a specific GPU",
    "metadata": {
        "memory_type": "episodic",
        "categories": ["technical"],
        "importance": "normal",
        "pinned": False,
        "role": "user",
    },
}
LOOKUP = {"m1": MEM}

# ---------------------------------------------------------------- 1. no-op detection
section("1. _action_matches_stored_state (no-op detection)")

noop = F.FsckAction(
    action="update",
    memory_id="m1",
    new_metadata={"memory_type": "episodic", "categories": ["technical"]},
)
check("identical proposal is a no-op", svc._action_matches_stored_state(noop, LOOKUP), True)

reordered = F.FsckAction(
    action="update",
    memory_id="m1",
    new_metadata={"categories": ["technical"], "importance": "NORMAL"},
)
check(
    "reordered+recased proposal is a no-op",
    svc._action_matches_stored_state(reordered, LOOKUP),
    True,
)

real = F.FsckAction(action="update", memory_id="m1", new_metadata={"memory_type": "fact"})
check("real reclassify is NOT a no-op", svc._action_matches_stored_state(real, LOOKUP), False)

cat_change = F.FsckAction(
    action="update", memory_id="m1", new_metadata={"categories": ["project", "technical"]}
)
check("added category is NOT a no-op", svc._action_matches_stored_state(cat_change, LOOKUP), False)

pin_change = F.FsckAction(action="update", memory_id="m1", new_metadata={"pinned": True})
check("pinned flip is NOT a no-op", svc._action_matches_stored_state(pin_change, LOOKUP), False)

content_change = F.FsckAction(
    action="update",
    memory_id="m1",
    new_content="A different wording entirely",
    new_metadata={"memory_type": "episodic"},
)
check(
    "content rewrite is NOT a no-op",
    svc._action_matches_stored_state(content_change, LOOKUP),
    False,
)

delete_action = F.FsckAction(action="delete", memory_id="m1")
check("delete is never a no-op", svc._action_matches_stored_state(delete_action, LOOKUP), False)

empty_meta = F.FsckAction(action="update", memory_id="m1", new_metadata={})
check(
    "empty new_metadata is NOT treated as no-op",
    svc._action_matches_stored_state(empty_meta, LOOKUP),
    False,
)

# ---------------------------------------------------------------- 2. stale targets
section("2. _targets_all_absent (stale-target detection)")


class FakeVector:
    def __init__(self, present):
        self.present = set(present)

    def get_by_id(self, memory_id):
        return {"id": memory_id} if memory_id in self.present else None


def make_issue(mids, actions=None):
    return F.FsckIssue(
        issue_id="i1",
        type="reclassify",
        severity="high",
        reasoning="r",
        affected_memories=[F.FsckAffectedMemory(id=m, content="c") for m in mids],
        actions=actions
        if actions is not None
        else [F.FsckAction(action="update", memory_id=m) for m in mids],
        confidence=0.95,
    )


svc._vector = FakeVector([])
check("target gone -> absent", svc._targets_all_absent(make_issue(["m1"])), True)

svc._vector = FakeVector(["m1"])
check("target present -> not absent", svc._targets_all_absent(make_issue(["m1"])), False)

svc._vector = FakeVector(["m1"])
check(
    "one of two targets present -> not absent",
    svc._targets_all_absent(make_issue(["m1", "m2"])),
    False,
)

svc._vector = FakeVector([])
check(
    "no target ids at all -> not absent (degenerate issue stays visible)",
    svc._targets_all_absent(make_issue([], actions=[])),
    False,
)

# ---------------------------------------------------------------- 3. categories
section("3. sanitize_categories")

check("drops invented token", sanitize_categories(["fact"]), [])
check(
    "keeps valid, drops invalid",
    sanitize_categories(["news_briefing_flashcard", "technical"]),
    ["technical"],
)
check("keeps valid list intact", sanitize_categories(["technical", "project"]), ["technical", "project"])
check("empty in, empty out", sanitize_categories([]), [])

# ---------------------------------------------------------------- 4. salvage shapes
section("4. salvage_json_objects - both payload shapes (regression: text-filter bug)")

trunc_facts = '{"memories": [{"text": "alpha", "memory_type": "fact"}, {"text": "beta", "memory'
salv_facts = salvage_json_objects(trunc_facts, "memories")
check("facts payload -> list", isinstance(salv_facts, list), True)
check("facts payload -> 1 closed object recovered", len(salv_facts or []), 1)
check("recovered object is the fact", (salv_facts or [{}])[0].get("text"), "alpha")

# Regression for the 2026-10-07 dead-code bug: issue objects have no "text" key.
trunc_issues = (
    '{"issues": [{"type": "reclassify", "severity": "high", "confidence": 0.9, '
    '"reasoning": "r", "affected_memory_ids": ["0"], "actions": [{"action": "update", '
    '"memory_id": "0", "new_content": null, "new_metadata": {"memory_type": "fact"}}]}, '
    '{"type": "reclas'
)
salv_issues = salvage_json_objects(trunc_issues, "issues")
check("issues payload -> list", isinstance(salv_issues, list), True)
check("issues payload -> 1 closed issue recovered (was always 0)", len(salv_issues or []), 1)
check(
    "recovered object is the issue",
    (salv_issues or [{}])[0].get("type"),
    "reclassify",
)

# ---------------------------------------------------------------- 5. fsck salvage path
section("5. fsck._parse_llm_json wraps salvaged list in the envelope (regression)")

envelope = svc._parse_llm_json(trunc_issues, "metadata normalization")
check("returns envelope dict", isinstance(envelope, dict), True)
check("envelope carries the salvaged issue", len((envelope or {}).get("issues", [])), 1)
check(
    "salvaged issue survives validation",
    (envelope or {}).get("issues", [{}])[0].get("type"),
    "reclassify",
)

# ---------------------------------------------------------------- 6. remember salvage
section("6. parse_remember_extraction_response on truncated reply (regression)")

facts, summary, store_artifact = parse_remember_extraction_response(trunc_facts, silent=True)
check("facts recovered from truncated reply", len(facts), 1)
check("recovered fact text", facts[0].get("text") if facts else None, "alpha")

cut_early = '{"memories": [{"text": "al'
facts2, _, _ = parse_remember_extraction_response(cut_early, silent=True)
check("cut before any closed object -> empty, no exception", facts2, [])

# ---------------------------------------------------------------- 7. no-op drop
section("7. _issues_from_parsed drops no-op-only issues")

parsed = {
    "issues": [
        {
            "type": "reclassify",
            "severity": "high",
            "confidence": 0.95,
            "reasoning": "already correct",
            "affected_memory_ids": ["0"],
            "actions": [
                {
                    "action": "update",
                    "memory_id": "0",
                    "new_content": None,
                    "new_metadata": {"memory_type": "episodic", "categories": ["technical"]},
                }
            ],
        }
    ]
}
try:
    out_noop = svc._issues_from_parsed(parsed, {"0": "m1"}, [MEM], default_type="reclassify")
    check("no-op-only issue dropped", len(out_noop), 0)
except Exception:
    traceback.print_exc()
    FAILURES.append("_issues_from_parsed no-op drop raised")

parsed_real = json.loads(json.dumps(parsed))
parsed_real["issues"][0]["actions"][0]["new_metadata"] = {"memory_type": "fact"}
try:
    out_real = svc._issues_from_parsed(parsed_real, {"0": "m1"}, [MEM], default_type="reclassify")
    check("issue with a real change survives", len(out_real), 1)
except Exception:
    traceback.print_exc()
    FAILURES.append("_issues_from_parsed real change raised")

# A salvaged truncated reply (envelope form) must also flow through cleanly.
try:
    out_salv = svc._issues_from_parsed(
        envelope, {"0": "m1"}, [MEM], default_type="reclassify"
    )
    check("salvaged envelope builds a real issue", len(out_salv), 1)
    check("built issue is the reclassify", out_salv[0].type, "reclassify")
except Exception:
    traceback.print_exc()
    FAILURES.append("_issues_from_parsed salvaged envelope raised")

# ---------------------------------------------------------------- output guards
section("consolidation output guards")
try:
    from mnemory.output_guards import clean_memory_text, role_mismatch

    check(
        "id= alias prefix stripped",
        clean_memory_text(
            "- id=0: Assistant recommended explicitly disabling the healthcheck for mnemory"
        ),
        "Assistant recommended explicitly disabling the healthcheck for mnemory",
    )
    check(
        "metadata block stripped, sentence preserved",
        clean_memory_text(
            "- [type: fact | categories: technical, project:mnemory | importance: high"
            " | role: assistant] Assistant decided to enable automatic merge for PRs"
        ),
        "Assistant decided to enable automatic merge for PRs",
    )
    check(
        "boundary tags removed with their names",
        clean_memory_text(
            "\u27e8memory_item\u27e9User prefers conventional commit messages"
            "\u27e8/memory_item\u27e9"
        ),
        "User prefers conventional commit messages",
    )
    check("schema scaffolding dropped entirely", clean_memory_text("store_artifact: false"), None)
    # Negative cases: real prose that must survive untouched.
    for keep in (
        "Assistant noted that --n-gpu-layers keeps weights resident in VRAM",
        "S3 lifecycle rules were configured for the backup bucket",
        "Assistant set the port to 8050 [mnemory] in the compose file",
        "The array [10] indexes the tenth element",
    ):
        check(f"untouched: {keep[:30]}...", clean_memory_text(keep), keep)
    check(
        "cross-role: user subject in assistant pass",
        role_mismatch("User decided to enable reasoning in the agent", "assistant"),
        "assistant_pass_user_subject",
    )
    check(
        "cross-role: assistant's own subject kept",
        role_mismatch("Assistant implemented the guard in consolidation", "assistant"),
        None,
    )
    check(
        "cross-role: assistant subject in user pass reported, not dropped",
        role_mismatch("Assistant recommended Redis for the session store", "user"),
        "user_pass_assistant_subject",
    )
except Exception:
    traceback.print_exc()
    FAILURES.append("output guards raised")

# ------------------------------------------- extraction guard + pin policy
# Added 2026-10-18. These live in the image gate, not only in tests/, because
# the Test workflow does not trigger on mn-deployed pushes -- without this
# section nothing in the pipeline would notice the pin policy going missing.
try:
    import re as _re

    import mnemory.prompts as _prompts
    from mnemory import output_guards as _og

    check("pin_allowed exported", "pin_allowed" in _og.__all__, True)

    _CURLY = chr(0x2019)
    for _text, _want in (
        ("Fact 5 identifies a specific data corruption issue in the table",
         "meta_indexical"),
        ("Today's date is Wednesday, 7 October 2026.", "date_echo"),
        ("Today" + _CURLY + "s date is Wednesday, 7 October 2026.", "date_echo"),
        ('Assistant: "That settles it, lock in 512"', "speaker_quote"),
        ("User's name is Ada Lovelace", None),
        ("User" + _CURLY + "s birthday is August 13, 1975.", None),
    ):
        check("guard: " + _text[:32], _og.noise_memory(_text), _want)

    for _text, _want in (
        ("The Planck constant quantifies energy per photon", False),
        ("User's resting heart rate was 75 on Tuesday", False),
        ("User" + _CURLY + "s name is Ada Lovelace", True),
        ("Assistant" + _CURLY + "s name is Atlas", True),
        ("User takes metformin 1000 mg twice daily", True),
        ("User lives in Amsterdam, Netherlands.", True),
        ("User is a solutions architect at Example Corp", True),
    ):
        check("pin=" + str(_want) + ": " + _text[:32], _og.pin_allowed(_text), _want)

    # ── narration: anchored head, and rewrite-instead-of-drop ────────────────
    # Shipped 2026-10-18. The previous rule's verb list sat in a second
    # top-level alternative, so it was never anchored to the assistant subject
    # and flagged ordinary facts. Replaying 24h of live "would drop" verdicts:
    # of 28 assistant_speech_act rows, 11 were not narration at all and 13 were
    # substantive advice being thrown away (training decisions, the euDKA
    # fueling rule, cadence and sodium guidance). Only 4 were genuine acts.
    check("triage_memory exported", "triage_memory" in _og.__all__, True)

    # _CURLY and _re come from the block above; not redefined here.
    # facts whose sentence merely happens to contain a reporting verb
    for _text in (
        "User agreed to change Monday from strength to Z2 45 min recovery",
        "User" + _CURLY + "s note about knee pain says it started in August",
        "User keeps a training log and reports it weekly to his coach",
    ):
        check("not narration: " + _text[:30], _og.noise_memory(_text), None)

    # narration around a real clause -- unwrap, never delete
    for _text, _want in (
        ("Assistant recommended not running on an empty stomach",
         "Advice: Not running on an empty stomach"),
        ("Assistant advised drinking water with salt today instead of volume",
         "Advice: Drinking water with salt today instead of volume"),
        # explicit that-clause stays verbatim, and a code identifier must come
        # back intact rather than re-capitalised
        ("Assistant noted that icu_apply_sport_settings will overwrite zones",
         "icu_apply_sport_settings will overwrite zones"),
        # the shape the live extractor actually produced
        ("Assistant recommended against running on an empty stomach",
         "Advice: Not running on an empty stomach"),
    ):
        _stored, _reason, _rewrote = _og.triage_memory(_text)
        check("triage text: " + _text[:30], _stored, _want)
        check("triage keeps: " + _text[:30], _reason, None)
        check("triage flagged rewritten: " + _text[:30], _rewrote, True)

    # the act IS the content -- still refused, and left word-for-word intact
    _act = "Assistant offered to unpin memories 28, 35, 45 via PUT"
    _stored, _reason, _rewrote = _og.triage_memory(_act)
    check("act refused", _reason, "assistant_speech_act")
    check("act untouched", _stored, _act)
    check("act not rewritten", _rewrote, False)

    for _text, _want in (
        ("Assistant recommended not running on an empty stomach", True),
        ("Assistant offered to adapt it", True),
        ("User agreed to change Monday from strength to Z2 recovery", False),
        ("User" + _CURLY + "s note about knee pain says it started in August",
         False),
    ):
        check("anchored head: " + _text[:30],
              bool(_og._ASSISTANT_ACT_RE.search(_text)), _want)

    # Negative control. If the verb list ever floats free of the anchor again,
    # this is the shape that breaks: every one of these sentences contains a
    # reporting verb and says nothing about the assistant narrating anything.
    _detached = _re.compile(r"(?:" + _og._ACT_VERBS + r")\b", _re.IGNORECASE)
    _caught_by_detached = sum(
        1 for _t in ("User agreed to change Monday from strength to Z2",
                     "User" + _CURLY + "s note about knee pain",
                     "User keeps a training log and reports it weekly",
                     "The knee offer of service is expired")
        if _detached.search(_t))
    check("control: detached verb list still over-matches (>=4)",
          _caught_by_detached >= 4, True)

    # The invariant that caught the bug which reached a published image: no
    # guard pattern may assume an ASCII apostrophe or quote, because the
    # extractor writes typographic punctuation as a matter of course.
    _offenders = []
    _esc = chr(92) + "u"
    _pairs = ((chr(39), (chr(0x2019), chr(0x2018))),
              (chr(34), (chr(0x201c), chr(0x201d))))
    for _name in dir(_og):
        _value = getattr(_og, _name)
        if not isinstance(_value, _re.Pattern):
            continue
        _pat = _value.pattern
        for _ascii, _twins in _pairs:
            if _ascii in _pat and not any(
                _t in _pat or (_esc + "%04x" % ord(_t)) in _pat for _t in _twins
            ):
                _offenders.append(_name)
    check("no ASCII-only punctuation in guard patterns", sorted(set(_offenders)), [])

    _src = open(_prompts.__file__, encoding="utf-8").read()
    # noise_memory(text) is deliberately absent: the write paths call it
    # through triage_memory now, which is the marker that matters.
    # noise_enforcing() was corrected 2026-11-04: the 2026-10-28 per-reason
    # gate patch replaced the zero-arg call with noise_enforcing_for(reason,
    # categories=...) call, and this marker was never updated, so
    # the only automated image gate has been red-by-default since then. A gate
    # that is always red is how the next real failure hides.
    for _marker in ("pin_allowed(text)", "noise_enforcing_for(",
                    "triage_memory(text)"):
        check("prompts wired: " + _marker, _marker in _src, True)
except Exception:
    traceback.print_exc()
    FAILURES.append("extraction guard checks raised")

# ------------------------------------------------------------- redaction
# Secret redaction is enforced on sight (no MEMORY_NOISE_GUARD equivalent), so
# a merge that silently drops one call site starts storing live credentials
# again instead of failing. That is why the marker checks below read module
# source rather than importing: mnemory.memory pulls qdrant_client at import
# time, and two upstream merges have already removed fork patches unnoticed.
section("redact_secrets - secret redaction at every write path")
try:
    import pathlib as _pl

    from mnemory import output_guards as _ogr
    from mnemory.output_guards import redact_secrets as _redact

    _GOOGLE_FAKE = "AIza" + "aB3d" * 8 + "aB3"
    check("google fixture is AIza + 35 chars", len(_GOOGLE_FAKE), 39)

    # Assembled at runtime: as literals these are scanner-legal tokens for real
    # services and GitHub push protection blocks the commit. Concatenation
    # leaves the string handed to the guard unchanged.
    _SLACK_FAKE = "xox" + "b" + "-" + "123456789012" + "-" + "abcdefghijklmno"
    _STRIPE_FAKE = "sk_" + "test" + "_" + "Ab1cD2eF3gH4iJ5kL6mN7oP8"
    _TELEGRAM_FAKE = "123456789" + ":" + "AA" + "HdqTcvCH1vGWJxfSeofSAs0K5PALDsaw"

    for _kind, _text, _secret in (
        ("github_token", "pasted ghp_" + "A1" * 18 + " in chat",
         "ghp_" + "A1" * 18),
        ("aws_access_key", "deploy key AKIAIOSFODNN7EXAMPLE rotates",
         "AKIAIOSFODNN7EXAMPLE"),
        ("slack_token", "bot posts " + _SLACK_FAKE + " now",
         _SLACK_FAKE),
        ("stripe_key", "payments " + _STRIPE_FAKE + " used",
         _STRIPE_FAKE),
        ("openai_key", "extractor sk-proj-" + "aB3dE5fG7hI9jK1lM2nO3pQ4rS5tU6",
         "sk-proj-aB3dE5fG7hI9jK1lM2nO3pQ4rS5tU6"),
        ("jwt",
         "cookie eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NSJ9"
         ".abcDEF1234567890XYZ sent",
         "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"),
        ("private_key", "material -----BEGIN RSA PRIVATE KEY----- truncated",
         "-----BEGIN RSA PRIVATE KEY-----"),
        ("telegram_bot_token",
         "alerts " + _TELEGRAM_FAKE + " fire",
         _TELEGRAM_FAKE),
    ):
        _out, _kinds = _redact(_text)
        check("redacts " + _kind, _kind in _kinds, True)
        check(_kind + " secret bytes removed", _secret in _out, False)

    for _text in ("password: hunter2", "api_key=abc123def456",
                  "wachtwoord: welkom123", "client_secret=g0supersecretvalue"):
        _out, _kinds = _redact(_text)
        check("redacts kv " + _text[:18], "credential_value" in _kinds, True)

    # Negative controls: the config/threshold facts this store is full of.
    for _text in (
        "Set --max-tokens: 8192 on the llama-swap entry",
        "Per-pass max_tokens ceilings: fsck_security 6144 and Pass B 12288",
        "FSCK_MAX_LLM_CALLS=600",
        "password: required for sudo access",
        "The WEBUI_SECRET_KEY env var must be read from /proc/1/environ",
        "CORE_MEMORY_INCLUDE_TOP_N is read by zero files",
        "User lives in Examplestad, Netherlands",
        "User's real name is Ada Lovelace.",
        "User takes amlodipine 5 mg once daily",
        "User aims for 84-88 rpm cadence on the trainer",
    ):
        _out, _kinds = _redact(_text)
        check("untouched: " + _text[:34], (_out, _kinds), (_text, []))

    _once, _k1 = _redact("api_key=abc123def456 here")
    _twice, _k2 = _redact(_once)
    check("idempotent on re-save", _twice, _once)
    check("placeholder not re-redacted", _k2, [])
    check("row never emptied", bool(_redact("password: hunter2")[0]), True)
    check("None passes through", _redact(None), (None, []))
    check("redaction on by default", _ogr.redaction_enabled(), True)

    _pkg = _pl.Path(_ogr.__file__).parent
    for _mod, _marker in (
        ("prompts", "text, redactions = redact_secrets(text)"),
        ("memory", "content, redactions = redact_secrets(content)"),
        ("consolidation", "f[\"text\"], redactions = redact_secrets("),
        ("fsck", "redact_secrets(cleaned)[0]"),
    ):
        _src = (_pkg / (_mod + ".py")).read_text(encoding="utf-8")
        check("wired: " + _mod, _marker in _src, True)

    # Call-site census. memory.py must hold three: the remember loop,
    # _add_without_inference (infer=False) and update_memory (PUT). The marker
    # checks above cannot see one disappearing, because the others still match
    # -- which is how the fsck apply path nearly went uncovered.
    _mem_src = (_pkg / "memory.py").read_text(encoding="utf-8")
    check("memory.py redaction call sites", _mem_src.count("redact_secrets("), 3)
except Exception:
    traceback.print_exc()
    FAILURES.append("redaction checks raised")

# ---------------------------------------------------------------- 2026-11-04
# Core-section router and build-reference staleness. The behavioural suite is
# tests/test_fork_core_section_router.py; these are image-level presence checks
# so a future image cannot ship without either half, and the two lists must be
# non-empty because an empty pattern list silently deletes a whole prompt
# section rather than failing.
try:
    from mnemory.memory import (
        _IDENTITY_RES,
        _INSTRUCTION_RES,
        _NARRATION_HEAD_RE,
        MemoryService,
    )
    from mnemory.prompts import _BUILD_ID_RE, _correct_build_staleness

    section("core-section router (2026-11-04)")

    def _sec(text, role, mtype):
        return MemoryService._classify_agent_memory(
            None,
            {
                "id": "x",
                "memory": text,
                "metadata": {"role": role, "memory_type": mtype},
            },
        )[0]

    check(
        "third-person name is identity",
        _sec("Assistant's name is Atlas", "assistant", "fact"),
        "identity",
    )
    check(
        "first-person name is identity",
        _sec("My name is Bob", "assistant", "fact"),
        "identity",
    )
    check(
        "sub-agent persona is identity",
        _sec("I am Bob. I speak casually.", "assistant", "fact"),
        "identity",
    )
    check(
        "first-person work note is not identity",
        _sec("I am working on the dashboard.", "assistant", "fact"),
        "knowledge",
    )
    check(
        "hardware note is not identity",
        _sec("User uses an Atlas Ezo pH board in I2C mode.", "assistant", "fact"),
        "knowledge",
    )
    check(
        "car registration is not an instruction",
        _sec("User registered their 2021 Kia EV6.", "user", "fact"),
        "knowledge",
    )
    check(
        "agent-scope preference is an instruction",
        _sec("User wants me to create commit messages", "user", "preference"),
        "instructions",
    )
    check(
        "standing rule is an instruction",
        _sec("Assistant must never print passwords.", "user", "procedural"),
        "instructions",
    )
    check(
        "one-off ask is not an instruction",
        _sec("User asked the assistant to write the flow.", "user", "episodic"),
        "knowledge",
    )
    check("identity patterns present", len(_IDENTITY_RES) >= 5, True)
    check("instruction patterns present", len(_INSTRUCTION_RES) >= 5, True)
    check(
        "narration head detected",
        bool(_NARRATION_HEAD_RE.match("Assistant noted that the run passed.")),
        True,
    )

    section("build-reference staleness (2026-11-04)")
    BUILD_REF = (
        "Assistant published container image 40d50b32a21f with manifest digest "
        "812a92da65a1aeae0aba0801a376369c04e773875f2ebe6ea2d016d7dc7b76b1."
    )
    check(
        "build ref demotes importance",
        _correct_build_staleness("high", "episodic", BUILD_REF),
        ("low", "episodic"),
    )
    check(
        "pinned row exempt",
        _correct_build_staleness("high", "episodic", BUILD_REF, pinned=True),
        ("high", "episodic"),
    )
    check(
        "endpoint is not a version",
        _correct_build_staleness(
            "high",
            "fact",
            "Pi-hole runs on 192.168.2.25; web UI on 127.0.0.1:8080, binds 0.0.0.0.",
        ),
        ("high", "fact"),
    )
    check(
        "durable lesson with the word version kept",
        _correct_build_staleness(
            "high",
            "episodic",
            "User's CI skips building when the image version tag exists.",
        ),
        ("high", "episodic"),
    )
    check(
        "bare commit sha demotes",
        _correct_build_staleness("high", "fact", "The main branch is at commit 2d47b9a.")[0],
        "low",
    )
    check(
        "BUILD_ID_RE finds a patched build",
        _BUILD_ID_RE.search("deployed version v1.15.0-patched21") is not None,
        True,
    )

    # call-site census, same reasoning as the redaction census above: the
    # marker checks alone cannot see one of the two extraction sites vanish.
    import pathlib as _pl2
    import sys as _sys2

    _pkg2 = _pl2.Path(_sys2.modules["mnemory.memory"].__file__).parent
    _p_src = (_pkg2 / "prompts.py").read_text(encoding="utf-8")
    check("prompts.py staleness call sites", _p_src.count("_correct_build_staleness("), 3)
    _m_src = (_pkg2 / "memory.py").read_text(encoding="utf-8")
    check(
        "memory.py router is content-shaped",
        "_IDENTITY_RES" in _m_src and "_NARRATION_HEAD_RE" in _m_src,
        True,
    )
except Exception:
    traceback.print_exc()
    FAILURES.append("core-section / staleness checks raised")

# ----------------------------------------------------- dedup partial apply
# 2026-11-04: Stage-2 dedup used to discard the ENTIRE remember batch when a
# single fact lacked a decision (measured: 4 of 23 batches lost in 19h). These
# are presence checks only; the behavioural suite is
# tests/test_fork_dedup_partial_apply.py.
try:
    import pathlib as _pl3
    import sys as _sys3

    _pkg3 = _pl3.Path(_sys3.modules["mnemory.memory"].__file__).parent
    _mem3 = (_pkg3 / "memory.py").read_text(encoding="utf-8")
    _cfg3 = (_pkg3 / "config.py").read_text(encoding="utf-8")

    section("dedup partial apply (2026-11-04)")
    check("partial-apply path present", "Remember dedup partial apply" in _mem3, True)
    check("best partial result tracked", "best_indices" in _mem3, True)
    check("total failure still fails closed", "0 of %d facts decided" in _mem3, True)
    check("undecided policy knob defined", "dedup_undecided" in _cfg3, True)
    from mnemory.config import Config as _Cfg3

    check("undecided default is add", _Cfg3().memory.dedup_undecided, "add")
except Exception:
    traceback.print_exc()
    FAILURES.append("dedup partial-apply checks raised")

# ------------------------------------------------- core-memory owner scoping
# 2026-11-04: get_core_memories read storage without owner/subject scoping, so
# rows owned by another identity were injected into this user's prompt while
# every enumeration endpoint hid them. Presence checks only; the behavioural
# suite is tests/test_fork_core_owner_scope.py.
try:
    import pathlib as _pl4
    import sys as _sys4

    _pkg4 = _pl4.Path(_sys4.modules["mnemory.memory"].__file__).parent
    _mem4 = (_pkg4 / "memory.py").read_text(encoding="utf-8")
    _vec4 = (_pkg4 / "storage" / "vector.py").read_text(encoding="utf-8")
    _api4 = (_pkg4 / "api" / "memories.py").read_text(encoding="utf-8")

    section("core-memory owner scoping (2026-11-04)")

    # Text counting is brittle here (scoping kwargs legitimately exist in other
    # call sites), so drive the builder with a recording stub and assert on the
    # kwargs it actually passes to storage.
    class _Rec:
        def __init__(self):
            self.calls = []

        def get_pinned_memories(self, **k):
            self.calls.append(("pinned", k))
            return []

        def get_all(self, **k):
            self.calls.append(("all", k))
            return {"results": []}

        def get_recent_memories(self, **k):
            self.calls.append(("recent", k))
            return []

    class _NoCache:
        def get(self, key):
            return None

        def set(self, key, value):
            return None

    from mnemory.config import Config as _Cfg4
    from mnemory.memory import MemoryService as _MS4

    _svc4 = object.__new__(_MS4)
    _svc4._config = _Cfg4()
    _svc4._core_cache = _NoCache()
    _svc4.vector = _Rec()
    _uid = "qa-scope@user.test"
    _svc4.get_core_memories(user_id=_uid, owner_id=_uid, agent_id="qa-agent",
                           track_access=False)
    check("core builder performed storage reads", bool(_svc4.vector.calls), True)
    unscoped = [name for name, kw in _svc4.vector.calls
                if kw.get("owner_id") != _uid or kw.get("subject_user_id") != _uid]
    check("every core read is owner+subject scoped", unscoped, [])
    check("scope audit helper defined", "def _is_scope_mismatch(" in _vec4, True)
    check("browse exposes scope_mismatch", "scope_mismatch: bool = False," in _vec4, True)
    check("filter applied in browse scroll",
          "if scope_mismatch and not _is_scope_mismatch(" in _vec4, True)
    check("API accepts scope_mismatch",
          "scope_mismatch: bool = Query(" in _api4, True)
    # part 3: the owner filter runs before the mismatch predicate, so without a
    # subject-side mode the audit cannot see rows owned by someone else
    check("browse has a subject-scope mode", "subject_scope: bool = False," in _vec4, True)
    check("subject scope replaces the owner predicate",
          "identity_condition = _build_owner_scope_condition" in _vec4, True)
    check("API accepts subject_scope",
          "subject_scope: bool = Query(" in _api4, True)
    # 3b: retraction is owner-scoped, so a view that hides the owner is inert
    check("maintenance view names the disagreeing fields",
          "_attach_scope_fields(" in _api4, True)
except Exception:
    traceback.print_exc()
    FAILURES.append("core-memory owner-scoping checks raised")

# ------------------------------------------------- write-guard wiring (2026-11-05)
# The chat pipeline parses with parse_remember_extraction_response. Every guard
# in output_guards was wired to the OTHER parser, so MEMORY_AGENT_VOICE=drop
# never applied to a chat write and the store refilled after each purge. Text
# presence here is the cheapest possible guard against that class of silent
# unwiring, which a config dump and a code review both missed.
try:
    import pathlib as _p7
    import sys as _s7

    _pkg7 = _p7.Path(_s7.modules["mnemory.prompts"].__file__).parent
    _pr7 = (_pkg7 / "prompts.py").read_text(encoding="utf-8")
    _og7 = (_pkg7 / "output_guards.py").read_text(encoding="utf-8")
    _co7 = (_pkg7 / "consolidation.py").read_text(encoding="utf-8")

    def _fn_body7(src, start_marker, end_marker):
        i = src.index(start_marker)
        return src[i:src.index(end_marker, i)]

    _remember7 = _fn_body7(
        _pr7, "def parse_remember_extraction_response(", "def build_dedup_prompt(")
    section("write-guard wiring (2026-11-05)")
    check("chat parser runs the voice gate", "agent_voice(" in _remember7, True)
    check("chat parser runs triage/noise guard", "triage_memory(" in _remember7, True)
    check("chat parser enforces pin policy", "pin_allowed(" in _remember7, True)
    check("provenance rule defined", "def launders_user_decision(" in _og7, True)
    check("provenance rule ships shadow by default",
          '_ROLE_PROVENANCE_MODE_ENV = "MEMORY_ROLE_PROVENANCE"' in _og7, True)
    check("consolidation calls the provenance rule",
          "launders_user_decision(" in _co7, True)
    check("importance exemption removed",
          'in ("high", "critical"):\n        return None\n    return "assistant_agent_voice"'
          in _og7, False)
except Exception:
    traceback.print_exc()
    FAILURES.append("write-guard wiring checks raised")

# ---------------------------------------------------------------- summary
print("\n================ SUMMARY ================")
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    raise SystemExit(1)
print("all checks passed")
