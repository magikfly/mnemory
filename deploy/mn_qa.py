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

# ---------------------------------------------------------------- summary
print("\n================ SUMMARY ================")
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    raise SystemExit(1)
print("all checks passed")
