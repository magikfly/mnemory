"""Unit tests for mnemory.output_guards.

Runnable three ways: ``python3 tests/test_output_guards.py``, ``pytest
tests/test_output_guards.py``, or via the container QA gate (deploy/mn_qa.py
carries the same assertions). Pure functions, no service, no store.

Every positive case is an artifact actually observed in a production memory
store; every negative case is real text that must survive untouched.
"""

import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnemory.output_guards import clean_memory_text, role_mismatch  # noqa: E402

CASES_CLEAN = [
    # (input, expected)
    # --- artifacts observed in the live store, must be removed
    (
        "- id=0: Assistant recommended explicitly disabling the healthcheck for mnemory",
        "Assistant recommended explicitly disabling the healthcheck for mnemory",
    ),
    (
        "- [type: fact | categories: technical, project:mnemory | importance: "
        "high | role: assistant] Assistant decided to enable automatic merge for PRs",
        "Assistant decided to enable automatic merge for PRs",
    ),
    (
        "\u27e8memory_item\u27e9User prefers conventional commit messages\u27e8/memory_item\u27e9",
        "User prefers conventional commit messages",
    ),
    ("[S3]: Assistant rebuilt the image with mcp pinned", "Assistant rebuilt the image with mcp pinned"),
    ("- - User runs every other day and lifts on the intervening days",
     "User runs every other day and lifts on the intervening days"),
    # --- clean text must pass through byte-identical
    ("User lives in Zoetermeer, Netherlands", "User lives in Zoetermeer, Netherlands"),
    ("Assistant set the port to 8050 [mnemory] in the compose file",
     "Assistant set the port to 8050 [mnemory] in the compose file"),
    ("Assistant noted that --n-gpu-layers keeps weights resident in VRAM",
     "Assistant noted that --n-gpu-layers keeps weights resident in VRAM"),
    ("S3 lifecycle rules were configured for the backup bucket",
     "S3 lifecycle rules were configured for the backup bucket"),
    ("The array [10] indexes the tenth element", "The array [10] indexes the tenth element"),
    # --- nothing usable left
    ("- id=0:", None),
    ("   ", None),
    ("", None),
    ("store_artifact: false", None),
]

CASES_ROLE = [
    ("User decided to enable reasoning in the agent", "assistant", "assistant_pass_user_subject"),
    ("User's wife is named Lisandra Fiorini", "assistant", "assistant_pass_user_subject"),
    ("the user asked for a summary of the run", "assistant", "assistant_pass_user_subject"),
    ("Assistant implemented the guard in consolidation", "assistant", None),
    ("User decided to rebuild the image", "user", None),
    ("Assistant recommended Redis for the session store", "user", "user_pass_assistant_subject"),
    ("", "assistant", None),
    (None, "assistant", None),
]


def test_clean_memory_text():
    for text, want in CASES_CLEAN:
        got = clean_memory_text(text)
        assert got == want, f"clean({text!r})\n  got  {got!r}\n  want {want!r}"


def test_role_mismatch():
    for text, role, want in CASES_ROLE:
        got = role_mismatch(text, role)
        assert got == want, f"role_mismatch({text!r}, {role!r}) got {got!r} want {want!r}"


def test_fsck_scope_flavour_is_stripped():
    """The fsck dedup prompt emitted "[scope: ... | type: ... | created: ...]".

    Regression guard: the original regex keyed on "type:" only, so this whole
    flavour - 75 of 116 stored blocks on 2026-10-09 - passed straight through.
    """
    raw = ("User's deployed image contains 7 instances of a token "
           "[scope: open-webui | type: episodic | categories: project:mnemory | "
           "importance: high | event_date: 2026-10-07T22:00:00+00:00 | created: 2026-10-08]")
    out = clean_memory_text(raw)
    assert out is not None
    assert "scope:" not in out and "created:" not in out and "importance:" not in out
    assert out.startswith("User's deployed image contains 7 instances of a token")


def test_alias_prefix_with_pinned_flag():
    """build_fsck_duplicate_prompt now writes "- id=0 (pinned): text"; the alias
    prefix guard has to accept the flag, or the new format becomes the echo."""
    out = clean_memory_text(
        "- id=0 (pinned): Assistant prefers pinned tests [scope: shared | type: fact]")
    assert out == "Assistant prefers pinned tests", out


def test_ordinary_brackets_survive():
    for text in [
        "User runs llama.cpp with [S1] as the source alias reference line",
        "The 3rd array index [10] is out of range on the qdrant collection",
        "Wiki style [[memory_item]] links should stay visible in the text body",
        "The bracketed word [type] without a colon is ordinary prose here ok",
    ]:
        assert clean_memory_text(text) == text, text


def test_fsck_prompt_emits_no_inline_metadata():
    """The root-cause fix: metadata must not sit next to memory text in the prompt."""
    from mnemory.prompts import build_fsck_duplicate_prompt
    cluster = [
        {"id": "a" * 36, "memory": "User prefers conventional commit messages for git.",
         "agent_id": "open-webui",
         "metadata": {"memory_type": "preference", "importance": "high", "pinned": True,
                      "categories": ["git"], "created_at_utc": "2026-10-01T00:00:00+00:00",
                      "event_date": "2026-10-01"}},
        {"id": "b" * 36, "memory": "User prefers conventional commit messages for git.",
         "agent_id": None, "metadata": {"memory_type": "preference"}},
    ]
    out = build_fsck_duplicate_prompt(cluster)
    blob = str(out)
    assert "[scope:" not in blob and " | created:" not in blob, "inline metadata block is back"
    assert "(pinned)" in blob, "pinned flag should still reach the model"
    assert "- id=0 (pinned): User prefers" in blob or "id=0 (pinned)" in blob


def test_fsck_write_paths_are_guarded():
    """fsck.py must import the guard and use it at both model-authored writes.

    Checked at source level, not by import: this module is deliberately free of
    third-party dependencies so it runs without qdrant_client installed.
    """
    src = (pathlib.Path(__file__).resolve().parents[1] / "mnemory" / "fsck.py").read_text()
    assert "from mnemory.output_guards import clean_memory_text" in src
    assert src.count("clean_memory_text(") >= 2, "a fsck write path lost its guard"



if __name__ == "__main__":
    import sys as _sys, inspect as _inspect, traceback as _tb
    names = sorted(k for k, v in globals().items()
                   if k.startswith("test_") and _inspect.isfunction(v))
    fails = 0
    for n in names:
        try:
            globals()[n](); print("PASS", n)
        except Exception:
            fails += 1; print("FAIL", n); _tb.print_exc()
    print("\n%d tests, %d failures" % (len(names), fails))
    _sys.exit(1 if fails else 0)
