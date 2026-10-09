"""Unit tests for mnemory.output_guards.

Runnable three ways: ``python3 tests/test_output_guards.py``, ``pytest
tests/test_output_guards.py``, or via the container QA gate (deploy/mn_qa.py
carries the same assertions). Pure functions, no service, no store.

Every positive case is an artifact actually observed in a production memory
store; every negative case is real text that must survive untouched.
"""

import os
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


if __name__ == "__main__":
    failures = 0
    for fn in (test_clean_memory_text, test_role_mismatch):
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {fn.__name__}: {exc}")
    raise SystemExit(1 if failures else 0)
