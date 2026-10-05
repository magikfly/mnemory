#!/usr/bin/env python3
"""Build the mn-fork patch for the deployed mnemory_filter v0.4.2.

Reads the baseline (extracted verbatim from the Open WebUI function DB),
applies surgical edits, writes the patched file + a unified diff.
Every edit is marked FORK-PATCH in-source so it never blurs into the
user's own LOCAL-PATCH conventions.
"""
import difflib
import pathlib
import re

BASE = pathlib.Path("/home/user/mnemory_filter_deployed.py").read_text()
src = BASE

edits = []


def sub(old, new, label):
    global src
    n = src.count(old)
    assert n == 1, f"{label}: anchor found {n} times (expected 1)"
    src = src.replace(old, new, 1)
    edits.append(label)


# ── 1. Valve so this is switchable ────────────────────────────────────
sub(
    '        recall_carry_max: int = Field(',
    '''        # FORK-PATCH(1/4): currency labels on recalled memories.
        recall_currency_labels: bool = Field(
            default=True,
            description=(
                "Tag each recalled memory with its type and as-of date "
                "(plus age in days) so the model can tell a dated plan "
                "from a standing fact. Set False to restore the bare "
                "upstream text-only rendering."
            ),
        )
        recall_carry_max: int = Field(''',
    "valve recall_currency_labels",
)

# ── 2. Helpers ────────────────────────────────────────────────────────
HELPERS = '''    # ── FORK-PATCH(2/4): carry provenance into the recall block ───────
    def _memory_line(self, item: dict) -> str:
        """Render one search result as `text [type: … | as of: … (Nd ago)]`.

        Upstream injects m["memory"] alone, so a plan recorded months ago
        arrives as a bare present-tense assertion with nothing indicating
        it is stale — that is exactly the shape a model reads as "now".
        Items with no usable metadata are labelled `type: untyped` so the
        absence of provenance is at least visible to the reader.
        """
        text = (item.get("memory") or "").strip()
        if not text:
            return ""
        if not self.valves.recall_currency_labels:
            return text
        meta = item.get("metadata") or {}
        bits: list[str] = []
        mtype = meta.get("memory_type") or meta.get("type")
        if mtype:
            bits.append(f"type: {mtype}")
        stamp = meta.get("event_date") or meta.get("created_at") or meta.get("created")
        age = self._age_days(stamp)
        if age is not None:
            if isinstance(stamp, (int, float)):
                day = time.strftime("%Y-%m-%d", time.localtime(float(stamp)))
            else:
                day = str(stamp)[:10]
            bits.append(f"as of: {day} ({age}d ago)")
        if not bits:
            return f"{text} [type: untyped]"
        return f"{text} [{' | '.join(bits)}]"

    @staticmethod
    def _age_days(stamp) -> int | None:
        """Whole days between a memory timestamp and today; None if unusable."""
        if not stamp:
            return None
        if isinstance(stamp, (int, float)):
            return max(0, int((time.time() - float(stamp)) // 86400))
        s = str(stamp)
        for fmt, cut in (("%Y-%m-%dT%H:%M:%S", 19), ("%Y-%m-%d", 10)):
            try:
                parsed = time.strptime(s[:cut], fmt)
            except ValueError:
                continue
            return max(0, int((time.time() - time.mktime(parsed)) // 86400))
        return None

    def _carry_recalled('''
sub("    def _carry_recalled(", HELPERS, "helper methods _memory_line/_age_days")

# ── 3. Carry loop: append rendered line, hygiene-check the raw text ───
sub(
    '''            bucket.append(text)''',
    '''            # FORK-PATCH(3/4): metadata rides with the text into the bucket
            bucket.append(self._memory_line(m))''',
    "carry loop renders metadata",
)

# ── 4a. Non-carry path: same rendering ────────────────────────────────
sub(
    '''                [m["memory"] for m in result["search_results"] if m.get("memory")],''',
    '''                # FORK-PATCH(4a): same rendering as the carry path
                [
                    self._memory_line(m)
                    for m in result["search_results"]
                    if m.get("memory")
                ],''',
    "non-carry path renders metadata",
)

# ── 4b. Header: tell the reader what the tags mean ────────────────────
sub(
    '''        block = "\\n\\n## Recalled Memories\\n" + "\\n".join(f"- {m}" for m in memories)''',
    '''        # FORK-PATCH(4b): header states the semantics of the tags.
        if self.valves.recall_currency_labels:
            head = (
                "\\n\\n## Recalled Memories — every line is tagged `type:` and "
                "`as of:` (date, age in days). These are records of past "
                "sessions and of past states of the system. A plan carrying "
                "an old date is history, not a live commitment; a config or "
                "path detail may already be superseded. Do not state any of "
                "them as current without verifying current state.\\n"
            )
        else:
            head = "\\n\\n## Recalled Memories\\n"
        block = head + "\\n".join(f"- {m}" for m in memories)''',
    "recall header semantics",
)

out_dir = pathlib.Path("/home/user/mnemory-fork/deployed")
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "mnemory_filter.v0.4.2.baseline.py").write_text(BASE)
(out_dir / "mnemory_filter.v0.4.2.mn-fork.py").write_text(src)

diff = "".join(
    difflib.unified_diff(
        BASE.splitlines(keepends=True),
        src.splitlines(keepends=True),
        fromfile="deployed/mnemory_filter.v0.4.2.baseline.py",
        tofile="deployed/mnemory_filter.v0.4.2.mn-fork.py",
    )
)
(out_dir / "mn-fork.patch").write_text(diff)

print("edits applied:")
for e in edits:
    print("  -", e)
print("baseline lines:", BASE.count("\n"), "-> patched lines:", src.count("\n"))
print("diff size:", len(diff), "chars")
