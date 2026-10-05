#!/usr/bin/env python3
"""Behavioural test for the mn-fork rendering patch.

The deployed filter needs aiohttp + pydantic, which aren't installed here,
so we lift the FORK-PATCH(2/4) helper block straight out of the patched
file and exercise the real code, not a re-typed copy of it.
"""
import time

PATCHED = "/home/user/mnemory-fork/deployed/mnemory_filter.v0.4.2.mn-fork.py"
src = open(PATCHED).read()
start = src.index("    # ── FORK-PATCH(2/4)")
end = src.index("    def _carry_recalled(")
block = src[start:end]

ns = {"time": time}
exec(compile("class Harness:\n" + block, "<fork-patch>", "exec"), ns)
Harness = ns["Harness"]


class Valves:
    recall_currency_labels = True


h = Harness()
h.valves = Valves()

today = time.strftime("%Y-%m-%d")
cases = [
    ("typed + dated (the Shuffle case)",
     {"memory": "User plans to perform a 'Shuffle' run on the next training session",
      "metadata": {"memory_type": "episodic", "event_date": "2026-09-28T18:00:00+00:00"}}),
    ("typed, fact, no date",
     {"memory": "User prefers concise system prompts",
      "metadata": {"memory_type": "preference"}}),
    ("core-memory style with created stamp",
     {"memory": "User swapped the Llama Swap entry to Qwen3.5-4B-Q2_K_XL",
      "metadata": {"memory_type": "episodic", "event_date": "2026-10-03T22:00:00+00:00",
                   "created": "2026-10-05"}}),
    ("NO metadata — the failure mode",
     {"memory": "User plans to integrate llama-swap functionality after deployment"}),
    ("epoch int stamp must not crash",
     {"memory": "User adjusted the DFlash n_max setting to 12",
      "metadata": {"memory_type": "decision", "created_at": 1791000000}}),
    ("garbage stamp must not crash",
     {"memory": "User wants wg-easy", "metadata": {"memory_type": "fact",
                                                   "event_date": "not-a-date"}}),
    ("empty text", {"memory": "", "metadata": {"memory_type": "fact"}}),
]

print("=" * 78)
for label, item in cases:
    print(f"{label}\n  -> {h._memory_line(item)!r}")

print("=" * 78)
h.valves.recall_currency_labels = False
print("valve OFF ->",
      repr(h._memory_line({"memory": "User plans a Shuffle run",
                           "metadata": {"memory_type": "episodic",
                                        "event_date": "2026-09-28"}})))
print("age math today =", today, "| 7d ago ->",
      h._age_days(time.strftime("%Y-%m-%d", time.localtime(time.time() - 7 * 86400))))
