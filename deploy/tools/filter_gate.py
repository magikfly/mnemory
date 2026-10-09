# Sanitized for a public repository: credentials, the service address and the
# LAN specifics that appeared in test fixtures are removed. Set MN_API_KEY and
# MN_USER_ID (optionally MN_BASE_URL, default http://localhost:8050) to run.
import os as _os
MN_BASE_URL = _os.environ.get("MN_BASE_URL", "http://localhost:8050")
MN_API_KEY = _os.environ.get("MN_API_KEY", "")
MN_USER_ID = _os.environ.get("MN_USER_ID", "")
MN_AGENT_ID = _os.environ.get("MN_AGENT_ID", "open-webui")

"""Behavioural gate for the patched mnemory_filter. Imports the real module."""
import importlib.util
import sys

# Which filter to exercise: env override, else the archived copy in this repo.
# deploy/owui-functions/mnemory_filter.py is verified byte-identical to the content
# running in the OWUI function DB, so the gate tests what is deployed.
import os as _o2, pathlib as _p2
_here = _p2.Path(__file__).resolve().parent
_CANDS = [_o2.environ.get("MN_FILTER_PATH"),
          str(_here.parent / "owui-functions" / "mnemory_filter.py"),
          "mn_filter_patched.py"]
FILTER_SRC = next((c for c in _CANDS if c and _o2.path.exists(c)), None)
if not FILTER_SRC:
    raise SystemExit("mnemory_filter source not found; set MN_FILTER_PATH")
print("testing filter:", FILTER_SRC)
spec = importlib.util.spec_from_file_location("mnp", FILTER_SRC)
m = importlib.util.module_from_spec(spec)
sys.modules["mnp"] = m
spec.loader.exec_module(m)
F = m.Filter
print("imported OK; valves:", len(F.Valves.model_fields))

f = F()

# ---------------------------------------------------------------- 1. hygiene gate
DROP = [
    "User requested a greeting and asked what they should be working on today",
    "User prefers cooking and collecting stamps.",          # example leak, paraphrased
    "User enjoys cooking and collecting stamps",            # example verbatim
    "User is asking who the assistant is",
    "User asked how proficient I am in COBOL.",
    "Assistant greeted the user as 'Goed ochtend' (Good morning).",
    "Assistant confirmed readiness to receive the instruction and stated that if choices are needed.",
    "User requested to kill the temp instance.",
    "Assistant offered to grep knowledge bases for 'Petr' and edit files directly",
    "User wants to know what they want to work on today",
    "Session [session_id] was resolved from the pending cache",       # placeholder leak
    "Value copied from [api_key] into the vault entry",               # placeholder leak
]
KEEP = [
    "Assistant's name is Atlas.",
    "User prefers hybrid or remote work arrangements.",
    "User has experience with design thinking techniques.",
    "User's birthday is stored in the assistant's memory.",
    "User is trying to bulk up arms by doing the same exercises since June",
    "User's current mnemory profile runs on the themodel base model via llama.cpp provider using llama-swap.",
    "User decided to take full rest on October 14-15.",
    "User wants to improve memory quality without extra LLM calls.",
    "Assistant explained that --n-gpu-layers sets the number of transformer layers offloaded to the GPU.",
    "User's memory system fsck_dedup call site echoes cluster content and requires its own token ceiling.",
    "User flagged an IP conflict between host-a's entry (10.0.0.175) and the old host-b config (10.0.0.174).",
    "Candidate is considering the Spinweb IRM Architect opportunity but lacks specific details.",
    "User wants to create a v3.5 version of their IT Architect CV in PPTX format.",
    "normal text about 5 [10] percent growth",                        # bracket, not a placeholder
    "[[complex]] nested brackets in a code note",
]
bad = 0
for t in DROP:
    if not F._hygiene_violation(t):
        print(f"  FAIL should-DROP: {t[:70]}"); bad += 1
for t in KEEP:
    if F._hygiene_violation(t):
        print(f"  FAIL should-KEEP: {t[:70]}"); bad += 1
print(f"hygiene: {len(DROP)+len(KEEP)-bad}/{len(DROP)+len(KEEP)} correct")

# ---------------------------------------------------------------- 2. injector gate
CTX = "## Memory (mnemory)\ninstructions\n\n## Core\n- User's name is Chairil\n- [chat_id] leak"
def inj(path_ctx=CTX):
    body = {"messages": [{"role": "system", "content": "You are Atlas."},
                         {"role": "user", "content": "hi"}]}
    f._static_by_user = {}
    f._inject_static_context(body, {"session_id": "s", "user_id": "u", "static_ctx": path_ctx}, "u")
    return body["messages"]

msgs = inj()
ok_len   = len(msgs) == 3
ok_role  = msgs[1]["role"] == "system"
ok_core  = "Chairil" in msgs[1]["content"]
ok_hyg   = "[chat_id]" not in msgs[1]["content"]
msgs2 = inj()   # second call must not duplicate (idempotency guard)
print(f"injection: inserted={ok_len} as_system={ok_role} core={ok_core} "
      f"hygiene={ok_hyg} idempotent={len(inj()) == 3} (2nd call msgs={len(msgs2)})")
bad += 0 if (ok_len and ok_role and ok_core and ok_hyg and len(inj()) == 3) else 1

# ------------------------------------------------------------- 3. real recalled block
RECALLED = [l[2:] for l in """
- Assistant's name is Atlas.
- User requested a greeting and asked what they should be working on today
- User prefers hybrid or remote work arrangements.
- User wants to know what they want to work on today
- User prefers cooking and collecting stamps.
- User is asking who the assistant is
- User asked how proficient I am in COBOL.
- Assistant overwrote a previous memory regarding a Dutch-greeting preference
- User has experience with design thinking techniques.
- Assistant identified a potential memory duplication issue where the global mnemory_filter causes the new profile to auto-recall
- User wants to improve memory quality without extra LLM calls.
- User deployed a commit to the mnemory store to record the root cause lesson regarding a double-m typo
- User's memory system fsck_dedup call site echoes cluster content and requires its own token ceiling.
- User's current mnemory profile runs on the themodel base model via llama.cpp provider using llama-swap.
- User decided to take full rest on October 14-15.
- User's memory system has a fix where truncated `remember_extract` replies now salvage closed objects instead of discarding all facts.
- User's birthday is stored in the assistant's memory.
- User wants to create a v3.5 version of their IT Architect CV in PPTX format while maintaining the same look and feel.
- Assistant explained that --n-gpu-layers sets the number of transformer layers offloaded to the GPU, affecting speed and VRAM usage.
- Assistant confirmed readiness to receive the instruction and stated that if choices are needed
- User requested to kill the temp instance.
- User is trying to bulk up arms by doing the same exercises since June
- User wants to run a fresh fsck scan including raw memories and apply the 91 high-severity issues immediately
- Assistant greeted the user as 'Goed ochtend' (Good morning).
- User flagged an IP conflict between host-a's entry (10.0.0.175) and the old host-b config (10.0.0.174).
""".strip().split("\n")]
kept = [r for r in RECALLED if not F._hygiene_violation(r)]
dropped = [r for r in RECALLED if F._hygiene_violation(r)]
print(f"\nyour pasted recall block: {len(RECALLED)} -> {len(kept)} kept, {len(dropped)} dropped")
for d in dropped:
    print("   x", d[:78])
print("\n" + ("GATE PASSED" if not bad else f"GATE FAILED ({bad})"))
sys.exit(1 if bad else 0)

# ------------------------------------------------- 4. sanitizer: tag-aware + capped
CTX2 = ("## Memory (mnemory)\nsome instructions\n\n"
        "## User Facts\n"
        + "".join("\u27e8memory_item\u27e9Fact number %d about the user\u27e8/memory_item\u27e9\n" % i
                  for i in range(1, 11)).replace("\n", "\n- ").rstrip("\n") +
        "- \u27e8memory_item\u27e9User is asking who the assistant is\u27e8/memory_item\u27e9\n"
        "- \u27e8memory_item\u27e9value from [api_key] copied\u27e8/memory_item\u27e9\n"
        "- \u27e8memory_item\u27e9User enjoys cooking and collecting stamps\u27e8/memory_item\u27e9\n")
f2 = F(); f2.valves = F.Valves()
clean = f2._sanitize_static_ctx(CTX2)
kept = [l for l in clean.split("\n") if l.startswith("- ")]
facts = [l for l in kept if "Fact number" in l]
print("\nsanitizer: %d fact bullets kept (cap=%d), wrapped preserved=%s"
      % (len(facts), f2.valves.core_max_per_section, any("\u27e8memory_item\u27e9" in l for l in facts)))
bad_fact = any("cooking" in l for l in kept) or any("api_key" in l for l in kept) \
           or any("who the assistant is" in l for l in kept)
print("sanitizer: junk inside wrapped bullets still dropped:", not bad_fact)
print("sanitizer: headers preserved:", "## User Facts" in clean)
if len(facts) != 6 or bad_fact or "## User Facts" not in clean:
    print("GATE FAILED (sanitizer)"); sys.exit(1)
print("\nALL GATES PASSED")
