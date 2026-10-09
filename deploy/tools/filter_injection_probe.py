# Sanitized for a public repository: credentials, the service address and the
# LAN specifics that appeared in test fixtures are removed. Set MN_API_KEY and
# MN_USER_ID (optionally MN_BASE_URL, default http://localhost:8050) to run.
import os as _os
MN_BASE_URL = _os.environ.get("MN_BASE_URL", "http://localhost:8050")
MN_API_KEY = _os.environ.get("MN_API_KEY", "")
MN_USER_ID = _os.environ.get("MN_USER_ID", "")
MN_AGENT_ID = _os.environ.get("MN_AGENT_ID", "open-webui")

#!/usr/bin/env python3
"""show_injected_context.py — what the mnemory filter actually puts in the prompt.

Runs the DEPLOYED function content out of the OWUI sqlite DB (not a local copy)
and executes the real inlet() against live mnemory, so it answers "is it working?"
without asking a model to introspect on its own system prompt.

Usage:
  python3 <workdir>/show_injected_context.py      # print + gate
  python3 <workdir>/show_injected_context.py -v    # dump the probe only

Always gates; prints PASS/FAIL per check and exits 1 if any check fails, so it
is safe to drop in cron. "core memories empty" is a WARN, not a failure - it is
correct output when nothing in the store is pinned.
Persists across OpenTerminal recreates (bind mount); the probe itself runs
inside the OpenWebUI container, which is where aiohttp and the DB live.
"""
import base64
import subprocess
import sys

PROBE = r'''
import asyncio, json, re, sqlite3, sys, time, types

DB = "/app/backend/data/webui.db"
row = sqlite3.connect(DB).execute(
    "select content, valves from function where id='mnemory_filter'"
).fetchone()
if not row:
    print("mnemory_filter not found in DB"); sys.exit(2)

mod = types.ModuleType("deployed"); sys.modules["deployed"] = mod
exec(compile(row[0], "<deployed>", "exec"), mod.__dict__)

f = mod.Filter()
f.valves = mod.Filter.Valves(**json.loads(row[1] or "{}"))
u = {"id": "probe", "email": MN_USER_ID, "name": "Chairil", "role": "admin"}

JUNK = ["cooking and collecting stamps", "how proficient I am in COBOL",
        "is asking who the assistant is", "requested a greeting",
        "greeted the user as", "confirmed readiness", "requested to kill the temp"]

async def main():
    body = await f.inlet(
        {"model": "x", "chat_id": "probe-%d" % int(time.time()),
         "messages": [{"role": "system", "content": "You are Atlas."},
                      {"role": "user", "content": "hello, who am I and what are my hobbies?"}]},
        __user__=u, __event_emitter__=None)
    msgs = body["messages"]
    blob = "\n".join(m.get("content") or "" for m in msgs)

    print("roles:", [m["role"] for m in msgs])
    print("valves: threshold=%s carry_max=%s" % (f.valves.recall_score_threshold,
                                                 f.valves.recall_carry_max))
    if len(msgs) > 1 and msgs[1]["role"] == "system":
        print("\n===== INJECTED STATIC CONTEXT (%d chars) =====" % len(msgs[1]["content"]))
        print(msgs[1]["content"])
        print("===== END =====\n")

    m = re.search(r"## Recalled Memories\n(.*)", blob, re.S)
    lines = [l[2:] for l in (m.group(1).split("\n") if m else []) if l.startswith("- ")]
    print("recalled: %d" % len(lines))
    for l in lines:
        print("   -", l[:100])

    # core sections and whether each has content under it
    print("\ncore-memory sections:")
    for sec in ("Agent Identity", "Agent Knowledge", "User Facts",
                "User Preferences", "Other User Memories"):
        blk = re.search(r"## %s\n(.*?)(?=\n## |\Z)" % re.escape(sec), blob, re.S)
        n = len([x for x in (blk.group(1).split("\n") if blk else []) if x.strip()])
        print("   %-22s %d line(s)" % (sec, n))

    checks = {
        "instructions injected": "## Memory (mnemory)" in blob,
        "no placeholder leaks":  not re.search(r"\[[a-z][a-z0-9]*(?:_[a-z0-9]+)+\]", blob),
    }
    for j in JUNK:
        checks["no junk: " + j[:26]] = j.lower() not in blob.lower()
    print()
    bad = [k for k, v in checks.items() if not v]
    for k, v in checks.items():
        print("   %s %s" % ("PASS" if v else "FAIL", k))
    core_empty = all(
        not [x for x in (re.search(r"## %s\n(.*?)(?=\n## |\Z)" % re.escape(sec), blob, re.S)
                         .group(1).split("\n")) if x.strip()]
        for sec in ("Agent Identity", "User Facts", "User Preferences"))
    print("   WARN core memories empty (nothing pinned in the store)" if core_empty
          else "   core memories populated")
    sys.exit(1 if bad else 0)

asyncio.run(main())
'''

if "--verbose" in sys.argv or "-v" in sys.argv:
    print(PROBE)
    sys.exit(0)

b64 = base64.b64encode(PROBE.encode()).decode()
cmd = (f"printf '%s' '{b64}' > /tmp/sic.b64 && base64 -d /tmp/sic.b64 > /tmp/sic.py && "
       f"docker cp /tmp/sic.py OpenWebUI:/tmp/sic.py && docker exec OpenWebUI python3 /tmp/sic.py")
r = subprocess.run(["python3", "<ssh-helper>", cmd, "150"])
sys.exit(r.returncode)
