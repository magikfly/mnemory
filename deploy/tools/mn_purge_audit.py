#!/usr/bin/env python3
"""Read-only audit of one user's mnemory store, with optional bulk repair.

Why this exists
---------------
`/api/memories` reports `has_more: false` at its cap and ignores `cursor`,
`offset` and `page`, so a naive listing silently truncates and the store looks
unexportable. It is not: `limit` accepts 5000 and the list is partitionable by
`memory_type`, `memory_layer` and `include_decayed`, and the union of those
partitions is a complete dump. Verified 2026-10-09: 2675 unique rows for one
user, 94% of the count `/api/stats` reports for them (the remainder are
superseded revisions the list endpoint does not return).

What the audit found on that store, in decreasing order of surprise
-------------------------------------------------------------------
1. Prompt-scaffolding echoes. 113 stored rows contained a serialised metadata
   block; 111 of them were the fsck dedup prompt's own `[scope: ... | type: ...
   | created: ...]` format, because a 4B copier reproduces whatever sits next to
   the text it is asked to rewrite. Root-caused in commit 5fe0193.
2. Retraction is not deletion and there is no un-retract. `DELETE /api/memories
   /{id}` stamps `revision_state: retracted`, and `_active_revision_condition`
   filters it out of every read, so purging works; recovering means re-adding
   from a backup, which is why this tool writes one before `--apply`.
3. "Delete the expired ones" is not free. 329 of 733 decayed rows were cited by
   consolidated rows via `derived_from` / `lineage_id`. Deleting those breaks
   provenance and re-consolidation, so they are classified `review`, never
   `delete`. `evidence_root_ids` is empty in the list projection - use the other
   two keys, or the check silently finds nothing and looks reassuring.
4. Exact duplicates are detectable: `metadata.fact_hash` groups them (24 groups,
   50 rows here). Keep the newest per hash.
5. Ranking, not storage, limits quality: two greeting-narration rows scored 0.823
   for `goedemorgen` while real answers scored under 0.60, so no single threshold
   separates them. See recall_eval_threshold.py, which measured that sweep.

Usage
-----
    export MN_BASE_URL=http://host:8050 MN_API_KEY=... MN_USER_ID=... MN_AGENT_ID=open-webui
    python3 mn_purge_audit.py                 # dry run, prints the tier table
    python3 mn_purge_audit.py --csv plan.csv  # also writes per-row actions
    python3 mn_purge_audit.py --apply delete --yes
    python3 mn_purge_audit.py --apply strip --yes

Pinned rows are never touched. The X-User-Id header scopes every call, so other
users' rows cannot be reached from here at all.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = os.environ.get("MN_BASE_URL", "http://localhost:8050")
API_KEY = os.environ.get("MN_API_KEY", "")
USER_ID = os.environ.get("MN_USER_ID", "")
AGENT_ID = os.environ.get("MN_AGENT_ID", "open-webui")

HDRS = {
    "Authorization": "Bearer " + API_KEY,
    "X-User-Id": USER_ID,
    "X-Agent-Id": AGENT_ID,
    "Content-Type": "application/json",
}

# ---- predicates -------------------------------------------------------------
ART_BLOCK = re.compile(r"\[\s*(?:scope|type)\s*:[^\]]{0,400}?\]", re.I)
ART_OTHER = re.compile(r"categories:\s*\w+\s*\||\bid=0\b", re.I)
NARRATION = re.compile(
    r"\b(user (inquired|is asking|was asking|wanted to know|asked how|thanked|requested to)"
    r"|assistant (greeted|acknowledged|offered to|provided a summary|stated that|clarified"
    r"|confirmed readiness|is ready))\b", re.I)
GREETING = re.compile(r"assistant greeted", re.I)
POINTER = re.compile(
    r"stored (as|in) (a|the) (pinned|consolidated|separate)|details \(model name"
    r"|see (the )?(pinned|memory)|pointer (row|memory)", re.I)
# Rows that document a bug are not the bug. Without this the classifier deletes
# its own diagnostics - it did exactly that on the first run.
DIAGNOSTIC = re.compile(
    r"lost during consolidation|is not being retrieved|contaminated row|instances of"
    r"|indicating a (leak|s)", re.I)
LINEAGE_KEYS = ("derived_from", "supersedes", "lineage_id")


def meta(row, key, default=None):
    return (row.get("metadata") or {}).get(key, default)


def parse_ts(value):
    if not value:
        return None
    try:
        d = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def is_decayed(row):
    now = dt.datetime.now(dt.timezone.utc)
    if parse_ts(meta(row, "decayed_at")):
        return True
    exp = parse_ts(meta(row, "expires_at"))
    return exp is not None and exp < now


def get(path):
    req = urllib.request.Request(BASE + path, headers=HDRS)
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read().decode())


def send(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers=HDRS, method=method)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, r.read().decode()


def harvest():
    """Partitioned full dump. One call is not enough - see the docstring."""
    rows, parts = {}, []
    for sort in ("newest", "oldest"):
        for layer in (None, "raw", "consolidated"):
            q = f"/api/memories?limit=5000&include_decayed=true&sort={sort}"
            parts.append(q + (f"&memory_layer={layer}" if layer else ""))
    for mt in ("episodic", "context", "procedural", "fact", "preference"):
        parts.append(f"/api/memories?limit=5000&include_decayed=true&memory_type={mt}")
    for q in parts:
        try:
            for r in (get(q).get("results") or []):
                rows.setdefault(r["id"], r)
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            print(f"  partition failed ({exc}): {q[:60]}", file=sys.stderr)
    return list(rows.values())


def classify(rows):
    """Return {id: (action, reason)}. Actions: delete_*, strip, review, keep."""
    lineage = set()
    for r in rows:
        if meta(r, "memory_layer") != "consolidated":
            continue
        for k in LINEAGE_KEYS:
            v = meta(r, k) or []
            lineage |= {x for x in ([v] if isinstance(v, str) else v) if isinstance(x, str)}

    groups = collections.defaultdict(list)
    for r in rows:
        fh = meta(r, "fact_hash")
        if fh:
            groups[fh].append(r)
    dup_old = set()
    for members in groups.values():
        if len(members) > 1:
            keep_newest = sorted(members, key=lambda x: parse_ts(meta(x, "created_at_utc")) or dt.datetime.min.replace(tzinfo=dt.timezone.utc))
            dup_old |= {m["id"] for m in keep_newest[:-1]}

    plan = {}
    for r in rows:
        rid, text = r["id"], (r.get("memory") or "")
        if meta(r, "pinned"):
            plan[rid] = ("keep", "pinned")
            continue
        if DIAGNOSTIC.search(text):
            plan[rid] = ("review", "documents a defect; not itself rot")
        elif GREETING.search(text):
            plan[rid] = ("delete_greeting", "greeting narration")
        elif rid in dup_old:
            plan[rid] = ("delete_dup", "exact duplicate by fact_hash, newest kept")
        elif ART_BLOCK.search(text) and len(ART_BLOCK.sub("", text).strip()) < 20:
            plan[rid] = ("delete_artifact", "scaffolding only, nothing left after strip")
        elif POINTER.search(text) and len(text) < 260:
            plan[rid] = ("delete_pointer", "pointer row, carries no payload")
        elif NARRATION.search(text) and len(NARRATION.sub("", ART_BLOCK.sub("", text)).strip()) < 24:
            plan[rid] = ("delete_narration", "speech act only, no fact survives")
        elif is_decayed(r) and rid in lineage:
            plan[rid] = ("review", "expired but cited as consolidated lineage")
        elif is_decayed(r):
            plan[rid] = ("delete_expired", "expired and not cited as lineage")
        elif ART_BLOCK.search(text) or ART_OTHER.search(text):
            rest = ART_OTHER.sub("", ART_BLOCK.sub("", text)).strip()
            plan[rid] = ("strip" if len(rest) >= 30 else "delete_artifact",
                         "metadata block stored inside text")
        elif NARRATION.search(text):
            plan[rid] = ("review", "narration with a residual fact - rewrite, do not delete")
        else:
            plan[rid] = ("keep", "")
    return plan


def clean(text):
    """Same normalisation the server guard applies. Kept inline so this tool has
    no dependency on an installed mnemory package."""
    out = ART_BLOCK.sub(" ", text)
    out = ART_OTHER.sub(" ", out)
    out = re.sub(r"^\s*(?:[-*\u2022]\s+)+(?:\(\s*)?id\s*=\s*\d+\s*(?:\))?\s*(?:\(\s*pinned\s*\))?\s*[:\u2013-]\s*", "", out)
    out = re.sub(r"\u27e8/?[a-z_]*\u27e9", "", out)
    return re.sub(r"\s{2,}", " ", out).strip().strip("-\u2014").strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", metavar="PATH", help="write per-row actions")
    ap.add_argument("--apply", choices=("delete", "strip"), help="mutate the store")
    ap.add_argument("--yes", action="store_true", help="required with --apply")
    ap.add_argument("--limit", type=int, default=0, help="act on at most N rows")
    args = ap.parse_args()
    if not API_KEY or not USER_ID:
        raise SystemExit("set MN_API_KEY and MN_USER_ID")
    if args.apply and not args.yes:
        raise SystemExit("--apply needs --yes")

    rows = harvest()
    print(f"enumerated {len(rows)} rows for {USER_ID[:4]}…")
    plan = classify(rows)
    by = {r["id"]: r for r in rows}
    counts = collections.Counter(a for a, _ in plan.values())
    dele = sorted(i for i, (a, _) in plan.items() if a.startswith("delete"))
    strip = sorted(i for i, (a, _) in plan.items() if a == "strip")

    print("\naction           n")
    for a in sorted(counts):
        print(f"{a:15s} {counts[a]:5d}")
    print(f"\ndeletable {len(dele)}   strippable {len(strip)}   "
          f"store after delete {len(rows) - len(dele)}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["action", "reason", "id", "type", "layer", "importance", "role", "text"])
            for i in sorted(plan, key=lambda x: (plan[x][0], x)):
                r = by[i]
                w.writerow([plan[i][0], plan[i][1], i, meta(r, "memory_type"),
                            meta(r, "memory_layer"), meta(r, "importance"), meta(r, "role"),
                            (r.get("memory") or "")[:300]])
        print("wrote", args.csv)

    if not args.apply:
        print("\ndry run; nothing sent. Add --apply delete|strip --yes to act.")
        return

    backup = f"purge_backup_{int(time.time())}.json"
    targets = (dele if args.apply == "delete" else strip)[:args.limit or None]
    json.dump({i: by[i] for i in targets}, open(backup, "w"))
    print(f"backed up {len(targets)} rows to {backup}")

    ok = collections.Counter()
    for n, i in enumerate(targets, 1):
        try:
            if args.apply == "delete":
                ok[send("DELETE", f"/api/memories/{i}")[0]] += 1
            else:
                c = clean(by[i].get("memory") or "")
                if not c or len(c) > 1000 or len(c) < 20:
                    ok["skipped"] += 1
                    continue
                ok[send("PUT", f"/api/memories/{i}", {"content": c})[0]] += 1
        except urllib.error.HTTPError as exc:
            ok[exc.code] += 1
        except Exception:
            ok["error"] += 1
        if n % 50 == 0:
            print(f"  {n}/{len(targets)} {dict(ok)}", flush=True)
        time.sleep(0.05)
    print("done:", dict(ok))


if __name__ == "__main__":
    main()
