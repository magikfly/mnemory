#!/usr/bin/env python3
"""Sweep score_threshold on the 'search' recall path using the fixed query set.

Why this instead of the keyword weight: SEARCH_KEYWORD_WEIGHT is deprecated and
ignored, and 'find' mode costs +3.4s and an LLM call per turn. score_threshold is
a per-request override (schemas.py), free in latency and calls, and it is the
only knob that decides whether a short turn gets 8 rows or none.

Two opposing metrics are reported per threshold:
  good_answered  how many of the 14 real questions still get their answer
                 through the gate (hit@8) - dropping means collateral damage
  junk_short     rows returned for 'thanks' / 'goedemorgen' / 'do it' - the
                 complaint. Lower is better.
Fresh session per query, so this models a new chat; the carry bucket is a
separate mechanism and is not affected by this knob.
"""
import json
import time
import urllib.request

from recall_eval import QUERIES

BASE = BASE
HDRS = {
    "Authorization": "Bearer " + API_KEY,
    "X-User-Id": USER_ID,
    "X-Agent-Id": "open-webui",
    "Content-Type": "application/json",
}
THRESHOLDS = [0.5, 0.6, 0.65, 0.7, 0.75]
LIMIT = 8


def recall(query, threshold):
    body = {
        "session_id": f"thr-{threshold}-{int(time.time()*1000)}",
        "query": query,
        "search_mode": "search",
        "score_threshold": threshold,
    }
    req = urllib.request.Request(f"{BASE}/api/recall", data=json.dumps(body).encode(), headers=HDRS)
    with urllib.request.urlopen(req, timeout=120) as r:
        return (json.loads(r.read().decode()).get("search_results")) or []


def run(threshold):
    rows = []
    for query, expects in QUERIES:
        hits = recall(query, threshold)
        texts = [(m.get("memory") or "").lower() for m in hits]
        rank = next((i + 1 for i, t in enumerate(texts) if any(e in t for e in expects)), None)
        rows.append({"query": query, "expects": expects, "rank": rank, "returned": len(hits),
                     "top": texts[0][:60] if texts else ""})
    return rows


def summarize(threshold, rows):
    scored = [r for r in rows if r["expects"]]
    short = [r for r in rows if not r["expects"]]
    n = len(scored)
    hit = lambda k: sum(1 for r in scored if r["rank"] and r["rank"] <= k)
    mrr = sum((1 / r["rank"]) for r in scored if r["rank"]) / max(1, n)
    junk = sum(r["returned"] for r in short)
    print(f"\n=== threshold {threshold} ===")
    print(f"  answered@1 {hit(1)}/{n}  @3 {hit(3)}/{n}  @8 {hit(8)}/{n}  "
          f"MRR {mrr:.3f}  junk_short {junk}/{3*LIMIT}")
    lost = [r["query"] for r in scored if not r["rank"]]
    if lost:
        print("  no answer through the gate: " + "; ".join(q[:40] for q in lost))
    print("  short-turn returns: " + ", ".join(f"{r['query']}={r['returned']}" for r in short))
    return {"threshold": threshold, "hit1": hit(1), "hit3": hit(3), "hit8": hit(8),
            "mrr": round(mrr, 3), "junk_short": junk, "lost": lost,
            "short": {r["query"]: r["returned"] for r in short}}


if __name__ == "__main__":
    out = []
    for t in THRESHOLDS:
        out.append(summarize(t, run(t)))
    json.dump(out, open("./search_eval_thresholds.json", "w"), indent=2)
    print("\n=== roll-up ===")
    print(f"{'thr':>5} {'hit@1':>6} {'hit@3':>6} {'hit@8':>6} {'MRR':>6} {'junk':>5}")
    for r in out:
        print(f"{r['threshold']:>5} {r['hit1']:>6} {r['hit3']:>6} {r['hit8']:>6} "
              f"{r['mrr']:>6} {r['junk_short']:>5}")
    print("wrote ./search_eval_thresholds.json")
