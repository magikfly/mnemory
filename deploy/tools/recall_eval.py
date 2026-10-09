# Sanitized for a public repository: configure via environment.
#   MN_BASE_URL (default http://localhost:8050), MN_API_KEY, MN_USER_ID, MN_AGENT_ID
import os as _os
BASE = _os.environ.get("MN_BASE_URL", "http://localhost:8050")
API_KEY = _os.environ.get("MN_API_KEY", "")
USER_ID = _os.environ.get("MN_USER_ID", "")
AGENT_ID = _os.environ.get("MN_AGENT_ID", "open-webui")

#!/usr/bin/env python3
"""Retrieval quality harness for mnemory search.

Purpose: decide SEARCH_KEYWORD_WEIGHT by measurement instead of opinion. Search
currently runs with SEARCH_SIMILARITY_WEIGHT=0.9 / SEARCH_KEYWORD_WEIGHT=0.0,
i.e. pure embedding similarity with lexical matching switched off, which is the
suspected cause of topically-unrelated-but-unique rows surviving recall.

Method: fixed query set with expected-answer substrings derived only from store
rows already verified this session (never guessed), plus greeting probes that
should return nothing. Reports hit@1/3/8, MRR, and how many rows come back for
short turns. Diff this snapshot before and after a config change; if the two are
byte-identical, the lever is inert (most likely existing points carry no sparse
vectors) and the change must be reverted rather than kept on faith.

  python3 search_eval.py            # run and write a snapshot
  python3 search_eval.py a.json b.json   # diff two snapshots
"""
import json
import sys
import urllib.request

HDRS = {
    "Authorization": "Bearer " + API_KEY,
    "X-User-Id": USER_ID,
    "X-Agent-Id": "open-webui",
    "Content-Type": "application/json",
}
LIMIT = 8

# Expectations use only text confirmed present in the store during this session.
QUERIES = [
    ("what is the assistant's name", ["atlas"]),
    ("what is my real name", ["chairil"]),
    ("where do I live", ["zoetermeer"]),
    ("what is my wife's name", ["lisandra"]),
    ("what car do I drive", ["ev6"]),
    ("what git commit message style do I prefer", ["conventional"]),
    ("which language should the assistant reply in", ["language the user addresses", "match their input", "dutch in"]),
    ("how large is my investment portfolio", ["500,000"]),
    ("what DNS server runs on my network", ["pi-hole"]),
    ("what is my maximum heart rate", ["188"]),
    ("what heart rate is my Z5 zone", ["174"]),
    ("did the mnemory filter static context bug get fixed", ["static-context", "static context"]),
    ("which layer do core memories get built from", ["consolidated"]),
    ("when does the vault backup run", ["backup"]),
    # Short-turn probes: correct behaviour is few or no results.
    ("thanks", []),
    ("goedemorgen", []),
    ("do it", []),
]


def search(query):
    req = urllib.request.Request(
        f"{BASE}/api/memories/search",
        data=json.dumps({"query": query, "limit": LIMIT}).encode(),
        headers=HDRS,
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode()).get("results") or []


def evaluate():
    rows = []
    for query, expects in QUERIES:
        hits = search(query)
        texts = [(m.get("memory") or "").lower() for m in hits]
        scores = [m.get("score") for m in hits]
        ranks = [
            next((i + 1 for i, t in enumerate(texts) if any(e in t for e in expects)), None)
            for _ in [0]
        ][0]
        rows.append(
            {
                "query": query,
                "expects": expects,
                "returned": len(hits),
                "rank": ranks,
                "top_score": round(scores[0], 4) if scores and scores[0] is not None else None,
                "top": [t[:110] for t in texts[:3]],
            }
        )
    return rows


def summarize(rows, label):
    scored = [r for r in rows if r["expects"]]
    n = len(scored)
    hit = lambda k: sum(1 for r in scored if r["rank"] and r["rank"] <= k)
    mrr = sum((1 / r["rank"]) for r in scored if r["rank"]) / max(1, n)
    print(f"\n=== {label} ===")
    for r in rows:
        mark = "n/a" if not r["expects"] else (str(r["rank"]) if r["rank"] else "MISS")
        print(f"  @{mark:>4} n={r['returned']:2d} {r['query'][:44]:44s} "
              f"{(r['top'][0] if r['top'] else '')[:58]}")
    print(f"  hit@1 {hit(1)}/{n}  hit@3 {hit(3)}/{n}  hit@8 {hit(8)}/{n}  MRR {mrr:.3f}")
    noise = [r for r in rows if not r["expects"]]
    print("  short-turn returns: " + ", ".join(f"{r['query']}={r['returned']}" for r in noise))
    return {"hit1": hit(1), "hit3": hit(3), "hit8": hit(8), "mrr": round(mrr, 4), "n": n}


def main():
    if len(sys.argv) == 3:
        a = json.load(open(sys.argv[1]))
        b = json.load(open(sys.argv[2]))
        print("=== diff ===")
        for ra, rb in zip(a["rows"], b["rows"]):
            if ra["top"] != rb["top"] or ra["rank"] != rb["rank"]:
                print(f"  CHANGED {ra['query'][:40]:40s} rank {ra['rank']} -> {rb['rank']}")
        print("  before:", a["summary"], "\n  after: ", b["summary"])
        if a["rows"] == b["rows"]:
            print("  IDENTICAL: the lever did nothing - revert it, do not keep an")
            print("  unverified setting (existing points likely lack sparse vectors).")
        return

    rows = evaluate()
    summary = summarize(rows, "before-snapshot")
    out = "./search_eval_before.json"
    json.dump({"summary": summary, "rows": rows}, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
