#!/usr/bin/env python3
"""Deploy the mn-fork filter into Open WebUI (dry-run by default).

Baseline is fetched from the live function, backed up timestamped, and only
replaced when --apply is passed.

  python3 deploy_to_owui.py            # preview
  python3 deploy_to_owui.py --apply    # deploy

API key from netvault services.openwebui.bearer. Port 3001, NOT 3000 — 3000 is
a different Open WebUI instance that rejects this key (401 api-key.invalid).
After deploying, re-read the function to confirm the content landed.
"""
import argparse
import datetime
import pathlib
import sys

import requests

BASE = "http://192.168.2.1:3001"
FUNC_ID = "mnemory_filter"
API_KEY = "sk-5ab4756f2cdb4cbcafc5f49d482c2b80"
HERE = pathlib.Path(__file__).resolve().parent
PATCHED = HERE / "mnemory_filter.v0.4.2.mn-fork.py"
BACKUPS = HERE / "backups"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually write the function")
    ap.add_argument("--base", default=BASE, help="Open WebUI base URL")
    args = ap.parse_args()

    new_src = PATCHED.read_text()
    hdr = {"Authorization": f"Bearer {API_KEY}"}

    r = requests.get(f"{args.base}/api/v1/functions/id/{FUNC_ID}", headers=hdr, timeout=30)
    if r.status_code != 200:
        print(f"GET failed: {r.status_code} {r.text[:200]}")
        return 1
    current = r.json()
    old_src = current.get("content", "")

    BACKUPS.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = BACKUPS / f"mnemory_filter.{stamp}.py"
    bak.write_text(old_src)

    print(f"live version  : {current.get('meta', {}).get('version', '?')}")
    print(f"live source   : {len(old_src)} chars")
    print(f"fork source   : {len(new_src)} chars")
    print(f"backup written: {bak}")
    print(f"delta         : {len(new_src) - len(old_src):+d} chars")

    if "FORK-PATCH(1/4)" in old_src:
        print("\nAlready patched. Nothing to do.")
        return 0

    if not args.apply:
        print("\nDRY RUN — pass --apply to deploy.")
        return 0

    payload = dict(current)
    payload["content"] = new_src
    meta = dict(current.get("meta") or {})
    meta["version"] = f"{meta.get('version', '0.4.2')}+mnfork"
    payload["meta"] = meta
    # These are server-managed fields; the update endpoint rejects stale copies.
    for k in ("id", "user_id", "created_at", "updated_at"):
        payload.pop(k, None)

    u = requests.put(f"{args.base}/api/v1/functions/id/{FUNC_ID}", headers=hdr, json=payload, timeout=30)
    print(f"\nPUT -> {u.status_code}")
    if u.status_code != 200:
        print("response:", u.text[:400])
        print(f"restore with: cp {bak} and PUT, or paste into the Open WebUI function editor")
        return 1

    verify = requests.get(f"{args.base}/api/v1/functions/id/{FUNC_ID}", headers=hdr, timeout=30).json()
    ok = "FORK-PATCH(1/4)" in verify.get("content", "")
    print(f"re-read confirms patch present: {ok}")
    print(f"version now: {verify.get('meta', {}).get('version', '?')}")
    print("\nFilters are re-imported per request, so the next chat turn uses it.")
    print("Test in a throwaway chat before trusting a real one.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
