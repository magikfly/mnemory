# mn-fork — my edits to mnemory

Fork of [fpytloun/mnemory](https://github.com/fpytloun/mnemory) (cloned at
`95f359c`, "feat: add managed inference routing and service authentication")
carrying the Open WebUI filter patches that live on the homeserver.

Branch `mn-fork`. My edits are tagged `FORK-PATCH(n)` in-source so they never
blur into the user's own `LOCAL-PATCH` convention.

---

## The bug this fixes

One chat in which the assistant told the user "the next session is the Shuffle"
— a plan from a months-old conversation, presented as a live calendar fact.
The Intervals calendar was empty. The assistant had asserted a stale memory as
current state.

**Root cause is not the extractor, not the quant, and not the memory server.**
It is one line in the Open WebUI filter, verified in the deployed v0.4.2
source pulled from the Open WebUI function DB:

```python
# _carry_recalled()
for m in search_results or []:
    text = m.get("memory")          # <- metadata is on the same dict, unused
    ...
    bucket.append(text)

# _inject_recalled()
block = "\n\n## Recalled Memories\n" + "\n".join(f"- {m}" for m in memories)
```

`mnemory/api/schemas.py:967 format_memory_item()` returns `metadata` on every
search hit — `memory_type`, `event_date`, `scope`, `importance`, `created`.
The filter throws all of it away. The server-rendered *core memories*
(static context) keep their `[scope: … | type: episodic | event_date: …]`
suffixes; the *dynamic recall block* is bare prose. So the model receives
two visually different classes of statement, and the unlabelled ones are
exactly the ones that read as "now".

Observable in a single recall block: the extractor-swap entry arrives tagged
(`type: episodic | event_date: 2026-10-03`), while "User plans to integrate
llama-swap functionality after deployment" — long since completed — arrives
with no marker at all, next to "User wants to switch from llama-swap to
NINfer", which is superseded by the running system.

## The patch

Five edits, all in `mnemory_filter.py`, all switchable by one valve:

1. **Valve `recall_currency_labels` (default True)** — turn the whole thing off
   to restore upstream byte-for-byte behaviour.
2. **`_memory_line(item)` + `_age_days(stamp)`** — render
   `text [type: episodic | as of: 2026-09-28 (6d ago)]`. Handles ISO dates with
   tz offsets, bare dates, epoch ints, and junk stamps (junk → no `as of`,
   never an exception). Untagged items get `[type: untyped]` so the *absence*
   of provenance is visible instead of invisible.
3. **Carry loop** appends the rendered line; the hygiene filter still runs on
   the raw text, so it can't be fooled by the suffix.
4. **Non-carry path** renders identically, so behaviour doesn't fork on
   `recall_carry_max`.
5. **Header semantics** — the block header states what the tags mean and
   instructs the reader: a plan with an old date is history, not a commitment;
   a config detail may be superseded; verify current state before asserting.

### Rendered result

```
User plans to perform a 'Shuffle' run on the next training session [type: episodic | as of: 2026-09-28 (6d ago)]
User plans to integrate llama-swap functionality after deployment [type: untyped]
User swapped the Llama Swap entry to Qwen3.5-4B-Q2_K_XL [type: episodic | as of: 2026-10-03 (1d ago)]
User prefers concise system prompts [type: preference]
```

## Cost

~40 characters and ~10 tokens per line. At `recall_carry_max = 40` that is
roughly 400 tokens per turn, plus ~70 for the header. No API calls added, no
storage schema change, no re-index, no migration.

## Deliberate scope limits

- **Fixes readability, not storage.** An untyped `fact` memory stays
  untyped in the database; the patch makes the gap visible at injection time.
  Making plans require `type=episodic` + `event_date` at write time is an
  `instructions.py` / `prompts.py` change, and the server already has
  `ttl_episodic` (90 d default) and `memory.py:2086` (episodic always gets an
  `event_date`) — the write path is fine, the read path was dropping it.
- **Supersession is not solved.** Nothing here knows that "switch from
  llama-swap to NINfer" was overtaken by events. That needs a contradiction
  pass — `fsck.py` already has a contradiction check (194 actions in the last
  full scan) and it is not wired to expiry. Not my call to design.
- **Near-dedup interacts with the suffix.** `_near_dedup` uses char-4gram
  Jaccard and keeps the longest cluster member; the suffix makes lines longer,
  which is harmless, but the same text recalled with two different dates stays
  two entries. That is intended — they are two different statements.

## Files

```
deployed/mnemory_filter.v0.4.2.baseline.py  # deployed source, verbatim from the Open WebUI function DB
deployed/mnemory_filter.v0.4.2.mn-fork.py   # patched (compiles clean)
deployed/mn-fork.patch                       # the diff, 5 hunks
deployed/test_fork_render.py                 # behavioural test, exercises the real patched code
build_fork.py                                 # regenerates patch + diff from the baseline
```

## Deploy

`deploy_to_owui.py` is dry-run by default: it GETs the live function, writes a
timestamped backup, and PUTs the patched content only with `--apply`. It
targets the API on `192.168.2.1:3001` (3000 is a different instance that
rejects this key).

## Not verified

I have not run the patched filter in Open WebUI. It compiles, and the rendering
helpers pass the tests above, but the inlet path needs a live chat to prove
prompt-cache behaviour is unchanged — the static/dynamic two-position injection
is the whole point of this filter, and a header change moves the dynamic block
slightly. Test in a throwaway chat first.
