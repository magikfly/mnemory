# Open WebUI functions (deployed out-of-band)

Files here are **not** part of the mnemory image and are **not** loaded by this
repo's CI. They are archived so they survive loss, because in Open WebUI a
function exists only as a row in `webui.db` table `function`: re-saving it in the
Workspace UI, or re-importing it from the community listing, silently overwrites
local changes with the upstream copy.

## mnemory_filter.py

Filter `mnemory_filter`, version `0.4.3-atlas.2`. Injects mnemory core memories
and memory instructions into the Open WebUI prompt, sanitises recalled text, and
caps core sections.

* sha256 `0afb5d2334539058c94351309eda12cb376dfe04abe29147adc5ef8c674ddbdd`,
  52937 chars. Re-verified against the live DB row before archiving, 2026-10-08.
* Deployed into the OWUI sqlite DB (container `OpenWebUI`, host path
  `/mnt/cache/appdata/open-webui/webui.db`). HTTP writes are unavailable: this
  build answers `PUT /api/v1/functions/id/<id>` with 405, so the write path is
  sqlite3 inside the container.
* No restart needed after a change — `utils/plugin.py:375`
  `get_function_module_from_cache` re-reads `content` per request and only
  reuses the cached module while contents match.
* Valves in use: `recall_score_threshold=0.5`, `core_max_per_section=6`,
  `agent_id` unset (Field default `open-webui`, which is deliberately the same
  scope the write path uses).

Five bugs separate this from the community copy; all five were found by
executing code, and the reasoning is recorded in netvault under
`services.mnemory.owui_filter_patch`:

1. `_inject_static_context` cached the block and returned without inserting —
   the insertion statements were orphaned after another function's `return`.
   Core memories and memory instructions reached no model at all.
2. `_PLACEHOLDER_RE` was `\b`-anchored around the bracket, so it matched only
   glued forms (`foo[chat_id]bar`) and never fired in prose — placeholder and
   secret leaks went straight into prompts.
3. mnemory wraps core bullets in `⟨memory_item⟩`; `_TAG_LEAK_RE` treated that
   wrapper as a leak and deleted 60 of 60 bullets, leaving bare section headers.
   Fixed by unwrapping before the hygiene test while keeping the tags in output.
4. The extractor copies its own few-shot examples into the store, so example
   outputs are matched by distinctive content tokens and dropped.
5. Speech-act narration ("User is asking…", "Assistant greeted…") is dropped;
   the verb list is deliberately narrow because the first draft also blocked
   real technical facts.

## Restoring

1. Confirm the damage: `select length(content) from function where id='mnemory_filter'`
   — it should be 52937.
2. Back the current row up before touching it, then write this file's content
   into `function.content` with sqlite3 inside the container.
3. Verify by **executing** the deployed code, never by reading it:
   `python3 /home/user/prefill-probe/show_injected_context.py` pulls the row from
   the DB, runs the real `inlet()` against live mnemory, prints what reaches the
   model, and exits non-zero on regression.
4. `gate_test.py` in the same directory is the behavioural gate for the hygiene
   rules, including the false-positive case that would have deleted the
   assistant's own identity memory.
