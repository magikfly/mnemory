# Operational tooling for this deployment

Everything here runs against a live mnemory over HTTP. Nothing is imported by the
service, and nothing here contains credentials: configuration is by environment.

| var | meaning |
|---|---|
| `MN_BASE_URL` | mnemory root, default `http://localhost:8050` |
| `MN_API_KEY` | `mnm-…` API key |
| `MN_USER_ID` | user whose store is audited; also the isolation boundary |
| `MN_AGENT_ID` | agent scope, default `open-webui` |

## Tools

**`mn_purge_audit.py`** — full dump (partitioned list calls; the endpoint reports
`has_more: false` at its cap and ignores `cursor`/`offset`/`page`), classification
into delete / strip / review / keep, and bulk repair. Read-only unless `--apply` is
paired with `--yes`. Pinned rows are never touched. Writes a JSON backup before
mutating, because retraction has no un-retract verb.

**`recall_eval.py`** — fixed query set with expectations, reporting hit@1/3/8, MRR
and short-turn return counts. The only honest way to tell a retrieval change from
a good feeling; it is what showed a purge lifting MRR 0.655 → 0.845 while a
`score_threshold` raise destroyed answers.

**`recall_eval_threshold.py`** — same harness swept across thresholds. Conclusion
recorded in its output: junk and signal share a score band, so thresholding trades
answers for a small junk reduction.

**`filter_gate.py`** — behavioural gate for the OWUI recall filter archived at
`deploy/owui-functions/mnemory_filter.py`. It imports the real module and asserts on
its methods, because that filter is patched out-of-band in OWUI's database and a UI
re-save silently reverts it. Set `MN_FILTER_PATH` to test a different copy.

**`filter_injection_probe.py`** — runs the deployed filter content out of OWUI's DB
through the real `inlet()` and prints the messages that would reach the model.
Prefer this over asking a model whether its instructions arrived: two separate
wrong answers this week came from exactly that.

## Order to use them in

After any memory-quality change: `filter_gate.py` (did I break the filter) →
`recall_eval.py` (did retrieval change) → `mn_purge_audit.py` dry run (what is the
store made of now). All three exit non-zero on regression.
