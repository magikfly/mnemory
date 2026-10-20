# Memory Model

## Two-Tier Architecture

**Fast Memory** (vector store): Concise facts and summaries, max 1000 characters. Semantically searchable via embeddings. Stored in Qdrant (local embedded or remote).

**Slow Memory** (artifact store): Detailed content — research reports, analysis, logs, code, images, PDFs. Up to 10MB per artifact (configurable via `MAX_ARTIFACT_SIZE`). Stored in S3/MinIO or local filesystem. Attached to fast memories, retrieved on demand. Text artifacts support pagination; binary artifacts are returned in full.

## Memory Types

| Type | Purpose | Default TTL |
|---|---|---|
| `preference` | Likes, dislikes, style choices | Permanent |
| `fact` | Biographical, factual information | Permanent |
| `episodic` | Events, interactions, conclusions | 90 days |
| `procedural` | Workflows, habits, "how to" | 60 days |
| `context` | Session/short-term context | 7 days |

Default TTLs are configurable via environment variables (see [Configuration](configuration.md)).

## Categories

Predefined categories ensure LLM discoverability:

| Category | Description |
|---|---|
| `personal` | Personal life, family, relationships |
| `preferences` | Likes, dislikes, style preferences |
| `health` | Physical, mental, medical |
| `work` | Job, career, professional |
| `technical` | Tools, languages, infrastructure |
| `finance` | Money, investments, billing |
| `home` | House, appliances, maintenance |
| `vehicles` | Cars, bikes, maintenance history |
| `travel` | Trips, places, itineraries |
| `entertainment` | Movies, music, books, games |
| `goals` | Objectives, plans, ambitions |
| `decisions` | Conclusions, choices, reasoning |
| `project` | Project-specific (use `project:<name>`) |

Dynamic subcategories via prefix: `project:myapp`, `project:demo/deployment-files`.

## Importance Levels

| Level | Search Weight | Use For |
|---|---|---|
| `low` | 0.1 | Minor details, temporary notes |
| `normal` | 0.4 | Standard memories (default) |
| `high` | 0.7 | Important facts, key decisions |
| `critical` | 1.0 | Essential information, always-relevant |

Search uses hybrid retrieval: dense vectors (semantic similarity via OpenAI embeddings) and BM25 sparse vectors (keyword matching via FastEmbed) are fused server-side using Qdrant's Reciprocal Rank Fusion (RRF). After fusion, an importance-based score boost is applied: `score = rrf_score + importance_weight * importance_value`, where `importance_weight = 1 - SEARCH_SIMILARITY_WEIGHT` (default 0.1). This makes importance a tiebreaker rather than a primary ranking factor. Results below `SEARCH_SCORE_THRESHOLD_HYBRID` (default 0.0) are filtered out. RRF score range depends on Qdrant's k constant: with the default k=1, scores are ~0.1-1.0 (similar to cosine similarity); with k=60, scores are ~0.01-0.03.

## Pinned Memories

Memories with `pinned: true` are loaded at every conversation start via `get_core_memories`. Use for:
- User identity facts ("Lives in Prague", "DevOps engineer")
- Core preferences ("Prefers direct communication")
- Agent identity ("Your name is Bob", "You speak casually")
- Agent knowledge ("You researched X and concluded Y")

## TTL (Time-To-Live)

Memories can have a TTL that causes them to decay (soft-expire) after a set number of days. Each memory type has a configurable default TTL (see Memory Types table above). You can override the default by passing `ttl_days` to `add_memory`.

**Lifecycle:**
1. Memory is created with `expires_at` calculated from `ttl_days`
2. When `expires_at` passes, the memory enters **decayed** state (soft-deleted)
3. Decayed memories are excluded from search and list by default
4. Use `include_decayed=true` to browse historical/expired memories
5. Decayed memories can be restored via `update_memory` (set new `ttl_days`)

**Reinforcement:** When a memory is accessed via `search_memories`, its TTL is automatically reset — `expires_at` is recalculated from now + original `ttl_days`. This means frequently-used memories stay alive. Controlled by `TRACK_MEMORY_ACCESS` config.

**Pinned exemption:** Pinned memories (`pinned: true`) are exempt from TTL — they never decay, even if `expires_at` is set.

**Existing memories:** No migration needed. Memories without TTL fields are treated as permanent.

## Memory Layer

Memories have a `memory_layer` field that controls their role in the two-layer system:

| Layer | Source | Purpose | Recall Priority |
|---|---|---|---|
| `raw` | `remember` endpoint | Provisional evidence from conversation extraction | Lower (penalized in ranking) |
| `consolidated` | `add_memory`, consolidation service | Durable canonical knowledge | Higher (baseline) |

**Backward compatibility:** Memories without a `memory_layer` field are treated as `consolidated`.

### Related metadata fields

| Field | Type | Description |
|---|---|---|
| `memory_layer` | `"raw"` or `"consolidated"` | Which layer this memory belongs to |
| `superseded_by` | `str` or `null` | ID of the consolidated memory that replaced this raw memory |
| `derived_from` | `list[str]` or `null` | IDs of source raw memories (on consolidated memories from consolidation) |

### Dedup isolation

The `remember` pipeline only deduplicates against other raw memories — it never modifies consolidated memories. This prevents noisy raw extraction from overwriting high-quality consolidated content.

### Consolidation

A background consolidation service synthesizes durable knowledge from raw memories and their session summaries:

1. **Within-session**: After a session goes idle, reads the session summary + raw memories, LLM synthesizes consolidated memories
2. **Re-consolidation**: If a session continues after consolidation, new raw memories reset the session to idle. The next consolidation run is append-only — the LLM receives both new raw memories and previously consolidated memories as context, produces new consolidated outputs, and the new consolidated memory IDs are appended alongside previous ones
3. **Automated maintenance (fsck)**: An automated maintenance engine keeps durable memories coherent. Auto-fsck runs incrementally — only memories changed since the last check are processed. The four-phase pipeline (security → duplicates → content quality → metadata normalization) runs with bounded LLM cost (`FSCK_MAX_LLM_CALLS`). Manual fsck runs a full scan. Raw memories are excluded by default
4. **Garbage collection**: Old superseded raw memories without artifacts are deleted after a configurable retention period. GC runs independently of fsck during the maintenance cycle

### Recall ranking

Search results are scored with layer-aware penalties:
- Consolidated (or absent `memory_layer`): baseline score
- Raw, not superseded: small penalty (`RECALL_RAW_PENALTY`, default 0.05)
- Raw, superseded: larger penalty (`RECALL_SUPERSEDED_PENALTY`, default 0.15)

## Role

The `role` parameter controls who the memory is about:

| Role | Description | Example |
|---|---|---|
| `user` (default for `add_memory`) | Facts about the user | "User lives in Prague", "User prefers dark mode" |
| `assistant` | Facts about the agent itself | "Your name is Bob", "You speak casually" |

When `role="assistant"`, the server uses an agent-specific fact extraction prompt that focuses on the assistant's identity, personality, and capabilities. When `role="user"` (default for `add_memory`), user fact extraction is used. The `role` is stored in metadata and can be used as a filter in `search_memories` and `list_memories`.

The REST `POST /api/remember` endpoint also supports **auto mode** (`role=null`, the default for `remember`). In auto mode, the LLM extracts facts from all participants and attributes each fact to the correct role. Assistant facts are silently dropped if no `agent_id` is set in the session. This is the recommended mode for plugins that send full conversation exchanges.

`get_core_memories` uses `role` to organize agent-scoped memories into sections:
- **Agent Identity**: pinned, `role=assistant`, fact/preference type
- **Agent Knowledge**: pinned, `role=assistant`, other types
- **Agent Instructions**: pinned, `role=user`, agent-scoped (user preferences specific to this agent)

## Scoping

- `user_id` (required): Every memory belongs to a user. Shared across all agents. Can be set at session level via API key mapping (`MCP_API_KEYS`) or `X-User-Id` header, eliminating the need to pass it per tool call.
- `agent_id` (optional): Set for agent-scoped memories. Two use cases:
  - **Agent identity** (`role="assistant"`): The agent's name, personality, capabilities, knowledge.
  - **Agent-scoped user preferences** (`role="user"`): User preferences that apply only to this agent (e.g., "User wants me to create commit messages").
  
  Different agents see different agent memories but share user memories. Can be set at session level via `X-Agent-Id` header.

### Sub-agents

Sub-agents allow creating multiple independent agent identities under a single session. Use a colon-separated prefix matching the session's `X-Agent-Id`:

```
X-Agent-Id: openwebui          <- session agent
agent_id: openwebui:bob        <- sub-agent "bob" (allowed)
agent_id: openwebui:alice      <- sub-agent "alice" (allowed)
agent_id: cursor:foo           <- different parent (blocked)
```

Sub-agents are fully independent — they have their own memories and do NOT inherit from the parent agent. The session agent can access and manage all its sub-agents' memories.

## Memory Metadata

Custom metadata is stored as flat fields in the Qdrant payload alongside standard fields:

| Field | Type | Description |
|---|---|---|
| `memory_type` | str | preference, fact, episodic, procedural, context |
| `categories` | list[str] | Category tags |
| `importance` | str | low, normal, high, critical |
| `pinned` | bool | Whether to include in core memories |
| `role` | str | "user" or "assistant" — who the memory is about. Always stored as one of these two values. |
| `artifacts` | list[dict] | Artifact metadata (id, filename, content_type, size, created_at) |
| `event_date` | str\|None | ISO 8601 UTC datetime of when the event occurred |
| `created_at_utc` | str | UTC timestamp |
| `ttl_days` | int\|None | Original TTL setting in days (None = permanent) |
| `expires_at` | str\|None | ISO 8601 expiration timestamp (None = never expires) |
| `decayed_at` | str\|None | When memory entered decayed state (None = active) |
| `labels` | dict\|None | Client-provided key-value metadata (e.g., project, topic, conversation_id). Bypasses LLM extraction. Filterable in search/list queries. |
| `last_accessed_at` | str\|None | Last time returned in search |
| `access_count` | int | Number of times accessed in search |

## Labels

Labels are client-provided key-value metadata that can be attached to any memory. Unlike categories and memory types, labels **bypass the LLM entirely** — they are stored exactly as provided and never inferred or modified by the extraction pipeline.

### Use Cases

- **Project scoping**: `{"project": "myapp"}` — filter memories by project
- **Conversation tracking**: `{"conversation_id": "conv-123"}` — link memories to conversations
- **Environment tagging**: `{"env": "prod", "region": "eu-west-1"}` — tag by deployment context
- **Custom workflows**: `{"reviewed": true, "priority": 1}` — application-specific metadata

### Value Types

Labels support flat values only:

| Type | Example |
|---|---|
| `str` | `"myapp"` |
| `int` | `42` |
| `float` | `3.14` |
| `bool` | `true` |
| `list[str]` | `["tag1", "tag2"]` |

Nested dicts and other complex types are not allowed.

### Inheritance

When using `infer=True` or the `remember` pipeline, labels are **inherited by all extracted facts**. If you call `add_memory(content="...", labels={"project": "myapp"})` and the LLM extracts 3 facts, all 3 get those labels.

### Filtering

Labels can be used as filters in `search_memories`, `find_memories`, and `list_memories`. Filtering uses AND logic across multiple keys, with list values using any-of (OR) within a single key:

```
labels={"project": "myapp", "env": "prod"}
```

This matches memories where `labels.project == "myapp"` AND `labels.env == "prod"`.

### Updating

- Passing `labels={"key": "value"}` to `update_memory` **merges** with existing labels (caller wins on key conflicts)
- Passing `labels={}` (empty dict) **clears** all labels
- Passing `labels=None` (or omitting) **preserves** existing labels

### Constraints

| Constraint | Default | Config |
|---|---|---|
| Max labels per memory | 20 | `LABELS_MAX_FIELDS` |
| Max key length | 64 chars | `LABELS_MAX_KEY_LENGTH` |
| Max string value length | 1000 chars | `LABELS_MAX_VALUE_LENGTH` |
| Key pattern | `^[a-zA-Z_][a-zA-Z0-9_]*$` | — |
| Reserved keys | System field names (e.g., `memory_type`, `user_id`, `labels`) | — |
