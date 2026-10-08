"""
title: Mnemory - Persistent Memory
author: mnemory
description: Automatic memory recall and storage for conversations
# Patched locally (atlas.1): injector fix, tag-aware + capped sanitizer,
# example-leak and speech-act hygiene, placeholder regex de-anchored.
# Upstream baseline was version: 0.4.2
version: 0.4.3-atlas.2
"""

import asyncio
import logging
import re
import time
from typing import Callable, Optional

import aiohttp
from pydantic import BaseModel, Field

_log = logging.getLogger(__name__)


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Filter priority (lower = runs first)",
        )
        mnemory_url: str = Field(
            default="http://localhost:8050",
            description="Mnemory server base URL",
        )
        api_key: str = Field(
            default="",
            description="API key for mnemory authentication",
        )
        agent_id: str = Field(
            default="open-webui",
            description="Agent ID sent to mnemory",
        )
        recall_mode: str = Field(
            default="always",
            description=(
                "When to recall memories: "
                "'always' = every message (recommended), "
                "'first_only' = first message only (no subsequent recalls)"
            ),
        )
        recall_search_mode: str = Field(
            default="search",
            description=(
                "Search mode for recall: "
                "'find' = AI-powered multi-query search (thorough, slower), "
                "'search' = single vector search (fast, no LLM)"
            ),
        )
        recall_find_first: bool = Field(
            default=True,
            description=(
                "When recall_search_mode is 'search', use 'find' for the "
                "first message in a session (thorough initial context). "
                "Ignored when recall_search_mode is 'find'."
            ),
        )
        recall_score_threshold: float = Field(
            default=0.5,
            description=(
                "Minimum relevance score (0.0-1.0) for recalled memories. "
                "Higher = fewer but more relevant memories injected. "
                "Prevents context bloat from weak matches on follow-up messages."
            ),
        )
        core_max_per_section: int = Field(
            default=6,
            description=(
                "Max bullets rendered per core-memory section (0 = unlimited). Pinned "
                "memories are ordered first by mnemory, so the cap drops low-importance "
                "top-N entries before pinned ones. Uncapped core measured ~14.6 KB/turn."
            ),
        )
        recall_carry_max: int = Field(
            default=40,
            description=(
                "Max recalled memories carried forward across turns. "
                "The server withholds already-sent memories (known_skipped) "
                "on the assumption the model still has them, but Open WebUI "
                "does not persist injected system messages — so without "
                "carry-forward each memory is visible for one turn only. "
                "Set 0 for upstream behaviour (this turn's results only)."
            ),
        )
        show_status: bool = Field(
            default=True,
            description="Show memory status messages in chat (can be overridden per-user)",
        )
        debug: bool = Field(
            default=False,
            description=(
                "Emit detailed debug info as chat status messages. "
                "Shows session resolution, query, API response stats, "
                "tool stripping, and injection details."
            ),
        )
        request_timeout: int = Field(
            default=30,
            description="HTTP request timeout in seconds for mnemory API calls",
        )
        laya_url: str = Field(
            default="http://192.168.2.1:8013",
            description=(
                "Laya /memcheck endpoint used as a semantic tie-breaker for "
                "near-duplicate recall memories. Fail-open: if unreachable "
                "the local n-gram dedup is used alone. Set empty to disable. "
                "NOTE: points at the laya-system-one container's published host port "
                "(stable across container recreates); the bridge is kept alive by "
                "the laya-system-one-selfheal cron on unraid."
            ),
        )
        laya_refine_interval: int = Field(
            default=900,
            description=(
                "Minimum seconds between Laya semantic refine passes per "
                "session (background, never blocks a turn)."
            ),
        )
        near_dup_jaccard: float = Field(
            default=0.55,
            description=(
                "Char-4gram Jaccard similarity threshold at which two recall "
                "memories are treated as near-duplicates (the more complete "
                "one is kept). 0.3-0.55 is the ambiguous zone that the "
                "Laya /memcheck pass resolves semantically."
            ),
        )
        strip_redundant_mcp_tools: bool = Field(
            default=True,
            description=(
                "Remove mnemory MCP tools that the filter handles automatically "
                "(initialize_memory, get_core_memories, get_recent_memories) "
                "from the request to reduce prompt token usage."
            ),
        )

    class UserValves(BaseModel):
        enabled: bool = Field(
            default=True,
            description="Enable memory for this user",
        )
        show_status: bool = Field(
            default=True,
            description="Show memory status messages in chat",
        )

    # Max tracked sessions before evicting oldest entries.
    # Prevents unbounded memory growth in long-running instances.
    _MAX_SESSIONS = 1000
    _MAX_PENDING_SESSIONS = 100

    # Mnemory MCP tools that the filter handles automatically.
    # Stripped from tool_ids and tools[] to save prompt tokens.
    _MANAGED_TOOL_SUFFIXES = {
        "initialize_memory",
        "get_core_memories",
        "get_recent_memories",
    }

    # ── Recall hygiene & near-duplicate collapse ──────────────────────
    # Unresolved template placeholders ([child_name], [date], [TODO]...
    #). Deliberately narrow — lowercase words joined by underscores inside
    # brackets — so footnote-style [2, 3], license plates, and the
    # [scope: ...] metadata suffix never match.
    # Exact strings the extractor is shown as few-shot EXAMPLES in
    # mnemory/prompts.py. A small model handed a non-English turn sometimes emits
    # an example's OUTPUT instead of extracting from the real input, which is how
    # 'User prefers cooking and collecting stamps.' entered the store (Example 12 is
    # a Czech greeting whose sample output is that hobby line) - and Dutch greetings
    # keep hitting that path. A match against these is fabrication, not memory.
    # Exact matching is not enough: the stored row says 'prefers', the example says
    # 'enjoys'. So compare only the DISTINCTIVE tokens, i.e. the example's content
    # words after removing the template words every example shares.
    _EXAMPLE_OUTPUTS: tuple = (
        "User's name is John",
        'User is a software engineer at Google',
        'User switched from VS Code to Neovim',
        'Caroline attended a LGBTQ support group',
        'Caroline was promoted to senior engineer at Google',
        'John proposed using Kubernetes',
        'Sarah prefers ECS over Kubernetes for their scale',
        "User's mother likes sweet drinks, especially Malibu",
        "User's mother loves Stephen King books",
        "User's mother has a garden",
        'User has a Kurilian Bobtail cat',
        'User wants to implement OIDC authentication for myapp using ALB and Cognito',
        'User wants to add distributed tracing to their platform using OpenTelemetry',
        'User decided to use PostgreSQL instead of MySQL for the billing service',
        "User's email is john@example.com",
        'User lives in Prague',
        'User works at Acme Corp',
        'Assistant is a helpful coding assistant specializing in Python and Rust',
        'Assistant prefers concise, direct answers',
        'Assistant researched Kubernetes networking, \\\nconcluded Cilium is the best CNI',
        "Assistant's name is Aria, a research assistant",
        "Assistant's name is Bob",
        'Assistant prefers verbose responses',
        "User's name is Petr",
        'User is from Ostrava',
        'User enjoys cooking and collecting stamps',
        'User wants to implement canary deployments with automatic rollback using Argo Rollouts',
        'Assistant specializes in scientific literature review and data analysis',
        'Assistant prefers to give concise, brief responses',
        'Assistant enjoys helping users solve complex problems and explaining technical concepts',
        'User needs help swapping the motor in their 2015 Skoda Octavia',
        'User wants to implement OIDC authentication for myapp',
        'Assistant decided to implement OIDC for myapp using ALB OIDC action with Cognito',
        'User lives in Berlin',
        'User has a cat named Luna',
        'User works at Google',
    )
    _EX_STOP = frozenset(['a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'from', 'has', 'have', 'i', 'if', 'in', 'is', 'it', 'its', 'me', 'my', 'no', 'not', 'of', 'on', 'or', 'our', 'she', 'so', 'than', 'that', 'the', 'their', 'them', 'they', 'this', 'to', 'too', 'we', 'will', 'with', 'your', 'you', 'user', 'users', 'assistant', 'assistants', 'name', 'names', 'prefers', 'prefer', 'enjoys', 'enjoy', 'liked', 'likes', 'like', 'decided', 'wants', 'want', 'needs', 'was', 'were', 'been', 'being', 'would', 'can', 'could', 'said', 'told', 'asked', 'about'])

    # Speech-act / greeting narration: records that something was asked, or that the
    # assistant said hello, with no durable fact attached. Crowds out useful recall
    # on short turns.
    _TRIVIA_RE = re.compile(
        '^(?:User is asking|User wants to know|User inquired about|User asked how|User requested a greeting|User requested (a |an )?(test|quiz|challenge)|User requested to (kill|run|try|check)\\b|Assistant greeted\\b|Assistant confirmed readiness\\b|Assistant offered to\\b|Assistant is ready to\\b)',
        re.I,
    )

    # Per-memory framing mnemory adds around stored content. Legitimate, so it is
    # stripped before the hygiene test - _TAG_LEAK_RE would otherwise fail every
    # core bullet and render the sections empty.
    _MEMORY_ITEM_WRAPPER_RE = re.compile("\\u27e8/?memory_item\\u27e9")



    _PLACEHOLDER_RE = re.compile(r"\[[a-z][a-z0-9]*(?:_[a-z0-9]+)+\]")
    # Leaked internal boundary tags from the mnemory store.
    _TAG_LEAK_RE = re.compile(
        r"\u27e8/?(?:memory_item|user_input|existing_memories|content"
        r"|extracted_memories)\u27e9"
    )
    # Process-noise tails ("Please provide X to complete ...").
    _NOISE_RE = re.compile(
        r"please provide .{0,80} to (?:complete|fill|verify)", re.I
    )

    _JACCARD_N = 4
    _GRAM_LOW = 0.30        # below this: definitely distinct (no Laya call)
    _LAYA_SAME = 0.90       # 'same' probability that triggers a merge
    _LAYA_PAIR_CAP = 20     # max ambiguous pairs per refine pass
    _last_laya: dict = {}   # session_id -> last refine-pass timestamp

    # NOTE: continuation lines start at column 0 on purpose.  Indenting
    # them to match the class body would ship leading whitespace on
    # every line of the prompt on every request.
    _MEMORY_INSTRUCTIONS = """## Memory (mnemory)
Recall and storage are automatic — a filter handles them. Relevant memories are already \
injected below; weave them into answers naturally, don't just acknowledge them.

Never call memory tools on your own initiative — not to initialize, load core memories, \
store facts, or "check for context." The filter already did all of that.

Only touch memory tools when the user explicitly asks:
- look something up -> search_memories / find_memories / ask_memories
- remember / change / forget something -> add_memory / update_memory / delete_memory
- browse -> list_memories / list_categories

Tool schemas document their own parameters. Read them at call time — don't rely on this \
prompt for them."""

    def __init__(self):
        self.valves = self.Valves()
        # Track which chats have been initialized.
        # Maps chat_id -> {"session_id": str, "user_id": str, "static_ctx": str | None}
        # static_ctx holds cached instructions + core memories from the
        # first turn, re-injected on every subsequent turn so the LLM
        # always has memory context and managed-mode guidance.
        self._sessions: dict[str, dict] = {}
        # Pending sessions from first messages when chat_id is not yet
        # available.  Open WebUI may not provide chat_id on the first
        # message of a new chat (chat not yet saved to DB).  Keyed by
        # user_id (email) to prevent cross-user session leakage when
        # multiple users start chats concurrently.
        self._pending_sessions: dict[str, dict] = {}
        # session_id -> list[str] of recalled memory texts.
        #
        # The dynamic "## Recalled Memories" block is appended to the
        # request body only; Open WebUI persists just user/assistant
        # messages, so the injected system message is gone by the next
        # turn.  The server meanwhile tracks which memory IDs it has
        # already returned for a session and withholds them
        # (known_skipped), assuming the model still has them.  Without
        # carry-forward each memory is therefore visible for exactly
        # one turn.  Keyed by session_id rather than stored on the
        # session dict because _save_session rebuilds that dict and
        # would drop extra keys.
        self._recalled: dict[str, list[str]] = {}
        # user_id -> static context text.  Fallback for when the
        # per-chat session is missing (no chat_id in the inlet, evicted
        # session, filter reload) so the static block never vanishes
        # from its fixed early position mid-chat.  A block that appears
        # on one turn and not the next rewrites the prompt prefix and
        # forces a full re-prefill on the server.
        self._static_by_user: dict[str, str] = {}

    # ── Helpers ───────────────────────────────────────────────────────

    async def _debug(self, emitter: Callable | None, msg: str) -> None:
        """Emit a debug status message into the chat if debug mode is on."""
        if not self.valves.debug or not emitter:
            return
        await emitter(
            {
                "type": "status",
                "data": {"description": f"[mnemory debug] {msg}", "done": True},
            }
        )

    async def _post(
        self,
        path: str,
        payload: dict,
        user: dict,
        emitter: Callable | None = None,
    ) -> dict | None:
        """Make a POST request to mnemory REST API."""
        headers = {
            "Content-Type": "application/json",
            "X-Agent-Id": self.valves.agent_id,
            "X-User-Id": user.get("email", user.get("id", "")),
        }
        if self.valves.api_key:
            headers["Authorization"] = f"Bearer {self.valves.api_key}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.valves.mnemory_url}{path}",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.valves.request_timeout),
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    await self._debug(
                        emitter,
                        f"API {path} returned {resp.status}",
                    )
                    return None
        except Exception as exc:
            await self._debug(emitter, f"API {path} error: {exc}")
            return None  # Graceful degradation

    def _strip_managed_tools(
        self, body: dict, emitter: Callable | None = None
    ) -> list[str]:
        """Remove mnemory MCP tools the filter handles automatically.

        Strips from both ``tool_ids`` (list of string identifiers) and
        ``tools`` (list of tool-definition dicts) so the LLM never sees
        the managed tools regardless of how Open WebUI passes them.

        Returns list of stripped tool names for debug logging.
        """
        stripped: list[str] = []

        # Strip from tool_ids (list[str])
        tool_ids = body.get("tool_ids")
        if tool_ids:
            kept = []
            for t in tool_ids:
                if isinstance(t, str) and any(
                    t.endswith(s) for s in self._MANAGED_TOOL_SUFFIXES
                ):
                    stripped.append(t)
                else:
                    kept.append(t)
            body["tool_ids"] = kept

        # Strip from tools (list[dict]) — Open WebUI may pass full tool
        # definitions here instead of (or in addition to) tool_ids.
        tools = body.get("tools")
        if tools and isinstance(tools, list):
            kept_tools = []
            for t in tools:
                if isinstance(t, dict) and self._is_managed_tool(t):
                    name = self._tool_name(t)
                    stripped.append(name or "unknown_tool")
                else:
                    kept_tools.append(t)
            body["tools"] = kept_tools

        return stripped

    def _is_managed_tool(self, tool: dict) -> bool:
        """Check if a tool definition dict matches a managed tool suffix."""
        name = self._tool_name(tool)
        if name and any(name.endswith(s) for s in self._MANAGED_TOOL_SUFFIXES):
            return True
        return False

    @staticmethod
    def _tool_name(tool: dict) -> str:
        """Extract the tool name from a tool definition dict."""
        for key in ("id", "name", "tool_id"):
            val = tool.get(key, "")
            if val and isinstance(val, str):
                return val
        func = tool.get("function")
        if isinstance(func, dict):
            name = func.get("name", "")
            if name and isinstance(name, str):
                return name
        return ""

    def _get_session(self, chat_id: str, user_id: str = "") -> dict | None:
        """Look up or adopt a session for the given chat_id.

        Handles the first-to-second-message transition where chat_id
        appears after the pending session was already created.

        Args:
            chat_id: Open WebUI chat ID (may be empty on first message).
            user_id: User identifier (email) for pending session lookup.
                Required for correct multi-user isolation.

        When adopting, the pending session is NOT cleared — the inlet
        may still need it on subsequent turns when chat_id is
        unavailable (Open WebUI provides chat_id in the outlet but not
        always in the inlet).  Pending sessions are only cleared when a
        new first turn starts (is_first=True in inlet).

        Note: if the same user opens two browser tabs and both send
        their first message before either receives a chat_id, the
        second pending session overwrites the first.  This is a known
        limitation scoped to a single user (not cross-user).
        """
        if not user_id:
            return None
        if chat_id:
            sess = self._sessions.get(chat_id)
            if sess is None:
                pending = self._pending_sessions.get(user_id)
                if pending:
                    sess = pending
                    self._sessions[chat_id] = sess
                    # Don't clear pending — inlet may still need it
                    # when chat_id is not available.
            # Defense-in-depth: verify session belongs to this user.
            # Prevents cross-user access if chat_ids ever collide.
            # Note: uses != (not truthiness check) so sessions stored
            # with user_id="" are never returned to an authenticated user.
            if sess and sess.get("user_id") != user_id:
                _log.warning(
                    "Session user_id mismatch: stored=%r requesting=%r "
                    "chat_id=%r — refusing access",
                    sess["user_id"],
                    user_id,
                    chat_id,
                )
                return None
            return sess
        return self._pending_sessions.get(user_id)

    def _save_session(
        self,
        chat_id: str,
        session_id: str,
        static_ctx: str | None = None,
        *,
        update_ctx: bool = False,
        user_id: str = "",
    ) -> None:
        """Store or update session data for a chat.

        Args:
            chat_id: Open WebUI chat ID (may be empty on first message).
            session_id: Server-side session ID from recall response.
            static_ctx: Cached instructions + core memories text.
            update_ctx: If True, overwrite static_ctx even when the new
                value is None (used on first turn to set the cache).
                If False, preserve the existing static_ctx.
            user_id: User identifier (email) for pending session scoping.
                Required for correct multi-user isolation.
        """
        if chat_id:
            existing = self._sessions.get(chat_id)
            sess = {
                "session_id": session_id,
                "user_id": user_id,
                "static_ctx": (
                    static_ctx
                    if update_ctx
                    else (static_ctx or (existing["static_ctx"] if existing else None))
                ),
            }
            self._sessions[chat_id] = sess
            # Clear this user's pending session now that we have a chat_id
            if user_id:
                self._pending_sessions.pop(user_id, None)
            # Evict oldest entries if over limit
            if len(self._sessions) > self._MAX_SESSIONS:
                excess = len(self._sessions) - self._MAX_SESSIONS
                for key in list(self._sessions)[:excess]:
                    del self._sessions[key]
                _log.warning(
                    "mnemory: evicted %d oldest sessions (limit=%d)",
                    excess,
                    self._MAX_SESSIONS,
                )
        elif user_id:
            existing = self._pending_sessions.get(user_id)
            self._pending_sessions[user_id] = {
                "session_id": session_id,
                "user_id": user_id,
                "static_ctx": (
                    static_ctx
                    if update_ctx
                    else (static_ctx or (existing["static_ctx"] if existing else None))
                ),
            }
            # Evict oldest pending entries if over limit
            if len(self._pending_sessions) > self._MAX_PENDING_SESSIONS:
                excess = len(self._pending_sessions) - self._MAX_PENDING_SESSIONS
                for key in list(self._pending_sessions)[:excess]:
                    del self._pending_sessions[key]
                _log.warning(
                    "mnemory: evicted %d oldest pending sessions (limit=%d)",
                    excess,
                    self._MAX_PENDING_SESSIONS,
                )
        else:
            _log.warning(
                "mnemory: _save_session called with empty chat_id and "
                "user_id — session data dropped. Check __user__ "
                "population in Open WebUI."
            )

    # ── Hygiene + near-dedup helpers ──────────────────────────────────

    @classmethod
    def _hygiene_violation(cls, text: str) -> bool:
        """True when a memory text is broken: unresolved placeholder,
        leaked boundary tag, or process-noise tail. Such text must never
        reach the prompt."""
        if not text:
            return False
        return bool(
            cls._PLACEHOLDER_RE.search(text)
            or cls._TAG_LEAK_RE.search(text)
            or cls._NOISE_RE.search(text)
            or cls._TRIVIA_RE.search(text)
            or _example_leak(text)
        )

    @staticmethod
    def _norm_text(t: str) -> str:
        return re.sub(r"\s+", " ", t or "").strip().lower()

    @classmethod
    def _grams(cls, t: str):
        n = cls._JACCARD_N
        s = cls._norm_text(t)
        if not s:
            return set()
        if len(s) < n:
            return {s}
        return {s[i:i + n] for i in range(len(s) - n + 1)}

    @classmethod
    def _jaccard(cls, a: str, b: str) -> float:
        ga, gb = cls._grams(a), cls._grams(b)
        if not ga or not gb:
            return 0.0
        return len(ga & gb) / len(ga | gb)

    def _near_dedup(self, items: list[str]) -> list[str]:
        """Collapse near-duplicates, keeping the more complete (longer)
        text of each cluster. Hygiene-violating entries are dropped.
        O(n^2) is fine: buckets are capped at recall_carry_max (40)."""
        clean: list[str] = []
        for text in items:
            if self._hygiene_violation(text):
                continue
            dup_idx = None
            for i, kept in enumerate(clean):
                if kept == text or self._jaccard(text, kept) >= self.valves.near_dup_jaccard:
                    dup_idx = i
                    break
            if dup_idx is None:
                clean.append(text)
            elif len(text) > len(clean[dup_idx]):
                clean[dup_idx] = text
        return clean

    def _maybe_laya_refine(self, session_id: str, bucket: list[str]) -> None:
        """Schedule an async Laya semantic pass over Jaccard-ambiguous
        pairs (_GRAM_LOW <= j < threshold). Throttled per session,
        fail-open, never blocks the turn. Merge only when Laya reports
        'same' >= _LAYA_SAME; on any error the pass is skipped and the
        next throttle window retries."""
        base = (self.valves.laya_url or "").rstrip("/")
        if not base or len(bucket) < 2:
            return
        now = time.time()
        if now - self._last_laya.get(session_id, 0.0) < self.valves.laya_refine_interval:
            return
        self._last_laya[session_id] = now
        pairs = []
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                a, b = bucket[i], bucket[j]
                if a == b:
                    continue
                g = self._jaccard(a, b)
                if self._GRAM_LOW <= g < self.valves.near_dup_jaccard:
                    pairs.append((i, j, a, b))
                if len(pairs) >= self._LAYA_PAIR_CAP:
                    break
            if len(pairs) >= self._LAYA_PAIR_CAP:
                break
        if not pairs:
            return
        snapshot = list(bucket)
        asyncio.create_task(
            self._laya_refine_pass(base, session_id, pairs, snapshot)
        )

    async def _laya_refine_pass(
        self, base: str, session_id: str, pairs: list, snapshot: list[str]
    ) -> None:
        """Background Laya pass: merge pairs Laya judges near-identical.
        Applied only if the bucket prefix still matches the snapshot
        (no concurrent growth); otherwise the pass is discarded."""
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as http:
                drops: set = set()
                for i, j, a, b in pairs:
                    if i in drops or j in drops:
                        continue
                    async with http.post(
                        f"{base}/memcheck", json={"a": a, "b": b}
                    ) as resp:
                        if resp.status != 200:
                            _log.debug(
                                "mnemory: laya /memcheck HTTP %d — pass aborted (fail-open)",
                                resp.status,
                            )
                            return
                        data = await resp.json()
                    same = data.get("same")
                    if isinstance(same, (int, float)) and same >= self._LAYA_SAME:
                        # keep the longer (more complete) of the pair
                        drops.add(j if len(b) >= len(a) else i)
                if drops:
                    cur = self._recalled.get(session_id)
                    if cur and len(cur) >= len(snapshot) and cur[:len(snapshot)] == snapshot:
                        self._recalled[session_id] = [
                            t for k, t in enumerate(cur) if k not in drops
                        ]
                        _log.info(
                            "mnemory: laya refine merged %d near-dup memor%s (session %s)",
                            len(drops),
                            "y" if len(drops) == 1 else "ies",
                            session_id,
                        )
        except Exception as exc:
            _log.debug("mnemory: laya refine pass skipped (fail-open): %s", exc)

    def _carry_recalled(
        self, session_id: str, search_results: list | None
    ) -> list[str]:
        """Merge this turn's results into the session's recall bucket.

        Preserves order, drops exact AND near duplicates (char-4gram
        Jaccard, longest text wins), skips hygiene-violating entries,
        trims to recall_carry_max (oldest first), and evicts old buckets
        to bound growth. Returns the full cleaned bucket to inject.
        Also schedules an async Laya semantic pass for the ambiguous
        Jaccard zone (fail-open, throttled).
        """
        bucket = self._recalled.setdefault(session_id, [])
        for m in search_results or []:
            text = m.get("memory")
            if not text:
                continue
            if self._hygiene_violation(text):
                _log.debug(
                    "mnemory: not carrying hygiene-violating recall: %.80s", text
                )
                continue
            bucket.append(text)

        # Re-clean the whole bucket: catches pre-patch accumulation
        # (historical near-dups) and replaces shorter cluster members
        # with the most complete one.
        cleaned = self._near_dedup(bucket)
        cap = self.valves.recall_carry_max
        if len(cleaned) > cap:
            cleaned = cleaned[len(cleaned) - cap:]
        self._recalled[session_id] = cleaned

        if len(self._recalled) > self._MAX_SESSIONS:
            excess = len(self._recalled) - self._MAX_SESSIONS
            for key in list(self._recalled)[:excess]:
                del self._recalled[key]
            _log.warning(
                "mnemory: evicted %d oldest recall buckets (limit=%d)",
                excess,
                self._MAX_SESSIONS,
            )

        self._maybe_laya_refine(session_id, cleaned)

        return cleaned

    # ── Inlet (before LLM) ───────────────────────────────────────────

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable] = None,
    ) -> dict:
        """Before LLM: recall memories and inject into context."""
        if not __user__:
            return body

        # Check user valves
        user_valves = __user__.get("valves")
        if user_valves and hasattr(user_valves, "enabled") and not user_valves.enabled:
            return body

        # Strip redundant mnemory MCP tools to save prompt tokens.
        # The filter handles recall automatically — these tools would
        # only waste tokens in the tools[] array on every LLM request.
        if self.valves.strip_redundant_mcp_tools:
            stripped = self._strip_managed_tools(body, __event_emitter__)
            if stripped:
                await self._debug(
                    __event_emitter__,
                    f"Stripped tools: {', '.join(stripped)}",
                )
            else:
                await self._debug(
                    __event_emitter__,
                    "No managed tools found to strip"
                    f" (tool_ids={body.get('tool_ids', 'absent')!r})",
                )

        chat_id = body.get("chat_id") or ""
        user_id = __user__.get("email", __user__.get("id", ""))
        sess = self._get_session(chat_id, user_id)
        session_id = sess["session_id"] if sess else None

        # Determine first turn from conversation history, not session
        # tracking.  Session-based detection is unreliable because
        # chat_id may be absent on the first message, causing the
        # empty-key entry in _sessions to collide across different chats.
        messages = body.get("messages", [])
        user_msg_count = sum(1 for m in messages if m.get("role") == "user")
        is_first = user_msg_count <= 1

        await self._debug(
            __event_emitter__,
            f"chat_id={chat_id!r} session_id={session_id!r} "
            f"user_id={user_id!r} is_first={is_first} user_msgs={user_msg_count}",
        )

        # On the first turn, always send session_id=None so the recall
        # endpoint treats it as a fresh session and loads core memories.
        # A stale pending session from a previous chat could otherwise
        # cause the server to skip core memory loading.
        if is_first:
            session_id = None
            # Clear stale pending session for THIS user only
            if user_id:
                self._pending_sessions.pop(user_id, None)

        # In first_only mode, skip recall on subsequent messages.
        # Still inject cached static context so the LLM keeps its
        # memory instructions and core memories, plus everything
        # already recalled earlier in this session.
        if not is_first and self.valves.recall_mode == "first_only":
            self._inject_static_context(body, sess, user_id)
            if sess and self.valves.recall_carry_max > 0:
                self._inject_recalled(body, self._recalled.get(sess["session_id"]))
            return body

        # Extract query from last user message
        query = ""
        for msg in reversed(body.get("messages", [])):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                query = content if isinstance(content, str) else ""
                break

        await self._debug(
            __event_emitter__,
            f"Query ({len(query)} chars): {query[:120]}{'...' if len(query) > 120 else ''}",
        )

        if not query and not is_first:
            self._inject_static_context(body, sess, user_id)
            if sess and self.valves.recall_carry_max > 0:
                self._inject_recalled(body, self._recalled.get(sess["session_id"]))
            return body  # No query on subsequent turn — skip search

        # Show status (admin valve AND user valve must both be true)
        show_status = self.valves.show_status
        if show_status and user_valves and hasattr(user_valves, "show_status"):
            show_status = user_valves.show_status
        if __event_emitter__ and show_status:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Recalling memories...",
                        "done": False,
                    },
                }
            )

        # Determine search mode for this call
        if is_first and self.valves.recall_find_first:
            search_mode = "find"
        else:
            search_mode = self.valves.recall_search_mode

        # Call recall endpoint
        payload: dict = {
            "session_id": session_id,
            "query": query,
            "search_mode": search_mode,
            "score_threshold": self.valves.recall_score_threshold,
        }
        # Also request on a session-less turn (filter reload, eviction,
        # OWUI restart mid-chat) so static context can be rebuilt.
        if is_first or session_id is None:
            payload["include_instructions"] = True
            payload["managed"] = True

        await self._debug(
            __event_emitter__,
            f"Calling /api/recall: mode={search_mode} "
            f"session_id={session_id!r} is_first={is_first}",
        )

        result = await self._post("/api/recall", payload, __user__, __event_emitter__)

        if result:
            stats = result.get("stats", {})
            await self._debug(
                __event_emitter__,
                f"Recall response: user_id={user_id!r} "
                f"session={result.get('session_id', '?')!r} "
                f"core={stats.get('core_count', 0)} "
                f"search={stats.get('search_count', 0)} "
                f"new={stats.get('new_count', 0)} "
                f"skipped={stats.get('known_skipped', 0)} "
                f"has_instructions={bool(result.get('instructions'))} "
                f"has_core={bool(result.get('core_memories'))} "
                f"latency={stats.get('latency_ms', 0)}ms",
            )
        else:
            await self._debug(__event_emitter__, "Recall returned None (API error)")

        # Update session tracking
        if result and result.get("session_id"):
            # Cache instructions + core memories as static context.
            # This text is re-injected at a fixed early position on
            # every subsequent turn so it becomes part of the stable
            # prompt prefix (good for server-side prompt caching).
            #
            # Rebuild ONLY when there is nothing cached yet.  Rebuilding
            # mid-chat swaps different text into an early position and
            # invalidates the entire cached prefix, forcing a full
            # re-prefill.  is_first is deliberately NOT a trigger: it
            # fires on regenerates and message edits too.
            static_ctx = None
            existing_ctx = (sess.get("static_ctx") if sess else None) or (
                self._static_by_user.get(user_id)
            )
            rebuild = not existing_ctx and bool(
                result.get("core_memories") or result.get("instructions")
            )
            if rebuild:
                static_parts = []
                if result.get("instructions"):
                    static_parts.append(self._MEMORY_INSTRUCTIONS)
                if result.get("core_memories"):
                    static_parts.append(result["core_memories"])
                static_ctx = "\n\n".join(static_parts) if static_parts else None
                if static_ctx:
                    static_ctx = self._sanitize_static_ctx(static_ctx) or None
                if static_ctx and user_id:
                    self._static_by_user[user_id] = static_ctx
                await self._debug(
                    __event_emitter__,
                    f"Cached static_ctx: {len(static_ctx) if static_ctx else 0} chars "
                    f"(instructions={bool(result.get('instructions'))}, "
                    f"core_memories={bool(result.get('core_memories'))})",
                )

            if is_first:
                # Fresh conversation — drop any stale carry bucket.
                self._recalled.pop(result["session_id"], None)

            self._save_session(
                chat_id,
                result["session_id"],
                static_ctx,
                update_ctx=rebuild,
                user_id=user_id,
            )
            # Re-read session after save so we have the latest data
            sess = self._get_session(chat_id, user_id)

        # --- Inject context into messages ---
        #
        # Two-position injection for optimal prompt caching:
        #
        # 1. STATIC CONTEXT (instructions + core memories):
        #    Inserted at a fixed early position — after the initial
        #    system message(s), before the first user message.  This
        #    becomes part of the stable, cacheable prompt prefix:
        #      [sys_prompt] [STATIC_CTX] [user_1] [asst_1] [user_2] ...
        #    Cached from the first turn onward; never re-processed.
        #
        # 2. DYNAMIC CONTEXT (recalled memories):
        #    Appended after the last user message.  Grows as new
        #    memories surface, so it sits outside the cached prefix.
        #    Accumulated across turns — see _carry_recalled().
        #
        # This is strictly better for caching than appending everything
        # after the last user message, because the static context
        # (often 1-2k tokens) is cached instead of being re-processed
        # on every turn.

        # 1. Static context — always inject from cache
        has_static = bool(
            (sess and sess.get("static_ctx")) or self._static_by_user.get(user_id)
        )
        self._inject_static_context(body, sess, user_id)
        await self._debug(
            __event_emitter__,
            f"Static context injected: {has_static}",
        )

        # 2. Dynamic context — this turn's results plus everything
        #    already recalled in this session.  The server withholds
        #    already-sent memories (known_skipped) assuming the model
        #    still has them, but Open WebUI does not persist injected
        #    system messages, so we re-inject them ourselves.
        sid = (result or {}).get("session_id") or (sess["session_id"] if sess else None)
        if sid and self.valves.recall_carry_max > 0:
            memories = self._carry_recalled(sid, (result or {}).get("search_results"))
            if memories:
                self._inject_recalled(body, memories)
                await self._debug(
                    __event_emitter__,
                    f"Injected {len(memories)} recalled memories "
                    f"(cap={self.valves.recall_carry_max})",
                )
        elif result and result.get("search_results"):
            # Carry-forward disabled — upstream behaviour: this turn only.
            self._inject_recalled(
                body,
                [m["memory"] for m in result["search_results"] if m.get("memory")],
            )

        # Show detailed status with stats
        if __event_emitter__ and show_status:
            desc = self._build_status(result, is_first)
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": desc, "done": True},
                }
            )
        return body

    def _inject_static_context(
        self, body: dict, sess: dict | None, user_id: str = ""
    ) -> None:
        """Inject cached static context at a fixed early position.

        Inserts after the initial system message(s), before the first
        user message.  This keeps the static context as part of the
        stable prompt prefix for LLM prompt caching.

        Falls back to the per-user cache when the per-chat session is
        unavailable, and is idempotent: if an identical block is
        already present the message list is left untouched.  Both
        guards exist because a static block that appears on one turn
        and not the next changes the prefix and costs a full
        re-prefill of the whole conversation.
        """
        ctx = (sess or {}).get("static_ctx") or self._static_by_user.get(user_id)
        if not ctx:
            return
        # Drop hygiene-violating bullet lines (placeholder leaks etc.).
        # Idempotent and monotonic per cached block, so the prompt
        # prefix never flickers between turns.
        ctx = self._sanitize_static_ctx(ctx)
        if not ctx:
            return
        if user_id:
            self._static_by_user[user_id] = ctx

        messages = body.get("messages", [])
        if any(m.get("role") == "system" and m.get("content") == ctx for m in messages):
            return
        # Find insertion point: after consecutive system messages at
        # the start of the conversation.
        insert_idx = 0
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                insert_idx = i + 1
            else:
                break
        messages.insert(
            insert_idx,
            {
                "role": "system",
                "content": ctx,
            },
        )

    def _sanitize_static_ctx(self, ctx: str) -> str:
        """Drop broken bullet lines and cap each section's rendered size.

        mnemory wraps every stored memory in ⟨memory_item⟩…⟨/memory_item⟩
        framing, and _TAG_LEAK_RE treats that tag as a leaked boundary marker, so
        all core bullets failed the hygiene check and the sections rendered as
        bare headers (measured: 60 dropped of 60). The wrapper is stripped before
        testing because it is legitimate framing, not a leak. The tags themselves
        stay in the output: the instructions block tells the model to treat
        tagged content as data, which is the prompt-injection guard on memories.

        The cap comes from the core_max_per_section valve (0 = unlimited).
        mnemory already orders pinned memories first, so a cap sheds
        low-importance top-N entries before it ever reaches a pinned one.
        """
        # Double getattr: a filter instance without valves must degrade to uncapped,
        # not raise inside inlet() and take every chat turn down with it.
        cap = int(getattr(getattr(self, "valves", None), "core_max_per_section", 0) or 0)
        out: list[str] = []
        seen = 0
        for line in ctx.split("\n"):
            stripped = line.lstrip()
            if stripped.startswith("## "):
                seen = 0
                out.append(line)
                continue
            if stripped.startswith("- ") or stripped.startswith("* "):
                probe = self._MEMORY_ITEM_WRAPPER_RE.sub("", stripped[2:])
                if self._hygiene_violation(probe):
                    continue
                seen += 1
                if cap and seen > cap:
                    continue
            out.append(line)
        # Trim lines that are now empty due to bullet removal
        return "\n".join(out).strip()

    def _inject_recalled(self, body: dict, memories: list[str] | None) -> None:
        """Append recalled memories to the last user message.

        NOT a system message: Open WebUI merges all system-role messages
        into messages[0], which would move this volatile block ahead of
        the whole conversation and invalidate the server's prompt cache
        every turn.

        Defense-in-depth: hygiene-violating entries are dropped here too
        (the carry bucket is normally already clean).
        """
        if not memories:
            return
        memories = [m for m in memories if not self._hygiene_violation(m)]
        if not memories:
            return
        block = "\n\n## Recalled Memories\n" + "\n".join(f"- {m}" for m in memories)
        messages = body.setdefault("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    msg["content"] = content + block
                elif isinstance(content, list):
                    content.append({"type": "text", "text": block})
                return
        messages.append({"role": "system", "content": block.strip()})

    @staticmethod
    def _build_status(result: dict | None, is_first: bool) -> str:
        """Build a detailed status message from recall stats."""
        if not result:
            return "Memory unavailable"

        stats = result.get("stats", {})
        ms = stats.get("latency_ms", 0)
        core = stats.get("core_count", 0)
        new = stats.get("new_count", 0)

        if is_first:
            if core and new:
                return f"Recalled {core} core + {new} relevant memories ({ms}ms)"
            if core:
                return f"Recalled {core} core memories ({ms}ms)"
            if new:
                return f"Found {new} relevant memories ({ms}ms)"
            return f"Memory ready ({ms}ms)"

        # Subsequent call
        if new:
            return f"Found {new} new memories ({ms}ms)"
        return f"No new memories ({ms}ms)"

    # ── Outlet (after LLM) ───────────────────────────────────────────

    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable] = None,
    ) -> dict:
        """After LLM: store memories from the exchange (fire-and-forget)."""
        if not __user__:
            return body

        user_valves = __user__.get("valves")
        if user_valves and hasattr(user_valves, "enabled") and not user_valves.enabled:
            return body

        chat_id = body.get("chat_id") or ""
        user_id = __user__.get("email", __user__.get("id", ""))
        sess = self._get_session(chat_id, user_id)
        session_id = sess["session_id"] if sess else None
        messages = body.get("messages", [])

        # Only user/assistant messages — exclude system prompts and tool results
        conversation = [m for m in messages if m.get("role") in ("user", "assistant")]

        if len(conversation) < 2:
            await self._debug(
                __event_emitter__,
                f"Outlet: skipping, only {len(conversation)} messages",
            )
            return body

        # Last 2 user/assistant messages (current exchange)
        last_two = conversation[-2:]

        # Build context from the first user message to give the extraction
        # LLM topic awareness. Without this, memories extracted from the
        # last exchange can be vague (e.g., "User wants to search the web")
        # because the LLM doesn't know what the conversation is about.
        context = None
        first_user_msg = next(
            (
                m.get("content", "")
                for m in conversation
                if m.get("role") == "user" and m.get("content")
            ),
            None,
        )
        if first_user_msg and isinstance(first_user_msg, str):
            # Cap context to avoid sending huge first messages
            context = f"Conversation topic: {first_user_msg[:500]}"

        await self._debug(
            __event_emitter__,
            f"Outlet: session={session_id!r} msgs={len(last_two)} chat_id={chat_id!r}",
        )

        # Fire-and-forget
        payload: dict = {"session_id": session_id, "messages": last_two}
        if context:
            payload["context"] = context
        # Attach labels for provenance tracking (chat_id links memories
        # to a specific conversation, source identifies the client)
        labels: dict[str, str] = {"source": "open-webui"}
        if chat_id:
            labels["chat_id"] = chat_id
        payload["labels"] = labels
        asyncio.create_task(
            self._post("/api/remember", payload, __user__, __event_emitter__)
        )

        return body


def _ml_norm(t):
    return re.sub(r"\s+", " ", t or "").strip().lower()


def _ml_tokens(t):
    """Distinctive content words: lowercased, edge punctuation and template
    words removed. Edge punctuation matters - 'stamps.' must yield {stamps} or
    the containment test misses every memory that ends in a full stop."""
    out = set()
    for w in re.findall(r"[a-z0-9][a-z0-9.+\-_@/]*", (t or "").lower()):
        w = re.sub("^[^0-9a-z]+|[^0-9a-z]+$", "", w)
        if w and w not in Filter._EX_STOP:
            out.add(w)
    return out


try:
    _ML_EXAMPLE_TOKENS = tuple(tk for tk in map(_ml_tokens, Filter._EXAMPLE_OUTPUTS) if tk)
except Exception:                                   # never break the filter on import
    _ML_EXAMPLE_TOKENS = ()


def _example_leak(text):
    """True when an example's distinctive tokens all reappear in `text`.

    'User prefers cooking and collecting stamps.' is dropped because the example
    contributes {cooking, collecting, stamps} and all three are present.
    'Assistant\'s name is Atlas.' survives even though the examples
    'Assistant\'s name is Bob/Aria' share its template, because those examples'
    distinctive token is {bob} / {aria, research} and neither is present.
    Whole-sentence similarity cannot tell those two cases apart.
    """
    if not text or not _ML_EXAMPLE_TOKENS:
        return False
    tk = _ml_tokens(text)
    if not tk:
        return False
    for et in _ML_EXAMPLE_TOKENS:
        if et <= tk:
            return True
    return False
