"""Guard tests for the 2026-10-27 memory-quality fixes.

Why each exists:
  * find_memories() applied SEARCH_SCORE_THRESHOLD - a constant calibrated for
    vector/RRF scores - to the rerank LLM's 0-1 rubric, and accepted no caller
    threshold at all, so the OWUI filter's recall_score_threshold valve was a
    silent no-op on the find path.
  * a candidate the rerank model failed to emit an idx for was DELETED, even
    though the rerank prompt commands "Include ALL memories in the response.
    Do not omit any". A small-model formatting failure became silent data loss,
    indistinguishable from a legitimate "scored irrelevant" in the drop counter.
  * the service computed searched/merged/reranked/dropped/returned but the REST
    response model had no field for it, so rerank attrition could only be read
    off the server log (measured: 53.7 candidates -> 5.8 returned).
  * list_categories / _get_available_categories / delete_all_memories scanned
    with limit=1000, so counts under-reported (999 claimed against 2,694 real)
    and agent-scoped deletes orphaned artifacts past row 1000.
  * nothing anywhere could remove expired rows: every fsck phase passes
    exclude_expired=True while query time also excludes them, so 650 corpses
    (24% of the active surface) accumulated, oldest 77 days past expiry.

The source-level guards are deliberate tripwires: they fail loudly if someone
reverts a block instead of fixing the behaviour.
"""
import inspect
import pathlib
import re

import pytest

pytest.importorskip('pydantic')

REPO = pathlib.Path(__file__).resolve().parents[1]


def _code_only(src: str) -> str:
    """Drop comment lines so an explanatory comment that quotes the old bad
    value cannot satisfy (or defeat) a 'this literal is gone' assertion."""
    return '\n'.join(l for l in src.splitlines() if not l.strip().startswith('#'))


def _own_body(fn) -> str:
    """Source of fn only - stops the window leaking into the next method."""
    src = inspect.getsource(fn)
    nl = src.index(chr(10))
    m = re.search(r'\n(    def |class )', src[nl + 1:])
    return src[: nl + 1 + m.start()] if m else src


def _find_src():
    from mnemory.memory import MemoryService
    return _code_only(inspect.getsource(MemoryService.find_memories))


def test_find_memories_accepts_caller_threshold():
    from mnemory.memory import MemoryService
    params = inspect.signature(MemoryService.find_memories).parameters
    assert 'score_threshold' in params, (
        'find_memories() must accept score_threshold; without it the server '
        'substitutes the vector-tuned SEARCH_SCORE_THRESHOLD and the caller '
        '(the OWUI recall filter) cannot express a relevance floor at all'
    )


def test_find_memories_threshold_precedence():
    assert 'if score_threshold is not None' in _find_src(), (
        'a caller-supplied threshold must beat the config default'
    )


def test_omitted_rerank_indices_are_retained_not_deleted():
    src = _find_src()
    assert 'unscored.append(mem)' in src, (
        'candidates the rerank LLM omitted must be retained at the score floor; '
        'deleting them contradicts the rerank prompt and loses memories silently'
    )
    assert 'rerank LLM omitted' in src, 'model omissions must be logged, not silent'


def test_request_schema_exposes_score_threshold():
    from mnemory.api.schemas import FindMemoriesRequest
    assert 'score_threshold' in FindMemoriesRequest.model_fields
    with pytest.raises(Exception):
        FindMemoriesRequest(question='x', score_threshold=1.5)


def test_response_schema_can_carry_stats():
    from mnemory.api.schemas import SearchMemoriesResponse
    assert 'stats' in SearchMemoriesResponse.model_fields, (
        'retrieval accounting must reach API clients, not just the log'
    )


def test_find_route_forwards_threshold_and_returns_stats():
    src = (REPO / 'mnemory' / 'api' / 'memories.py').read_text()
    assert 'score_threshold=req.score_threshold' in src
    assert 'stats=result.get("stats")' in src


def test_list_categories_does_not_truncate_at_1000():
    from mnemory.memory import MemoryService
    src = _code_only(_own_body(MemoryService.list_categories))
    assert 'limit=1000' not in src, (
        'a hard 1000-row scan makes every category count - and the reported '
        'store total - a fraction of reality'
    )
    assert 'TRUNCATED' in src, 'if a scan limit is ever hit it must be logged'


def test_no_silent_1000_row_scans_remain_in_memory_service():
    src = _code_only((REPO / 'mnemory' / 'memory.py').read_text())
    hits = [i for i, l in enumerate(src.splitlines(), 1) if 'limit=1000' in l]
    assert not hits, 'silent 1000-row scans remain (%d site(s))' % len(hits)


def test_ttl_reaper_phase_exists_and_is_scheduled():
    """The reaper must be BOTH defined and called from run_check - a phase that
    exists but is never invoked is how the expired-row backlog survived in the
    first place."""
    from mnemory import fsck
    runner = [c for _, c in inspect.getmembers(fsck, inspect.isclass)
              if hasattr(c, '_phase_ttl_reap')]
    assert runner, 'no fsck class implements _phase_ttl_reap'
    body = _code_only(inspect.getsource(runner[0].run_check))
    assert '_phase_ttl_reap(' in body, 'the TTL reaper is not wired into run_check'


def test_ttl_reaper_guards_pinned_evidence_and_durable_types():
    from mnemory import fsck
    runner = [c for _, c in inspect.getmembers(fsck, inspect.isclass)
              if hasattr(c, '_phase_ttl_reap')][0]
    body = _code_only(inspect.getsource(runner._phase_ttl_reap))
    assert 'pinned' in body, 'pinned memories must never be auto-reaped'
    assert 'evidence_root_ids' in body, 'memories cited as evidence must be held'
    assert 'fact' in body and 'preference' in body, (
        'durable types need a human decision, not an automated retraction'
    )
