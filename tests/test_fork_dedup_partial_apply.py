"""LOCAL-PATCH tests 2026-11-04: Stage-2 dedup partial apply.

Failure class: _dedup_with_llm raised whenever ANY fact lacked a decision and
discarded the facts the model DID decide. Measured over the 19h after the
2026-11-04 04:00 deploy: 8 incomplete attempts, 4 whole remember batches lost
(17 pct of remember writes), gap shapes decided 1 of 2, 4 of 5 and 10 of 11.
An eleven-fact batch was thrown away over one missing line.

Harness mirrors tests/test_dedup_budget.py: a bare MemoryService with stubbed
config and llm, so no store, no network.
"""
import json
import re
import types

import pytest

from mnemory import memory as memory_mod


def _svc(undecided="add", budget=150_000):
    svc = object.__new__(memory_mod.MemoryService)
    svc._config = types.SimpleNamespace(
        memory=types.SimpleNamespace(
            dedup_max_prompt_chars=budget,
            dedup_undecided=undecided,
        )
    )
    return svc


def _items(n, competitors=None):
    """competitors: set of fact indexes that have a similar existing memory."""
    competitors = set(range(n)) if competitors is None else set(competitors)
    return [
        {
            "index": i,
            "text": "fact number " + str(i),
            "candidates": (
                [{"id": "uuid-" + str(i), "memory": "similar " + str(i), "score": 0.92}]
                if i in competitors
                else []
            ),
        }
        for i in range(n)
    ]


def _facts(items):
    return [
        {
            "text": it["text"],
            "memory_type": "episodic",
            "categories": [],
            "importance": "normal",
            "pinned": False,
            "event_date": None,
            "role": "user",
        }
        for it in items
    ]


def _llm_omitting(omit, calls=None):
    """Fake extractor: answers every fact in the prompt except those in omit."""

    def fake_generate(messages, *, json_schema=None, temperature=None,
                      max_tokens=None, reasoning_effort=None, operation="unknown"):
        if calls is not None:
            calls.append(1)
        idxs = sorted({int(m) for m in re.findall(r"Fact (\d+):", messages[1]["content"])})
        decisions = [
            {"fact_index": i, "action": "ADD"} for i in idxs if i not in omit
        ]
        return json.dumps({"decisions": decisions})

    return types.SimpleNamespace(generate=fake_generate)


class TestPartialApply:
    def test_one_missing_decision_does_not_lose_the_batch(self):
        items = _items(2)
        svc = _svc()
        svc._llm = _llm_omitting({1})
        actions = svc._dedup_with_llm(items, _facts(items))
        assert len(actions) == 2
        assert {a["text"] for a in actions} == {f["text"] for f in _facts(items)}

    @pytest.mark.parametrize("n,omit", [(2, {1}), (5, {4}), (11, {7}), (11, {0, 5})])
    def test_shapes_seen_in_production(self, n, omit):
        items = _items(n)
        svc = _svc()
        svc._llm = _llm_omitting(omit)
        actions = svc._dedup_with_llm(items, _facts(items))
        assert len(actions) == n, "no fact may vanish over a missing decision"

    def test_nothing_decided_still_fails_closed(self):
        items = _items(8)
        svc = _svc()
        svc._llm = _llm_omitting(set(range(8)))
        with pytest.raises(RuntimeError):
            svc._dedup_with_llm(items, _facts(items))

    def test_decided_actions_are_not_rewritten(self):
        """A real UPDATE must survive with its target, not be flattened to ADD."""
        items = _items(2)

        def fake_generate(messages, *, json_schema=None, temperature=None,
                          max_tokens=None, reasoning_effort=None, operation="unknown"):
            return json.dumps({"decisions": [
                {"fact_index": 0, "action": "UPDATE", "target_id": "0"},
            ]})

        svc = _svc()
        svc._llm = types.SimpleNamespace(generate=fake_generate)
        actions = svc._dedup_with_llm(items, _facts(items))
        by_text = {a["text"]: a for a in actions}
        assert by_text["fact number 0"]["action"] == "UPDATE"
        assert by_text["fact number 0"]["target_id"] == "uuid-0"
        assert by_text["fact number 1"]["action"] == "ADD"

    def test_full_decision_set_takes_the_old_path(self):
        items = _items(3)
        svc = _svc()
        calls = []
        svc._llm = _llm_omitting(set(), calls)
        actions = svc._dedup_with_llm(items, _facts(items))
        assert len(actions) == 3
        assert len(calls) == 1, "a complete reply must not be retried"


class TestUndecidedPolicy:
    def test_skip_mode_drops_a_fact_that_has_a_competitor(self):
        # at least one fact must stay decided, or the total-failure path raises
        items = _items(3, competitors={0})
        svc = _svc(undecided="skip")
        svc._llm = _llm_omitting({0})
        actions = svc._dedup_with_llm(items, _facts(items))
        by_text = {a["text"]: a["action"] for a in actions}
        # fact 0 had a similar existing memory and no decision: not stored
        assert by_text["fact number 0"] == "SKIP"
        assert by_text["fact number 1"] == "ADD"
        assert by_text["fact number 2"] == "ADD"

    def test_a_fact_with_no_competitor_is_always_added(self):
        """Nothing to duplicate, so no policy choice is needed."""
        # competitors only on fact 0, which stays decided; the undecided fact 1
        # has no similar memory, so skip mode must not swallow it
        items = _items(3, competitors={0})
        svc = _svc(undecided="skip")
        svc._llm = _llm_omitting({1})
        actions = svc._dedup_with_llm(items, _facts(items))
        by_text = {a["text"]: a["action"] for a in actions}
        assert by_text["fact number 1"] == "ADD"

    def test_add_mode_is_the_default(self):
        import os
        assert os.environ.get("DEDUP_UNDECIDED", "add") == "add"
        items = _items(2)
        svc = _svc()
        svc._llm = _llm_omitting({1})
        actions = svc._dedup_with_llm(items, _facts(items))
        assert [a["action"] for a in actions] == ["ADD", "ADD"]


class TestConfigWiring:
    def test_config_field_exists_with_default(self):
        from mnemory.config import Config

        cfg = Config()
        assert cfg.memory.dedup_undecided in ("add", "skip")

    def test_undecided_metadata_carried_onto_recovered_actions(self):
        items = _items(2)
        svc = _svc()
        svc._llm = _llm_omitting({1})
        facts = _facts(items)
        actions = svc._dedup_with_llm(items, facts)
        recovered = [a for a in actions if a["text"] == "fact number 1"][0]
        assert recovered["role"] == "user"
        assert recovered["memory_type"] == "episodic"
        assert recovered["pinned"] is False
        assert recovered["target_id"] is None
