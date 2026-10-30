"""LOCAL-PATCH tests 2026-10-30: dedup prompt budget chunking + compaction.

Covers the failure class where a single Stage-2 dedup prompt was built at
~97k tokens against the shared 65536-token --kv-unified pool and both
attempts 400'd, losing the whole remember batch (298 log hits).
"""
import json
import re
import types

from mnemory import memory as memory_mod
from mnemory.prompts import build_dedup_prompt


def _fat_item(i, fact_pad=500, cand_pad=5000):
    return {
        "index": i,
        "text": f"fact {i} " + "y" * fact_pad,
        "candidates": [
            {"id": f"uuid-{i}", "memory": "z" * cand_pad, "score": 0.91}
        ],
    }


class TestDedupPromptCompaction:
    def test_candidate_text_truncated(self):
        item = _fat_item(0)
        messages, _, id_mapping = build_dedup_prompt([item])
        user = messages[1]["content"]
        # full 5000-char candidate must not be echoed
        assert "z" * 450 not in user
        assert "[truncated]" in user
        # id mapping preserved (real UUID target for UPDATE/CONFIRM)
        assert id_mapping["0"]["memory_id"] == "uuid-0"

    def test_short_candidate_not_truncated(self):
        item = {
            "index": 0,
            "text": "User's name is Ada",
            "candidates": [
                {"id": "uuid-1", "memory": "User's name is Ada", "score": 0.99}
            ],
        }
        _, _, id_mapping = build_dedup_prompt([item])
        assert id_mapping["0"]["memory_id"] == "uuid-1"

    def test_compact_json_no_indent_padding(self):
        item = _fat_item(0, fact_pad=10, cand_pad=10)
        messages, _, _ = build_dedup_prompt([item])
        user = messages[1]["content"]
        # separators=(",", ":") leaves no newline+indent inside the echo
        assert '\n    {' not in user


class TestDedupBudgetChunking:
    def _service(self, budget):
        svc = object.__new__(memory_mod.MemoryService)
        svc._config = types.SimpleNamespace(
            memory=types.SimpleNamespace(dedup_max_prompt_chars=budget)
        )
        return svc

    def _stub_llm(self, calls):
        def fake_generate(messages, *, json_schema=None, temperature=None,
                          max_tokens=None, reasoning_effort=None,
                          operation="unknown"):
            calls.append(sum(len(str(m.get("content", ""))) for m in messages))
            content = messages[1]["content"]
            idxs = sorted({int(m) for m in re.findall(r"Fact (\d+):", content)})
            decisions = [
                {"fact_index": i, "action": "ADD"} for i in idxs
            ]
            return json.dumps({"decisions": decisions})

        return types.SimpleNamespace(generate=fake_generate)

    def test_splits_when_over_budget(self):
        svc = self._service(budget=2000)
        calls = []
        svc._llm = self._stub_llm(calls)
        items = [_fat_item(i) for i in range(4)]
        facts = [{"text": it["text"]} for it in items]
        actions = svc._dedup_with_llm(items, facts)
        # 4 fat facts cannot fit one 2000-char prompt: recursion split them
        assert len(calls) >= 2, f"expected split, got {len(calls)} call(s)"
        # each leaf call saw a sub-batch, no facts lost
        assert len(actions) >= 1

    def test_leaf_chunk_still_over_budget_is_sent_anyway(self):
        # a single item over budget must not recurse forever
        svc = self._service(budget=10)
        calls = []
        svc._llm = self._stub_llm(calls)
        items = [_fat_item(0)]
        facts = [{"text": items[0]["text"]}]
        actions = svc._dedup_with_llm(items, facts)
        assert len(calls) == 1
        assert len(actions) == 1

    def test_small_batch_stays_one_call(self):
        svc = self._service(budget=150_000)
        calls = []
        svc._llm = self._stub_llm(calls)
        items = [
            {
                "index": i,
                "text": f"small fact {i}",
                "candidates": [
                    {"id": f"uuid-{i}", "memory": f"small cand {i}", "score": 0.9}
                ],
            }
            for i in range(3)
        ]
        facts = [{"text": it["text"]} for it in items]
        actions = svc._dedup_with_llm(items, facts)
        assert len(calls) == 1
        assert len(actions) == 3

    def test_merged_action_texts_cover_all_facts(self):
        svc = self._service(budget=2000)
        calls = []
        svc._llm = self._stub_llm(calls)
        items = [_fat_item(i, fact_pad=800) for i in range(2)]
        facts = [{"text": it["text"]} for it in items]
        actions = svc._dedup_with_llm(items, facts)
        texts = {a.get("text") for a in actions}
        assert {f["text"] for f in facts} <= texts


class TestDedupBudgetConfig:
    def test_default_budget(self):
        import os
        from mnemory.config import MemoryConfig

        assert MemoryConfig().dedup_max_prompt_chars == 150_000


class TestStrayTargetTolerance:
    """2026-10-30: SKIP entries carrying target_id (schema-forced field)
    must not nuke the whole parse (291 log hits, decided-0-of-N failures)."""

    def test_skip_with_stray_target_keeps_batch(self):
        from mnemory.prompts import parse_dedup_response

        resp = json.dumps(
            {
                "decisions": [
                    {
                        "fact_index": 0,
                        "action": "ADD",
                        "target_id": None,
                        "text": "fact zero text",
                    },
                    {
                        "fact_index": 1,
                        "action": "SKIP",
                        "target_id": "3",
                        "text": "fact one text",
                    },
                ]
            }
        )
        facts = [{"text": "fact zero text"}, {"text": "fact one text"}]
        id_mapping = {
            "3": {"memory_id": "uuid-x", "fact_index": 1}
        }
        actions, decided = parse_dedup_response(
            resp, id_mapping, facts, silent=True
        )
        assert decided == {0, 1}
        # ADD survives, SKIP carries no mutation target
        assert len(actions) >= 1
        assert all(a.get("target_id") is None for a in actions)

    def test_stray_target_never_resolves_to_a_real_mutation(self):
        from mnemory.prompts import parse_dedup_response

        resp = json.dumps(
            {
                "decisions": [
                    {
                        "fact_index": 0,
                        "action": "ADD",
                        "target_id": "9",
                        "text": "fact zero text",
                    }
                ]
            }
        )
        facts = [{"text": "fact zero text"}]
        actions, decided = parse_dedup_response(
            resp, {}, facts, silent=True
        )
        assert decided == {0}
        assert actions and actions[0].get("target_id") is None
