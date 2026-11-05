"""Fork tests: deterministic duplicate-flood guards in the remember path.

Written against the measured 2026-11-05 incident: one remember call stored 201
rows, 15 unique facts repeated ~17x, after a truncated extractor reply.
"""
import json

from mnemory.llm import salvage_json_objects
from mnemory.memory import (
    MemoryService,
    _batch_is_degenerate,
    _collapse_duplicate_actions,
    _norm_fact_text,
    _verbatim_existing_match,
)

WIFE = "User's wife is named Lisandra Fiorini"
EV6 = "User owns a 2021 Kia EV6 with plate N-456-KX"
BASE_FACTS = [
    WIFE,
    EV6,
    "User's sons are Lucca (14) and Pedro (12)",
    "User prefers conventional commit messages",
    "User has two knee injuries and fully recovered by 2026-10-29",
    "User works as Senior Enterprise Solution Architect",
    "User dropped Pi-hole configuration from mnemory store",
    "User decided to proceed with Option B",
    "User is a freelance professional",
    "User lives in Zoetermeer",
    "User runs a homelab on Unraid",
    "User owns a Red Sea aquarium",
    "User has a Fritz!Box router",
    "User drives an electric car",
    "User tracks training in Intervals.icu",
]


def _flood(cycles=17):
    return [{"text": t, "action": "ADD"} for _ in range(cycles) for t in BASE_FACTS]


class TestNormalisation:
    def test_folds_case_space_and_trailing_punctuation(self):
        assert _norm_fact_text("  User's   wife ") == "user's wife"
        assert _norm_fact_text(WIFE + ".") == _norm_fact_text(WIFE)
        assert _norm_fact_text(None) == ""

    def test_does_not_attempt_semantic_folding(self):
        assert _norm_fact_text(WIFE) != _norm_fact_text(EV6)


class TestIntraBatchCollapse:
    def test_the_measured_flood_collapses_to_unique(self):
        actions = _flood()
        assert len(actions) == 17 * 15
        assert len(_collapse_duplicate_actions(actions)) == 15

    def test_first_occurrence_and_order_survive(self):
        kept = _collapse_duplicate_actions(_flood(3))
        assert [a["text"] for a in kept][:2] == [WIFE, EV6]

    def test_update_and_delete_are_not_touched(self):
        actions = [
            {"text": WIFE, "action": "UPDATE", "target_id": "a"},
            {"text": WIFE, "action": "UPDATE", "target_id": "b"},
            {"text": EV6, "action": "DELETE", "target_id": "c"},
            {"text": EV6, "action": "DELETE", "target_id": "d"},
        ]
        assert len(_collapse_duplicate_actions(actions)) == 4

    def test_case_and_punctuation_variants_collapse(self):
        actions = [
            {"text": WIFE, "action": "ADD"},
            {"text": WIFE.upper() + "  ", "action": "ADD"},
            {"text": WIFE + ".", "action": "ADD"},
        ]
        assert len(_collapse_duplicate_actions(actions)) == 1


class TestVerbatimAgainstExisting:
    def test_same_text_already_stored_forces_a_match(self):
        cand = [{"id": "existing-1",
                 "text": WIFE.lower(),
                 "score": 0.99}]
        assert _verbatim_existing_match({"text": WIFE}, cand) is not None

    def test_paraphrase_is_left_to_the_llm(self):
        cand = [{"id": "x", "text": "Married to Lisandra since 2009", "score": 0.6}]
        assert _verbatim_existing_match({"text": WIFE}, cand) is None

    def test_no_candidates_cannot_match(self):
        assert _verbatim_existing_match({"text": WIFE}, []) is None


class TestDegenerateBatch:
    def test_repeated_batch_is_degenerate(self):
        assert _batch_is_degenerate([a["text"] for a in _flood()]) is True

    def test_distinct_batch_is_not(self):
        texts = ["fact %d about something different" % i for i in range(12)]
        assert _batch_is_degenerate(texts) is False

    def test_small_batches_are_not_judged_on_ratio(self):
        assert _batch_is_degenerate([WIFE, WIFE, EV6, EV6]) is False


class TestSalvageDeduplicates:
    @staticmethod
    def _reply(objects, trailing=True):
        parts = [json.dumps(o) for o in objects]
        body = ", ".join(parts)
        if trailing:
            body += ", " + '{"text": "unfinis'
        return '{"memories": [' + body

    def test_repeated_closed_objects_collapse(self):
        reply = self._reply([{"text": WIFE, "role": "user"},
                             {"text": EV6, "role": "user"}] * 6)
        assert len(salvage_json_objects(reply, "memories")) == 2

    def test_distinct_objects_are_all_kept(self):
        reply = self._reply([{"text": t, "role": "user"} for t in BASE_FACTS])
        assert len(salvage_json_objects(reply, "memories")) == len(BASE_FACTS)


class TestGuardsAreOnTheWritePath:
    """_dedup_with_llm is the single entry point every remember call uses.

    The worker is stubbed so these assert placement, not LLM behaviour: guards
    bolted onto a path nothing calls would pass every unit test above.
    """

    @staticmethod
    def _service(actions, capture):
        svc = MemoryService.__new__(MemoryService)

        def stub(items, facts, *, trusted, force_skip_undecided=False):
            capture["force_skip_undecided"] = force_skip_undecided
            return [dict(a) for a in actions]

        svc._dedup_llm_bounded = stub
        return svc

    def test_duplicate_adds_never_reach_execution(self):
        svc = self._service(_flood(), {})
        items = [{"index": i, "text": t, "candidates": []}
                 for i, t in enumerate(BASE_FACTS)]
        facts = [{"text": t} for t in BASE_FACTS]
        out = svc._dedup_with_llm(items, facts, trusted=True)
        assert len(out) == 15

    def test_verbatim_existing_memory_downgrades_add_to_skip(self):
        actions = [{"text": WIFE, "action": "ADD", "target_id": None}]
        items = [{"index": 0, "text": WIFE,
                  "candidates": [{"id": "mem-1", "text": WIFE.lower(), "score": 0.98}]}]
        out = self._service(actions, {})._dedup_with_llm(
            items, [{"text": WIFE}], trusted=True)
        assert out[0]["action"] == "SKIP"

    def test_degenerate_batch_forces_undecided_to_skip(self):
        capture = {}
        svc = self._service(_flood(), capture)
        items = [{"index": i, "text": t, "candidates": []}
                 for i, t in enumerate(BASE_FACTS)]
        facts = [{"text": t} for _ in range(17) for t in BASE_FACTS]
        svc._dedup_with_llm(items, facts, trusted=True)
        assert capture["force_skip_undecided"] is True

    def test_clean_batch_is_passed_through_unchanged(self):
        capture = {}
        actions = [{"text": "fact %d" % i, "action": "ADD"} for i in range(10)]
        items = [{"index": i, "text": "fact %d" % i, "candidates": []}
                 for i in range(10)]
        out = self._service(actions, capture)._dedup_with_llm(
            items, [{"text": a["text"]} for a in actions], trusted=True)
        assert len(out) == 10
        assert capture["force_skip_undecided"] is False
        assert all(a["action"] == "ADD" for a in out)

    def test_similar_but_distinct_facts_survive(self):
        texts = [
            "User runs the homelab on Unraid",
            "User runs training sessions on Tuesdays",
            "User runs a Red Sea reef aquarium",
            "User runs Ollama for inference",
            "User runs a weekly standup",
            "User runs Pilates on Fridays",
            "User runs a home NAS backup",
            "User runs two Raspberry Pi sensors",
            "User runs the local DNS on domo-pi",
            "User runs interval training weekly",
        ]
        actions = [{"text": t, "action": "ADD"} for t in texts]
        items = [{"index": i, "text": t, "candidates": []}
                 for i, t in enumerate(texts)]
        out = self._service(actions, {})._dedup_with_llm(
            items, [{"text": t} for t in texts], trusted=True)
        assert len(out) == 10
        assert all(a["action"] == "ADD" for a in out)
