"""Fork tests: fsck's duplicate pass must cover an oversized cluster.

Written after the owner asked why the 4-hour duplicate check had not fished out
the 2026-11-05 flood. It had tried: "6 clusters from 207 memories". The code then
dropped every member past _MAX_CLUSTER_SIZE=15 without a log, so most of the
flood never reached evaluation. These tests pin the coverage, not the wording.
"""
from mnemory.fsck import FsckCheck

try:
    from tests.test_fsck import _make_fsck_service, _make_memory
except ImportError:  # run as a package
    from .test_fsck import _make_fsck_service, _make_memory


def _all_similar(n):
    """n memories, every one similar to every other: one union-find cluster."""
    memories = [_make_memory(mid="m%d" % i, text="User fact number %d" % i)
                for i in range(n)]
    mem_by_id = {m["id"]: m for m in memories}
    svc = _make_fsck_service()

    def search(vector, user_id=None, agent_id=None, limit=None,
               exclude_ids=None, score_threshold=None):
        skip = set(exclude_ids or ())
        return [{"id": m["id"], "score": 0.99}
                for m in memories if m["id"] not in skip]

    svc._vector.search_by_vector_ids.side_effect = search
    return memories, mem_by_id, svc


class TestOversizedClusterCoverage:
    def test_40_member_cluster_is_split_not_truncated(self):
        memories, mem_by_id, svc = _all_similar(40)
        seen = []
        svc._evaluate_duplicate_cluster = lambda mems: seen.append(len(mems)) or []

        check = FsckCheck(check_id="c-split", user_id="filip")
        _issues, clustered = svc._phase_duplicate_detection(
            memories, mem_by_id, check)

        # 15 + 15 + 10: every member reaches an LLM call. Before this change the
        # cluster was cut to its first 15 and 25 members vanished silently.
        assert sorted(seen, reverse=True) == [15, 15, 10]
        assert len(clustered) == 40

    def test_call_count_is_measured_not_estimated(self):
        memories, mem_by_id, svc = _all_similar(40)
        svc._evaluate_duplicate_cluster = lambda mems: []

        check = FsckCheck(check_id="c-calls", user_id="filip")
        svc._phase_duplicate_detection(memories, mem_by_id, check)
        assert check.llm_calls == 3

    def test_small_cluster_still_costs_one_call(self):
        memories, mem_by_id, svc = _all_similar(3)
        seen = []
        svc._evaluate_duplicate_cluster = lambda mems: seen.append(len(mems)) or []

        check = FsckCheck(check_id="c-small", user_id="filip")
        _issues, clustered = svc._phase_duplicate_detection(
            memories, mem_by_id, check)
        assert seen == [3]
        assert len(clustered) == 3
        assert check.llm_calls == 1

    def test_exactly_at_the_cap_is_not_split(self):
        memories, mem_by_id, svc = _all_similar(15)
        seen = []
        svc._evaluate_duplicate_cluster = lambda mems: seen.append(len(mems)) or []

        check = FsckCheck(check_id="c-cap", user_id="filip")
        svc._phase_duplicate_detection(memories, mem_by_id, check)
        assert seen == [15]
        assert check.llm_calls == 1

    def test_one_over_the_cap_splits_and_keeps_the_tail_visible(self):
        memories, mem_by_id, svc = _all_similar(16)
        seen = []
        svc._evaluate_duplicate_cluster = lambda mems: seen.append(len(mems)) or []

        check = FsckCheck(check_id="c-cap1", user_id="filip")
        _issues, clustered = svc._phase_duplicate_detection(
            memories, mem_by_id, check)
        # The 16th member lands in a chunk of one, which is not a cluster and is
        # not sent to the LLM - but it IS enumerated in clustered_ids now, where
        # before the whole tail was dropped without a trace. It merges on a later
        # pass; it is no longer invisible.
        assert seen == [15]
        assert len(clustered) == 16
        assert check.llm_calls == 1

    def test_failed_cluster_still_counts_against_the_budget(self):
        """A wasted call is a spent call; the budget must see it."""
        memories, mem_by_id, svc = _all_similar(30)

        def boom(mems):
            raise RuntimeError("llm exploded")

        svc._evaluate_duplicate_cluster = boom
        check = FsckCheck(check_id="c-fail", user_id="filip")
        issues, _clustered = svc._phase_duplicate_detection(
            memories, mem_by_id, check)
        assert issues == []
        assert check.llm_calls == 2
