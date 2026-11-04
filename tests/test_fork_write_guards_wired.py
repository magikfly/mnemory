"""LOCAL-PATCH tests 2026-11-05: write-side guards must actually be wired.

Failure class measured, not assumed. MEMORY_AGENT_VOICE=drop has been set in
compose since 2026-10-30, yet six assistant-narration rows were written from one
ops conversation on 2026-11-04 and had to be purged. Cause: the guard is called
from parse_extraction_response, while the chat pipeline parses with
parse_remember_extraction_response, which ran no content guard at all. A guard
that exists, is configured to enforce, and is not on the code path is invisible
in every config dump - only the store shows it.

These tests exercise the real parsers with real payloads. Synthetic content only
(the repo is public).
"""
import json
import pathlib
import sys

import pytest

from mnemory.output_guards import (
    agent_voice,
    launders_user_decision,
    role_mismatch,
)
from mnemory.prompts import parse_remember_extraction_response


def reply_for(entries):
    return json.dumps({"memories": entries, "summary": "Session summary."})


NARRATION_NORMAL = {
    "text": "Everything is aligned and the image is refreshed",
    "memory_type": "episodic", "categories": ["technical"],
    "importance": "normal", "pinned": False, "role": "assistant",
    "event_date": None,
}
NARRATION_HIGH = {
    "text": "Assistant retracted a pinned fact after a false premise about the store",
    "memory_type": "episodic", "categories": ["technical"],
    "importance": "high", "pinned": False, "role": "assistant",
    "event_date": None,
}
LEGIT_USER = {
    "text": "User's homelab runs on an Unraid server at 192.0.2.10",
    "memory_type": "fact", "categories": ["technology"],
    "importance": "high", "pinned": False, "role": "user",
    "event_date": None,
}
LEGIT_PERSONA = {
    "text": "Assistant's name is Atlas",
    "memory_type": "fact", "categories": ["personal"],
    "importance": "critical", "pinned": True, "role": "assistant",
    "event_date": None,
}


class TestRememberParserIsGuarded:
    def _texts(self, entries):
        facts, _summary, _art = parse_remember_extraction_response(
            reply_for(entries), silent=True)
        return [f["text"] for f in facts]

    def test_drop_mode_refuses_assistant_voice(self, monkeypatch):
        monkeypatch.setenv("MEMORY_AGENT_VOICE", "drop")
        got = self._texts([NARRATION_NORMAL, LEGIT_USER])
        assert "User's homelab runs on an Unraid server at 192.0.2.10" in got
        assert "Everything is aligned and the image is refreshed" not in got

    def test_importance_cannot_rescue_narration(self, monkeypatch):
        """The escape hatch three of the six purged rows used."""
        monkeypatch.setenv("MEMORY_AGENT_VOICE", "drop")
        got = self._texts([NARRATION_HIGH, LEGIT_USER])
        assert not any(t.startswith("Assistant retracted") for t in got)

    def test_persona_and_user_facts_survive(self, monkeypatch):
        monkeypatch.setenv("MEMORY_AGENT_VOICE", "drop")
        got = self._texts([LEGIT_PERSONA, LEGIT_USER, NARRATION_NORMAL])
        assert "Assistant's name is Atlas" in got
        assert any("Unraid" in t for t in got)
        assert len(got) == 2

    def test_shadow_mode_keeps_everything(self, monkeypatch):
        """Enforcement is env-gated; unset must not start deleting."""
        monkeypatch.setenv("MEMORY_AGENT_VOICE", "shadow")
        got = self._texts([NARRATION_NORMAL, LEGIT_USER])
        assert len(got) == 2

    def test_parser_was_unguarded_before_this_change(self):
        """Pin the wiring so an upstream merge cannot silently drop it again.

        Defect A was exactly this: the guard function and its config switch both
        looked healthy in every review; only the call site was missing.
        """
        pkg = pathlib.Path(sys.modules["mnemory.prompts"].__file__).parent
        src = (pkg / "prompts.py").read_text(encoding="utf-8")
        start = src.index("def parse_remember_extraction_response(")
        body = src[start:src.index("def build_dedup_prompt(")]
        assert "agent_voice(" in body, "chat parser lost the agent-voice guard"
        assert "triage_memory(" in body, "chat parser lost the noise/triage guard"


class TestAgentVoiceImportance:
    def test_importance_no_longer_exempts_assistant_voice(self):
        """The hatch three of the six purged rows used is gone."""
        assert agent_voice(
            "assistant", "episodic", False, "critical", categories=["technical"]
        ) == "assistant_agent_voice"

    def test_pinned_is_still_exempt(self):
        assert agent_voice(
            "assistant", "episodic", True, "critical", categories=["technical"]
        ) is None

    def test_life_category_is_still_exempt(self):
        assert agent_voice(
            "assistant", "episodic", False, "high", categories=["health"]
        ) is None

    def test_durable_types_are_still_exempt(self):
        assert agent_voice(
            "assistant", "fact", False, "high", categories=["technical"]
        ) is None


class TestLaundersUserDecision:
    SUMMARY = (
        "The session covered the irrigation agent configuration. User decided on "
        "temperature 0.4, 12288 tokens and reasoning on for the irrigation agent. "
        "User also chose to keep the pipeline offline."
    )
    LAUNDERED = ("Assistant configured the irrigation agent with temperature 0.4, "
                 "12288 tokens, and enabled reasoning")

    def test_fires_on_the_measured_output(self):
        assert launders_user_decision(
            self.LAUNDERED, self.SUMMARY, "assistant") == "assistant_pass_user_decision"

    def test_old_guard_does_not_see_it(self):
        """Why this rule had to exist: the shipped guard returns nothing here."""
        assert role_mismatch(self.LAUNDERED, "assistant") is None

    def test_unrelated_assistant_work_is_kept(self):
        assert launders_user_decision(
            "Assistant fixed the prefill indicator by anchoring the spinner to "
            "the content container",
            "User decided to use PostgreSQL for billing.", "assistant") is None

    def test_numbers_alone_do_not_convict(self):
        assert launders_user_decision(
            "Assistant verified test results of 7 of 7 passing",
            "User decided to use PostgreSQL for billing.", "assistant") is None

    def test_user_role_is_not_applicable(self):
        assert launders_user_decision(
            self.LAUNDERED, self.SUMMARY, "user") is None

    def test_summary_without_a_user_decision_is_inert(self):
        assert launders_user_decision(
            self.LAUNDERED, "The session configured an irrigation agent.",
            "assistant") is None


class TestConsolidationWiring:
    def test_consolidation_calls_the_provenance_rule(self):
        pkg = pathlib.Path(sys.modules["mnemory.output_guards"].__file__).parent
        src = (pkg / "consolidation.py").read_text(encoding="utf-8")
        assert "launders_user_decision(" in src
        assert "role_provenance_enforcing()" in src

    def test_rule_ships_shadow_by_default(self, monkeypatch):
        from mnemory.output_guards import role_provenance_enforcing

        monkeypatch.delenv("MEMORY_ROLE_PROVENANCE", raising=False)
        assert role_provenance_enforcing() is False
        monkeypatch.setenv("MEMORY_ROLE_PROVENANCE", "drop")
        assert role_provenance_enforcing() is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
