"""Canaries for gates/agent-voice.

Two failure directions matter and only one of them is reversible, so the
KEEP-side cases are the important ones: mnemory has no un-retract endpoint.

The KEEP set is drawn from false positives already recorded against this
module (2026-10-18: a top-level alternation ate "agreed"/"note"/"reports" in
ordinary prose and dropped five ordinary user facts in 24h) plus the medical
advice that triage_memory() exists to rescue.
"""
import os

import pytest

from mnemory.output_guards import (
    agent_voice,
    noise_enforcing_for,
    noise_memory,
    triage_memory,
)


class TestUserSpeechAct:
    def test_requested_is_caught(self):
        assert noise_memory(
            "User requested to bump max-file to 5 in docker-compose.yml independently"
        ) == "user_speech_act"

    def test_expressed_frustration_is_caught(self):
        assert noise_memory(
            "User expressed frustration with memory's current state"
        ) == "user_speech_act"

    def test_asked_without_wh_word_is_caught(self):
        # The older _QUESTION_PAST_RE needs asked+wh; "asked about their age"
        # slipped through it and became a stored row.
        assert noise_memory("User asked about their age") == "user_speech_act"

    @pytest.mark.parametrize("text", [
        "User's partner Ada agreed to the move to Examplestad",
        "User's note about knee pain says it started in August",
        "User keeps a training log and reports it weekly to his coach",
        "User decided to pursue a freelance opportunity in the region",
        "User prefers hybrid work over full-time on-site",
        "User needs confirmation that the calibration was run in the morning",
        "User's best pace for runs over 5 km improved from 7:40/km to 6:20/km over 18 months",
    ])
    def test_durable_and_recorded_false_positives_survive(self, text):
        assert noise_memory(text) != "user_speech_act"


class TestAgentVoice:
    def test_transcribed_utterance_flagged(self):
        assert agent_voice("assistant", "episodic", False, "normal") == (
            "assistant_agent_voice"
        )
        assert agent_voice("assistant", "context", False, "normal") == (
            "assistant_agent_voice"
        )

    def test_persona_survives(self):
        assert agent_voice("assistant", "fact", True, "critical") is None
        assert agent_voice("assistant", "fact", False, "normal") is None
        assert agent_voice("assistant", "preference", False, "normal") is None
        assert agent_voice("assistant", "procedural", False, "normal") is None

    def test_user_facts_never_flagged(self):
        assert agent_voice("user", "episodic", False, "normal") is None
        assert agent_voice(None, "episodic", False, "normal") is None

    def test_high_importance_assistant_row_survives(self):
        assert agent_voice("assistant", "episodic", False, "high") is None


class TestPerReasonEnforcement:
    """The new rules must not inherit the live MEMORY_NOISE_GUARD=drop."""

    def test_legacy_reason_follows_global_flag(self, monkeypatch):
        monkeypatch.setenv("MEMORY_NOISE_GUARD", "drop")
        monkeypatch.delenv("MEMORY_NOISE_USER_ACTS", raising=False)
        assert noise_enforcing_for("assistant_speech_act") is True
        assert noise_enforcing_for("greeting_narration") is True

    def test_new_reason_defaults_to_shadow(self, monkeypatch):
        monkeypatch.setenv("MEMORY_NOISE_GUARD", "drop")
        monkeypatch.delenv("MEMORY_NOISE_USER_ACTS", raising=False)
        assert noise_enforcing_for("user_speech_act") is False

    def test_new_reason_can_be_forced(self, monkeypatch):
        monkeypatch.setenv("MEMORY_NOISE_USER_ACTS", "drop")
        assert noise_enforcing_for("user_speech_act") is True


class TestMedicalRescueUntouched:
    """triage_memory() must still rescue substance out of an assistant head.

    Recorded reason for the rescue: enforcing assistant_speech_act as written
    would have deleted real medical guidance that was wrapped in an assistant
    reporting head. Drug names and doses are deliberately fictionalised here -
    this file is in a public repo.
    """

    def test_that_clause_is_kept(self):
        text, reason, rewrote = triage_memory(
            "Assistant confirmed that the user's B12 supplementation is safe "
            "to take at will, oral form, hydroxysoluble"
        )
        assert rewrote is True
        assert reason is None or reason == "assistant_speech_act"
        assert "B12" in text

    def test_advice_clause_survives_head_strip(self):
        text, _reason, rewrote = triage_memory(
            "Assistant recommended not training fasted while on a "
            "sulfonylurea because of hypoglycaemia risk"
        )
        assert rewrote is True
        assert "fasted" in text
        assert "hypoglycaemia" in text
