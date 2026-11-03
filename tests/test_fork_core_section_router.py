"""Fork tests: content-shaped core-section routing and build-reference staleness.

Both features decide what lands in EVERY prompt, so both are asserted on rows
taken from the live store census of 2026-11-04 (paraphrased, no personal data):
the misrouting these tests pin is the one that was shipping.

Run inside the image (mnemory.memory imports qdrant_client):
    python3 -m pytest tests/test_fork_core_section_router.py -q
"""
import pytest

from mnemory.memory import MemoryService
from mnemory.prompts import _correct_build_staleness


def row(text, *, role="assistant", memory_type="fact", importance="high"):
    return {
        "id": "x",
        "memory": text,
        "metadata": {
            "role": role,
            "memory_type": memory_type,
            "importance": importance,
        },
    }


def section(text, **kw):
    return MemoryService._classify_agent_memory(None, row(text, **kw))[0]


class TestCoreSectionRouting:
    def test_name_row_is_identity(self):
        assert section("Assistant's name is Atlas") == "identity"

    def test_curly_apostrophe_name_row_is_identity(self):
        assert section("Assistant’s name is Atlas") == "identity"

    def test_hardware_note_is_not_identity(self):
        # the shipping bug: role=assistant + type=fact was enough for identity
        text = (
            "User uses an Atlas Ezo pH carrier board in I2C mode. The Atlas Ezo "
            "pH datasheet explicitly requires an external 4.7k pull-up resistor "
            "on the SDA line."
        )
        assert section(text) == "knowledge"

    def test_build_fix_note_is_not_identity(self):
        text = (
            "Assistant fixed second gate bug: workflow now always builds on push "
            "events for image-defining files."
        )
        assert section(text) == "knowledge"

    def test_project_hardware_spec_is_not_identity(self):
        text = (
            "Project Maya requirements: Linux x86-64, NVMe mandatory, 32 GB RAM "
            "minimum, CUDA 12.x, CC 7.0+ GPUs."
        )
        assert section(text) == "knowledge"

    def test_car_registration_is_not_an_instruction(self):
        # the other shipping bug: any agent-scoped role=user row was an instruction
        text = (
            "User registered their 2021 Kia EV6 in the Netherlands with a maximum "
            "mass of 2425 kg and a seating capacity of 5."
        )
        assert section(text, role="user") == "knowledge"

    def test_standing_language_rule_is_an_instruction(self):
        text = (
            "User requires the assistant to always reply in whatever language the "
            "user addresses them in."
        )
        assert section(text, role="user", memory_type="procedural") == "instructions"

    def test_hard_limit_is_an_instruction(self):
        text = (
            "User set hard limits: never reboot or restart containers, never edit "
            "config files."
        )
        assert section(text, role="user", memory_type="preference") == "instructions"

    def test_one_off_request_is_not_an_instruction(self):
        text = "User asked the assistant to write the workflow to ComfyUI."
        assert section(text, role="user", memory_type="episodic") == "knowledge"

    def test_narrated_ask_is_not_an_instruction(self):
        text = (
            "Assistant asked if User wants Assistant to write the patch and run "
            "tests."
        )
        assert section(text, memory_type="fact") == "knowledge"

    def test_instructions_section_is_reachable(self):
        # negative control: a router that can never emit 'instructions' is
        # silently deleting the section, so assert it still fires at all
        out = {
            section(t, role=r, memory_type=m)
            for t, r, m in (
                ("User wants the assistant to never print passwords.", "user", "preference"),
                ("Assistant must keep answers concise.", "assistant", "procedural"),
            )
        }
        assert "instructions" in out

    def test_identity_section_is_not_emptied(self):
        # second negative control: the fix must not delete identity outright
        assert section("Assistant's name is Atlas") == "identity"

    def test_episodic_row_cannot_be_an_instruction(self):
        text = "Assistant should check the vault next time, noted during review."
        assert section(text, memory_type="episodic") == "knowledge"

    # --- first-person forms: the defect my own third-person fixtures missed ---

    def test_first_person_name_is_identity(self):
        assert section("My name is Bob") == "identity"

    def test_subagent_persona_sentence_is_identity(self):
        # the documented sub-agent storage shape from the memory tool contract
        assert section("I am Bob. I speak casually and use humor.") == "identity"

    def test_first_person_work_note_is_not_identity(self):
        # same opening words, no self-designation: must stay knowledge
        assert section("I am working on the dashboard rebuild.") == "knowledge"

    def test_first_person_role_statement_is_identity(self):
        assert section("I am the code review assistant.") == "identity"

    # --- upstream contract preserved: preference in agent scope is a rule ---

    def test_preference_row_in_agent_scope_is_an_instruction(self):
        text = "User wants me to create commit messages"
        assert section(text, role="user", memory_type="preference") == "instructions"

    def test_legacy_row_without_role_falls_back_to_instructions(self):
        memory = {
            "id": "x",
            "memory": "Some old agent memory",
            "metadata": {"memory_type": "fact", "pinned": True},
        }
        got = MemoryService._classify_agent_memory(None, memory)[0]
        assert got == "instructions"


BUILD_ROW = (
    "Assistant published container image 40d50b32a21f with manifest digest "
    "812a92da65a1aeae0aba0801a376369c04e773875f2ebe6ea2d016d7dc7b76b1."
)


class TestBuildStaleness:
    def test_build_reference_demotes_importance(self):
        assert _correct_build_staleness("high", "episodic", BUILD_ROW) == (
            "low",
            "episodic",
        )

    def test_pinned_row_is_untouched(self):
        assert _correct_build_staleness("high", "episodic", BUILD_ROW, pinned=True) == (
            "high",
            "episodic",
        )

    def test_context_mode_also_retypes(self, monkeypatch):
        monkeypatch.setenv("MEMORY_BUILD_STALENESS", "context")
        assert _correct_build_staleness("high", "episodic", BUILD_ROW) == (
            "low",
            "context",
        )

    def test_off_mode_is_a_noop(self, monkeypatch):
        monkeypatch.setenv("MEMORY_BUILD_STALENESS", "off")
        assert _correct_build_staleness("high", "episodic", BUILD_ROW) == (
            "high",
            "episodic",
        )

    def test_unknown_mode_falls_back_to_importance(self, monkeypatch):
        monkeypatch.setenv("MEMORY_BUILD_STALENESS", "banana")
        assert _correct_build_staleness("high", "episodic", BUILD_ROW)[0] == "low"

    def test_endpoints_are_not_versions(self):
        # the first draft matched these; they are durable network facts
        for text in (
            "Pi-hole runs NATIVE on domo-pi at 192.168.2.25 and is the LAN DNS "
            "and DHCP server.",
            "Project Maya web UI listens on 127.0.0.1:8080 and the API binds "
            "0.0.0.0 only with a key.",
        ):
            assert _correct_build_staleness("high", "fact", text) == ("high", "fact")

    def test_durable_lesson_with_the_word_version_is_kept(self):
        text = (
            "User's CI workflow skips building when the image version tag exists, "
            "causing silent non-deployment of code changes."
        )
        assert _correct_build_staleness("high", "episodic", text) == ("high", "episodic")

    def test_commit_sha_row_demotes(self):
        assert _correct_build_staleness(
            "high", "fact", "User's main branch is at commit 2d47b9a."
        )[0] == "low"

    def test_deployed_version_row_demotes(self):
        text = "Assistant deployed version v1.15.0-patched21 closing the baseline pile."
        assert _correct_build_staleness("high", "episodic", text)[0] == "low"

    def test_low_importance_is_idempotent(self):
        assert _correct_build_staleness("low", "episodic", BUILD_ROW) == ("low", "episodic")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
