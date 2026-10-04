"""Managed inference starts before Cognis and follows credential rotation."""

import json
from dataclasses import replace

import httpx
import pytest
from openai import OpenAI

from mnemory.config import LLMConfig, load_config
from mnemory.embeddings import EmbeddingClient
from mnemory.llm import LLMClient


def test_managed_startup_and_rotation(monkeypatch, tmp_path):
    token_file = tmp_path / "token.jwt"
    monkeypatch.setenv("COGNIS_INFERENCE_URL", "http://cognis:8080/v1")
    monkeypatch.setenv("COGNIS_INFERENCE_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("EMBED_BASE_URL", "https://unrelated-upstream.invalid/v1")
    config = load_config()
    assert config.llm.model == "routes/mnemory_extraction"
    assert config.embed.model == "routes/mnemory_embedding"
    assert config.memory.find_model == "routes/mnemory_retrieval"
    assert config.memory.fsck_model == "routes/mnemory_maintenance"
    assert config.memory.consolidation_model == "routes/mnemory_consolidation"
    assert config.embed.base_url == "http://cognis:8080/v1"
    assert LLMConfig(reasoning_effort="high").reasoning_effort == "high"
    LLMClient(config.llm)
    EmbeddingClient(config.embed)
    with pytest.raises(RuntimeError, match="not available yet"):
        config.embed.api_key()
    observed = []

    def handle(request):
        observed.append(request.headers["Authorization"])
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "routes/mnemory_embedding",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    with OpenAI(
        api_key=config.embed.api_key,
        base_url=config.embed.base_url,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    ) as client:
        for token in ("service-token-before-rotation", "service-token-after-rotation"):
            token_file.write_text(token)
            client.embeddings.create(model=config.embed.model, input=["memory"])
    assert observed == [
        "Bearer service-token-before-rotation",
        "Bearer service-token-after-rotation",
    ]


def test_partial_managed_configuration_is_rejected(monkeypatch, tmp_path):
    monkeypatch.delenv("COGNIS_INFERENCE_URL", raising=False)
    monkeypatch.setenv("COGNIS_INFERENCE_TOKEN_FILE", str(tmp_path / "token.jwt"))
    with pytest.raises(ValueError, match="Managed inference requires"):
        load_config()


@pytest.mark.parametrize("explicit", [False, True])
def test_all_managed_roles_omit_implicit_effort_on_sdk_wire(
    monkeypatch, tmp_path, explicit
):
    from mnemory import llm as module

    monkeypatch.setenv("COGNIS_INFERENCE_URL", "http://cognis:8080/v1")
    token = tmp_path / "token.jwt"
    token.write_text("local-test-credential")
    monkeypatch.setenv("COGNIS_INFERENCE_TOKEN_FILE", str(token))
    for key in (
        "LLM_REASONING_EFFORT",
        "FIND_REASONING_EFFORT",
        "FSCK_REASONING_EFFORT",
        "CONSOLIDATION_REASONING_EFFORT",
    ):
        if explicit:
            monkeypatch.setenv(key, "high")
        else:
            monkeypatch.delenv(key, raising=False)
    observed = []

    def handle(request):
        observed.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-effort",
                "object": "chat.completion",
                "created": 0,
                "model": observed[-1]["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "{}"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    monkeypatch.setattr(
        module,
        "OpenAI",
        lambda **kwargs: OpenAI(
            **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle))
        ),
    )
    cfg = load_config()
    roles = [
        (cfg.llm, None),
        (
            replace(
                cfg.llm,
                model=cfg.memory.find_model,
                reasoning_effort=cfg.memory.find_reasoning_effort,
            ),
            None,
        ),
        (
            replace(
                cfg.llm,
                model=cfg.memory.consolidation_model,
                reasoning_effort=cfg.memory.consolidation_reasoning_effort,
            ),
            None,
        ),
        (
            replace(cfg.llm, model=cfg.memory.fsck_model),
            cfg.memory.fsck_reasoning_effort,
        ),
    ]
    for config, call_effort in roles:
        client = LLMClient(config)
        try:
            client.generate(
                [{"role": "user", "content": "Check"}], reasoning_effort=call_effort
            )
        finally:
            client._client.close()
    assert len(observed) == 4
    for request in observed:
        if explicit:
            assert request["reasoning_effort"] == "high"
        else:
            assert "reasoning_effort" not in request
