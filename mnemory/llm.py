"""LLM client wrapper for mnemory.

Provides a thin, cached OpenAI-compatible chat completions client with
structured output support (json_schema) and automatic fallback to JSON mode.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from openai import BadRequestError, OpenAI

from mnemory.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Cached OpenAI-compatible LLM client.

    Reuses a single HTTP connection pool across calls. Supports structured
    outputs (json_schema) with automatic fallback to plain JSON mode for
    providers that don't support it.
    """

    def __init__(self, config: LLMConfig):
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self._model = config.model
        self._temperature = config.temperature
        self._reasoning_effort = config.reasoning_effort
        self._supports_structured: bool | None = None
        # Tracks unsupported parameters for this model/provider.
        # Maps param name -> fix action. Populated on first BadRequestError.
        self._param_fixes: dict[str, str] = {}

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int = 16384,
        reasoning_effort: str | None = None,
        operation: str = "unknown",
    ) -> str:
        """Generate a chat completion, returning the content string.

        Args:
            messages: Chat messages (system + user).
            json_schema: If provided, attempts structured output first
                         (json_schema mode), falling back to json_object mode.
            temperature: Override default temperature.
            max_tokens: Maximum tokens to generate.
            reasoning_effort: Override instance-level reasoning effort for this
                              call only. Pass a value like "medium" or "high"
                              to override, or None to use the instance default.
            operation: Operation name for logging and metrics (e.g., "extract",
                       "query_gen", "rerank"). Defaults to "unknown".

        Returns:
            The raw content string from the LLM response.
        """
        temp = temperature if temperature is not None else self._temperature

        t0 = time.monotonic()

        if json_schema and self._supports_structured is not False:
            try:
                result = self._call(
                    messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": json_schema,
                    },
                    temperature=temp,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                )
                self._supports_structured = True
                self._log_timing(operation, t0)
                return result
            except Exception:
                if self._supports_structured is None:
                    logger.debug(
                        "Structured outputs not supported by provider, "
                        "falling back to JSON mode"
                    )
                    self._supports_structured = False
                else:
                    raise

        # JSON mode fallback (or no schema requested)
        response_format = {"type": "json_object"} if json_schema else None
        result = self._call(
            messages,
            response_format=response_format,
            temperature=temp,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        self._log_timing(operation, t0)
        return result

    def _log_timing(self, operation: str, t0: float) -> None:
        """Log and record metrics for an LLM call."""
        duration = time.monotonic() - t0
        duration_ms = int(duration * 1000)
        logger.info(
            "LLM call: operation=%s model=%s duration_ms=%d",
            operation,
            self._model,
            duration_ms,
        )
        from mnemory.metrics import get_collector

        collector = get_collector()
        if collector:
            collector.observe_llm_duration(operation, self._model, duration)

    def _call(
        self,
        messages: list[dict[str, str]],
        response_format: dict | None,
        temperature: float,
        max_tokens: int,
        reasoning_effort: str | None = None,
    ) -> str:
        """Execute a single chat completion call.

        Handles parameter incompatibilities across model versions and
        providers. When a model rejects a parameter (e.g., max_tokens,
        temperature), the fix is cached so subsequent calls skip the
        unsupported parameter without any retry overhead.

        Also retries on empty or truncated responses (transient LLM
        issues like content filters, refusals, or API glitches).
        """
        params = self._build_params(
            messages,
            response_format,
            temperature,
            max_tokens,
            reasoning_effort=reasoning_effort,
        )

        # Retry loop: some models reject multiple parameters (e.g., gpt-5-mini
        # rejects both max_tokens and temperature), each discovered one at a
        # time. After fixes are cached, subsequent calls have zero overhead.
        max_retries = 3
        response = None
        for attempt in range(1 + max_retries):
            try:
                response = self._client.chat.completions.create(**params)
                break
            except BadRequestError as e:
                if attempt < max_retries and self._try_fix_params(
                    e, params, max_tokens
                ):
                    continue
                raise

        assert response is not None  # guaranteed by break or raise above

        # Retry on empty content (transient LLM issues: content filters,
        # refusals, API glitches). Non-empty but truncated responses are
        # logged but not retried — the caller can decide how to handle.
        content_retries = 2
        for retry in range(content_retries):
            choice = response.choices[0]
            content = choice.message.content or ""

            if content.strip():
                if choice.finish_reason == "length":
                    logger.warning(
                        "LLM response truncated (finish_reason=length, content_len=%d)",
                        len(content),
                    )
                break

            refusal = getattr(choice.message, "refusal", None)
            logger.warning(
                "LLM returned empty content (finish_reason=%s, refusal=%s), "
                "retrying (%d/%d)",
                choice.finish_reason,
                refusal,
                retry + 1,
                content_retries,
            )
            response = self._client.chat.completions.create(**params)
        else:
            # All retries exhausted — use whatever we got
            choice = response.choices[0]
            content = choice.message.content or ""
            if not content.strip():
                refusal = getattr(choice.message, "refusal", None)
                logger.warning(
                    "LLM returned empty content after %d retries "
                    "(finish_reason=%s, refusal=%s)",
                    content_retries,
                    choice.finish_reason,
                    refusal,
                )

        return _clean_response(content)

    def _build_params(
        self,
        messages: list[dict[str, str]],
        response_format: dict | None,
        temperature: float,
        max_tokens: int,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        """Build API call parameters, applying any cached fixes."""
        params: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }

        # temperature: omit if model doesn't support custom values
        if "temperature" not in self._param_fixes:
            params["temperature"] = temperature

        # max_tokens: swap to max_completion_tokens if needed
        if "max_tokens" in self._param_fixes:
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens

        if response_format:
            params["response_format"] = response_format

        # reasoning_effort: per-call override takes priority over instance default.
        # Omit entirely if not set or if model rejected it previously.
        effective_effort = reasoning_effort or self._reasoning_effort
        if effective_effort and "reasoning_effort" not in self._param_fixes:
            params["reasoning_effort"] = effective_effort

        return params

    def _try_fix_params(
        self,
        error: BadRequestError,
        params: dict[str, Any],
        max_tokens: int,
    ) -> bool:
        """Try to fix params based on a BadRequestError.

        Parses the error to identify the unsupported parameter, applies
        the appropriate fix, caches it for future calls, and mutates
        params in-place for the retry.

        Returns True if a fix was applied and the call should be retried.
        """
        error_body = getattr(error, "body", None)
        if not isinstance(error_body, dict):
            return False

        param = error_body.get("param", "")
        code = error_body.get("code", "")

        if not param or code not in (
            "unsupported_parameter",
            "unsupported_value",
        ):
            return False

        # Already tried to fix this param — don't loop
        if param in self._param_fixes:
            return False

        if param == "max_tokens":
            logger.info(
                "Model %s requires max_completion_tokens instead of "
                "max_tokens — adapting",
                self._model,
            )
            self._param_fixes["max_tokens"] = "use_max_completion_tokens"
            params.pop("max_tokens", None)
            params["max_completion_tokens"] = max_tokens
            return True

        if param == "temperature":
            logger.info(
                "Model %s does not support custom temperature — omitting",
                self._model,
            )
            self._param_fixes["temperature"] = "omit"
            params.pop("temperature", None)
            return True

        # Generic: try omitting the unsupported parameter
        if param in params:
            logger.info(
                "Model %s does not support parameter '%s' — omitting",
                self._model,
                param,
            )
            self._param_fixes[param] = "omit"
            params.pop(param, None)
            return True

        return False


def _clean_response(text: str) -> str:
    """Strip markdown code fences and <think> blocks from LLM output.

    Some models (DeepSeek, etc.) wrap JSON in ```json...``` blocks or
    include <think>...</think> reasoning blocks. Strip these to get
    clean JSON.
    """
    # Remove <think>...</think> blocks (DeepSeek reasoning)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Remove markdown code fences
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (with optional language tag)
        text = re.sub(r"^```\w*\n?", "", text)
        # Remove closing fence
        text = re.sub(r"\n?```$", "", text)

    return text.strip()


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse a JSON response from the LLM, with fallback extraction.

    Tries direct JSON parse first. If that fails, attempts to extract
    JSON from within the text (e.g., surrounded by other text).

    Raises:
        ValueError: If no valid JSON can be extracted.
    """
    text = _clean_response(text)

    # Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Fallback: find JSON object in text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}")


def salvage_json_objects(text: str, key: str = "memories") -> list[dict[str, Any]] | None:
    """Recover complete objects from a truncated JSON array in an LLM reply.

    A response cut off by the token cap keeps every fully closed object
    before the cut. Walk the array, collect complete objects, ignore the
    trailing fragment. Returns None when nothing is recoverable.
    """
    cleaned = _clean_response(text)
    anchor = cleaned.find(f'"{key}"')
    start = cleaned.find("[", anchor + 1) if anchor != -1 else -1
    if start == -1:
        return None

    objects: list[dict[str, Any]] = []
    depth = 0
    obj_start = None
    in_str = False
    escaped = False

    for i in range(start, len(cleaned)):
        char = cleaned[i]
        if in_str:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_str = False
            continue
        if char == '"':
            in_str = True
        elif char == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                try:
                    obj = json.loads(cleaned[obj_start : i + 1])
                    if isinstance(obj, dict) and obj.get("text"):
                        objects.append(obj)
                except json.JSONDecodeError:
                    pass
                obj_start = None

    return objects or None
