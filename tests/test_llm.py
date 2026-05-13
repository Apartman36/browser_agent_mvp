from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
from openai import APIStatusError, RateLimitError

from agent.llm import LLMClient


def _chat_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _planner_done_response() -> SimpleNamespace:
    return _chat_response(
        json.dumps(
            {
                "thought": "The task is complete.",
                "tool": "done",
                "args": {"summary": "ok", "status": "success"},
                "risk": "low",
                "needs_user_confirmation": False,
                "new_facts": {},
            }
        )
    )


def _rate_limit_error() -> RateLimitError:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(429, request=request, headers={"retry-after": "0"})
    return RateLimitError("temporarily rate-limited upstream", response=response, body=None)


def _api_status_error() -> APIStatusError:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(502, request=request, headers={"retry-after": "0"})
    return APIStatusError("provider returned error", response=response, body=None)


class FakeCompletions:
    def __init__(self, responses_by_model: dict[str, list[object]]) -> None:
        self.responses_by_model = responses_by_model
        self.models: list[str] = []

    def create(self, model: str, **kwargs: object) -> object:
        self.models.append(model)
        responses = self.responses_by_model[model]
        if not responses:
            raise AssertionError(f"unexpected extra call for {model}")
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def _llm_with_fake_client(completions: FakeCompletions) -> LLMClient:
    llm = LLMClient()
    llm.client = FakeClient(completions)
    llm.model = "primary"
    llm.verifier_model = "primary"
    llm.model_fallbacks = ["fallback"]
    llm._sleep = lambda seconds: None
    return llm


def test_plan_retries_primary_then_uses_fallback(capsys) -> None:
    completions = FakeCompletions(
        {
            "primary": [_rate_limit_error(), _rate_limit_error(), _rate_limit_error()],
            "fallback": [_planner_done_response()],
        }
    )
    llm = _llm_with_fake_client(completions)

    action = llm.plan({"goal": "test"})

    assert action["tool"] == "done"
    assert completions.models == ["primary", "primary", "primary", "fallback"]
    assert "LLM provider failed for primary, trying fallback fallback..." in capsys.readouterr().out


def test_plan_returns_safe_ask_user_when_all_models_fail() -> None:
    completions = FakeCompletions(
        {
            "primary": [_api_status_error(), _api_status_error(), _api_status_error()],
            "fallback": [_api_status_error(), _api_status_error(), _api_status_error()],
        }
    )
    llm = _llm_with_fake_client(completions)

    action = llm.plan({"goal": "test"})

    assert action == {
        "thought": "The LLM provider is temporarily unavailable, so I need user guidance instead of crashing.",
        "tool": "ask_user",
        "args": {
            "question": (
                "The LLM provider returned an error or rate limit. Type 'retry' to try again, "
                "'stop' to stop, or switch MODEL in .env and rerun."
            )
        },
        "risk": "medium",
        "needs_user_confirmation": True,
        "new_facts": {},
    }


def test_query_page_provider_failure_returns_error_result() -> None:
    completions = FakeCompletions(
        {
            "primary": [_api_status_error(), _api_status_error(), _api_status_error()],
            "fallback": [_api_status_error(), _api_status_error(), _api_status_error()],
        }
    )
    llm = _llm_with_fake_client(completions)

    result = llm.query_page({"url": "https://example.test", "snapshot_yaml": ""}, "What is visible?")

    assert result["ok"] is False
    assert result["message"] == "LLM provider temporarily unavailable"
    assert "LLM provider error or rate limit after retries" in result["data"]["answer"]
