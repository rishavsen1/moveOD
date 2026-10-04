"""Offline checks for llm_client: parsing, caching, retry/daily-cap handling, JSON repair. No network, no CLI."""

import json
import subprocess
import time

import pytest

import llm_client
from llm_client import LLM, LLMError


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)


def make(tmp_path, backend="claude", model=None):
    return LLM(backend, model, cache_path=tmp_path / "cache.sqlite", min_interval_s=0)


class Resp:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body or {}, headers or {}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


def test_claude_parses_caches_and_sends_prompt_on_stdin(tmp_path, monkeypatch):
    calls = []
    stdout = json.dumps({"result": "hi", "is_error": False, "usage": {"input_tokens": 5, "output_tokens": 2}, "total_cost_usd": 0.001})

    def run(cmd, **kw):
        calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(llm_client.subprocess, "run", run)
    llm = make(tmp_path)
    first, second = llm.complete("q"), llm.complete("q")
    assert (first.text, first.cached, second.cached) == ("hi", False, True)
    assert first.usage["output_tokens"] == 2
    assert len(calls) == 1 and calls[0][1]["input"] == "q" and "--safe-mode" in calls[0][0]


def test_claude_error_result_raises(tmp_path, monkeypatch):
    stdout = json.dumps({"is_error": True, "api_error_status": 400, "result": "bad request"})
    monkeypatch.setattr(llm_client.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout=stdout, stderr=""))
    with pytest.raises(LLMError, match="bad request"):
        make(tmp_path).complete("q")


def test_openrouter_retries_429_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    replies = [Resp(429), Resp(200, {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})]
    monkeypatch.setattr(llm_client.requests, "post", lambda *a, **k: replies.pop(0))
    assert make(tmp_path, "openrouter", "x/y:free").complete("q").text == "ok"
    assert not replies


def test_daily_cap_is_not_retried(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    reset_ms = str(int((time.time() + 6 * 3600) * 1000))
    monkeypatch.setattr(llm_client.requests, "post", lambda *a, **k: Resp(429, headers={"X-RateLimit-Reset": reset_ms}))
    with pytest.raises(LLMError, match="daily cap"):
        make(tmp_path, "openrouter", "x/y:free").complete("q")


def test_openrouter_config_errors(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    with pytest.raises(ValueError, match="model id"):
        make(tmp_path, "openrouter")
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        make(tmp_path, "openrouter", "x/y:free").complete("q")


def test_complete_json_skips_fences_and_repairs_once(tmp_path, monkeypatch):
    replies = iter(["no json here", '```json\n{"a": 1}\n```'])
    llm = make(tmp_path)
    monkeypatch.setattr(llm, "_call", lambda *a, **k: (next(replies), {}))
    assert llm.complete_json("give json") == {"a": 1}
