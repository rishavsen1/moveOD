"""Minimal LLM client: two backends, one call signature, an on-disk cache.

    from llm_client import LLM
    llm = LLM()                                     # backend from $LLM_BACKEND, default "claude" (model "haiku")
    llm.complete("Summarise ...").text
    LLM("openrouter", "<vendor>/<name>:free").complete_json('Return {"ok": true}')

claude      Headless Claude Code (`claude -p --safe-mode`), prompt on stdin. Uses the Claude Code login, so calls
            count against the subscription's 5-hour and weekly windows. The CLI has no temperature or max-token
            flags: those arguments are ignored here and left out of the cache key.
openrouter  POST /api/v1/chat/completions with $OPENROUTER_API_KEY and an explicit model id (or $OPENROUTER_MODEL).
            ":free" models allow 20 requests/min and 50/day (1000/day with >= $10 lifetime credit); calls are paced
            under the minute cap and a 429 whose reset is far away raises instead of waiting.

Every successful response is cached in SQLite (default ~/.cache/moveod_llm/cache.sqlite, override with $LLM_CACHE),
so reruns are free. Usage per call is stored with it; for the experiment ledger:
    sqlite3 ~/.cache/moveod_llm/cache.sqlite "select backend, model, count(*) from cache group by 1, 2"

CLI: `echo "prompt" | python llm_client.py` (backend and model from the environment).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_CACHE = "~/.cache/moveod_llm/cache.sqlite"
DEFAULT_SYSTEM = "Answer directly and concisely."
RETRY_STATUS = {429, 500, 502, 503, 504, 529}
MAX_WAIT_S = 120  # a longer rate-limit wait means a daily cap, which retrying cannot fix


class LLMError(RuntimeError):
    """The backend failed and retrying will not help, or retries ran out."""


class _Retry(Exception):
    def __init__(self, msg: str, wait_s: float | None = None) -> None:
        super().__init__(msg)
        self.wait_s = wait_s


@dataclass(frozen=True)
class LLMResponse:
    text: str
    backend: str
    model: str
    cached: bool = False
    usage: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0


class LLM:
    """One instance per (backend, model); safe to share across threads."""

    def __init__(
        self,
        backend: str | None = None,
        model: str | None = None,
        *,
        cache_path: str | Path | None = None,
        min_interval_s: float | None = None,
        timeout_s: float = 300.0,
        retries: int = 4,
    ) -> None:
        self.backend = backend or os.environ.get("LLM_BACKEND", "claude")
        if self.backend not in ("claude", "openrouter"):
            raise ValueError(f"unknown backend {self.backend!r}; use 'claude' or 'openrouter'")
        self.model = model or ("haiku" if self.backend == "claude" else os.environ.get("OPENROUTER_MODEL", ""))
        if not self.model:
            raise ValueError("openrouter needs a model id: pass model= or set OPENROUTER_MODEL ('<vendor>/<name>:free')")
        free = self.backend == "openrouter" and self.model.endswith(":free")
        self.min_interval_s = min_interval_s if min_interval_s is not None else (3.1 if free else 0.0)
        self.timeout_s, self.retries = timeout_s, retries
        self._last_call = 0.0
        self._lock = threading.Lock()
        path = Path(cache_path or os.environ.get("LLM_CACHE", DEFAULT_CACHE)).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS cache "
            "(key TEXT PRIMARY KEY, backend TEXT, model TEXT, text TEXT, usage TEXT, created REAL)"
        )

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        use_cache: bool = True,
    ) -> LLMResponse:
        params: list[Any] = [self.backend, self.model, system, prompt]
        if self.backend == "openrouter":
            params += [temperature, max_tokens]
        key = hashlib.sha256(json.dumps(params).encode()).hexdigest()
        if use_cache:
            with self._lock:
                row = self._db.execute("SELECT text, usage FROM cache WHERE key = ?", (key,)).fetchone()
            if row:
                return LLMResponse(row[0], self.backend, self.model, True, json.loads(row[1]))
        t0 = time.monotonic()
        text, usage = self._with_retries(lambda: self._call(prompt, system, temperature, max_tokens))
        latency = time.monotonic() - t0
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?, ?, ?)",
                (key, self.backend, self.model, text, json.dumps(usage), time.time()),
            )
            self._db.commit()
        log.info("llm %s/%s %.1fs %s", self.backend, self.model, latency, usage)
        return LLMResponse(text, self.backend, self.model, False, usage, latency)

    def complete_json(self, prompt: str, **kw: Any) -> Any:
        """complete() plus parsing, with one repair attempt. Returns the parsed JSON value."""
        # ponytail: no schema validation; the first bracket wins. Add jsonschema if outputs get structured.
        ask = f"{prompt}\n\nReply with valid JSON only: no prose, no code fences."
        try:
            return _parse_json(self.complete(ask, **kw).text)
        except ValueError as e:
            fix = f"{ask}\n\nYour previous reply was not valid JSON ({e}). Reply with the JSON only."
            return _parse_json(self.complete(fix, **kw).text)

    def _call(self, prompt: str, system: str | None, temperature: float, max_tokens: int) -> tuple[str, dict[str, Any]]:
        if self.backend == "claude":
            return self._call_claude(prompt, system)
        return self._call_openrouter(prompt, system, temperature, max_tokens)

    def _with_retries(self, call: Callable[[], tuple[str, dict[str, Any]]]) -> tuple[str, dict[str, Any]]:
        for attempt in range(self.retries + 1):
            self._pace()
            try:
                return call()
            except _Retry as e:
                if attempt == self.retries:
                    raise LLMError(f"gave up after {attempt + 1} attempts: {e}") from e
                wait = e.wait_s if e.wait_s is not None else min(2 ** (attempt + 1), 60) + random.random()
                log.warning("llm retry %d/%d in %.0fs: %s", attempt + 1, self.retries, wait, e)
                time.sleep(wait)
        raise AssertionError("unreachable")

    def _pace(self) -> None:
        # ponytail: per-process pacing; run free-tier sweeps from one process (or lower the thread count).
        with self._lock:
            wait = self._last_call + self.min_interval_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        with self._lock:
            self._last_call = time.monotonic()

    def _call_claude(self, prompt: str, system: str | None) -> tuple[str, dict[str, Any]]:
        cmd = [
            "claude", "-p", "--safe-mode", "--model", self.model, "--output-format", "json",
            "--no-session-persistence", "--tools", "", "--system-prompt", system or DEFAULT_SYSTEM,
        ]
        try:
            p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=self.timeout_s, check=False)
        except subprocess.TimeoutExpired as e:
            raise _Retry(f"claude timed out after {self.timeout_s:.0f}s") from e
        except FileNotFoundError as e:
            raise LLMError("`claude` CLI not found on PATH") from e
        try:
            out = json.loads(p.stdout)
        except json.JSONDecodeError as e:
            raise LLMError(f"claude exit {p.returncode}, non-JSON output: {(p.stderr or p.stdout)[-300:]!r}") from e
        if out.get("is_error"):
            status = out.get("api_error_status")
            msg = f"claude error {status}: {str(out.get('result'))[:300]}"
            raise (_Retry(msg) if status in RETRY_STATUS else LLMError(msg))
        u = out.get("usage") or {}
        return out["result"], {
            "input_tokens": u.get("input_tokens"),
            "output_tokens": u.get("output_tokens"),
            "cost_usd": out.get("total_cost_usd"),  # list-price equivalent, not what the subscription bills
        }

    def _call_openrouter(
        self, prompt: str, system: str | None, temperature: float, max_tokens: int
    ) -> tuple[str, dict[str, Any]]:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise LLMError("OPENROUTER_API_KEY is not set")
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        try:
            r = requests.post(OPENROUTER_URL, headers={"Authorization": f"Bearer {api_key}"}, json=body, timeout=self.timeout_s)
        except requests.RequestException as e:
            raise _Retry(f"network error: {e}") from e
        if r.status_code in RETRY_STATUS:
            raise _Retry(f"HTTP {r.status_code}", _rate_limit_wait(r))
        if r.status_code != 200:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        if not text.strip():  # free models sometimes return an empty body with an error field
            raise _Retry(f"empty completion {str(data.get('error') or '')[:200]}")
        u = data.get("usage") or {}
        return text, {"input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens"), "cost_usd": u.get("cost")}


def _rate_limit_wait(r: requests.Response) -> float | None:
    """Seconds to wait from Retry-After or X-RateLimit-Reset; a wait beyond MAX_WAIT_S raises (daily cap)."""
    h, wait = r.headers, None
    retry_after, reset = h.get("Retry-After", ""), h.get("X-RateLimit-Reset", "")
    if retry_after.replace(".", "", 1).isdigit():
        wait = float(retry_after)
    elif reset.isdigit():
        stamp = int(reset)
        wait = max(0.0, (stamp / 1000 if stamp > 10**11 else stamp) - time.time())  # epoch ms or s
    if wait is not None and wait > MAX_WAIT_S:
        raise LLMError(f"rate limit resets in {wait / 3600:.1f} h (daily cap?); not waiting")
    return wait


def _parse_json(text: str) -> Any:
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON object or array found")
    try:
        return json.JSONDecoder().raw_decode(text[min(starts):])[0]
    except json.JSONDecodeError as e:
        raise ValueError(str(e)) from e


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    prompt = " ".join(args) if args else sys.stdin.read()
    sys.stdout.write(LLM().complete(prompt).text + "\n")


if __name__ == "__main__":
    main()
