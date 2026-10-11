"""Recorded, resumable no-tools Chat Completions for SDE reasoning runs."""

import asyncio
import hashlib
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from lm_eval.api.instance import GenerationResult, Instance
from lm_eval.api.model import LM, CacheHook
from lm_eval.api.registry import register_model
from lm_eval.models.sde_stream import (
    connect_failed,
    pre_response_retryable,
    retry_delay,
)


def digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def save_record(path: Path, record: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as out:
        temporary = Path(out.name)
        try:
            json.dump(record, out, ensure_ascii=False)
            out.flush()
            os.fsync(out.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def classify(response: dict[str, Any]) -> tuple[str, str]:
    choices = response.get("choices") or []
    if len(choices) != 1:
        return "protocol_error", ""
    choice = choices[0]
    message = choice.get("message") or {}
    answer = message.get("content") or ""
    if not isinstance(answer, str) or message.get("tool_calls"):
        return "protocol_error", ""
    reason = choice.get("finish_reason")
    if reason == "length":
        return "truncated", answer
    if reason == "content_filter" or message.get("refusal"):
        return "refusal", answer
    if reason != "stop":
        return "generation_incomplete", answer
    return ("ok" if answer.strip() else "empty"), answer


@register_model("sde-chat-completions")
class SDEChatCompletion(LM):
    """A DeepSeek profile with explicit budgets and per-question raw records.

    No-tools calls omit temperature, seed and stop; task/CLI max_tokens overrides
    the required model_args fallback. Response caching is forbidden for repeats.
    """

    sde_recorded_results = True

    def __init__(
        self,
        model: str,
        max_tokens: int,
        record_dir: str,
        base_url: str = "https://api.deepseek.com/v1/chat/completions",
        api_key_env: str = "DEEPSEEK_API_KEY",
        reasoning_effort: str = "max",
        thinking: str = "enabled",
        num_concurrent: int = 4,
        timeout: float = 3600,
        max_attempts: int = 3,
        batch_size: int = 1,
        max_batch_size: Any = None,
        device: Any = None,
    ) -> None:
        super().__init__()
        if (
            int(max_tokens) < 1
            or int(num_concurrent) < 1
            or int(max_attempts) < 1
            or float(timeout) <= 0
            or thinking not in ("enabled", "disabled")
            or int(batch_size) != 1
        ):
            raise ValueError("Invalid generation budget, thinking mode or batch_size")
        self.validate_auth(api_key_env)
        self.model = model
        self.max_tokens = int(max_tokens)
        self.record_dir = Path(record_dir).resolve()
        self.config: dict[str, Any] = {
            "model": model,
            "base_url": base_url,
            "api_key_env": api_key_env,
            "reasoning_effort": reasoning_effort,
            "thinking": thinking,
            "max_tokens": self.max_tokens,
            "num_concurrent": int(num_concurrent),
            "timeout": float(timeout),
            "max_attempts": int(max_attempts),
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }

    def validate_auth(self, api_key_env: str) -> None:
        if not os.environ.get(api_key_env, "").strip():
            raise ValueError(f"API key not found in {api_key_env}")

    def get_api_key(self) -> str:
        return os.environ[self.config["api_key_env"]].strip()

    @property
    def tokenizer_name(self) -> str:
        return "sde-json-chat-v1"

    def chat_template(self, chat_template=False) -> str:
        return "JSON role/content messages"

    def apply_chat_template(self, chat_history, add_generation_prompt=True) -> str:
        return json.dumps(chat_history, ensure_ascii=False)

    def loglikelihood(self, requests):
        raise NotImplementedError("Use generate_until tasks for Chat Completions")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("Use generate_until tasks for Chat Completions")

    def set_cache_hook(self, cache_hook: CacheHook) -> None:
        if cache_hook.dbdict is not None:
            raise ValueError("Do not use --use_cache for independent SDE draws")
        super().set_cache_hook(cache_hook)

    def chat_input(self, request: Instance) -> tuple[list[dict[str, str]], int]:
        prompt, kwargs = request.args
        messages = json.loads(prompt)
        if not isinstance(messages, list) or not messages:
            raise ValueError("Use --apply_chat_template to provide JSON messages")
        if any(
            not isinstance(message, dict)
            or message.get("role") not in ("system", "user", "assistant")
            or not isinstance(message.get("content"), str)
            for message in messages
        ):
            raise ValueError("Expected text role/content messages")
        allowed = {"max_tokens", "max_gen_toks", "until", "do_sample", "temperature"}
        unknown = set(kwargs) - allowed
        if unknown:
            raise ValueError(f"Unsupported generation kwargs: {sorted(unknown)}")
        max_tokens = int(
            kwargs.get("max_tokens", kwargs.get("max_gen_toks", self.max_tokens))
        )
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        return messages, max_tokens

    def create_payload(self, request: Instance) -> dict[str, Any]:
        messages, max_tokens = self.chat_input(request)
        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "reasoning_effort": self.config["reasoning_effort"],
            "thinking": {"type": self.config["thinking"]},
            "stream": False,
        }

    def result(self, record: dict[str, Any], path: Path) -> GenerationResult:
        last = record["attempts"][-1] if record["attempts"] else {}
        response = last.get("response") or {}
        return GenerationResult(
            record.get("answer", "") if record["status"] == "ok" else "",
            {
                "status": record["status"],
                "record_path": str(path),
                "repeat_index": record["repeat_index"],
                "attempts": len(record["attempts"]),
                "budget_attempts": sum(
                    not attempt.get("budget_exempt") for attempt in record["attempts"]
                ),
                "deadline_windows": len(record.get("deadline_windows") or [None]),
                "usage": response.get("usage") or {},
                "attempt_usage": [
                    (attempt.get("response") or {}).get("usage") or {}
                    for attempt in record["attempts"]
                ],
                "returned_model": response.get("model"),
                "elapsed_seconds": record["elapsed_seconds"],
            },
        )

    async def request_once(
        self, session: Any, payload: dict[str, Any], key: str, attempt: dict[str, Any]
    ) -> None:
        import aiohttp

        chunks = []
        try:
            async with session.post(
                self.config["base_url"],
                json=payload,
                headers={"Authorization": f"Bearer {key}"},
            ) as response:
                attempt["http_status"] = response.status
                async for chunk in response.content.iter_chunked(65536):
                    chunks.append(chunk)
                if response.status != 200:
                    attempt["retry_after"] = response.headers.get("Retry-After")
            code = attempt["http_status"]
            if code == 200:
                body = b"".join(chunks).decode("utf-8").replace(key, "<redacted>")
                attempt["response"] = json.loads(body)
                attempt["status"], attempt["answer"] = classify(attempt["response"])
            else:
                attempt["status"] = (
                    "transport_failed"
                    if code in (408, 429) or code >= 500
                    else "http_error"
                )
                if pre_response_retryable(
                    code, b"".join(chunks).decode("utf-8", errors="replace")
                ):
                    attempt["budget_exempt"] = True
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            attempt.update(status="transport_failed", error=type(exc).__name__)
            if connect_failed(exc):
                attempt["budget_exempt"] = True
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            attempt.update(status="protocol_error", error=type(exc).__name__)
        finally:
            attempt["body"] = (
                b"".join(chunks)
                .decode("utf-8", errors="replace")
                .replace(key, "<redacted>")
            )

    async def generate_one(
        self,
        session: Any,
        semaphore: asyncio.Semaphore,
        request: Instance,
        payload: dict[str, Any],
        repeat: int,
    ) -> GenerationResult:
        identity = {
            "config": self.config,
            "task": request.task_name,
            "doc_id": request.doc_id,
            "request_index": request.idx,
            "doc_sha256": digest(request.doc),
            "payload": payload,
            "repeat_index": repeat,
        }
        path = self.record_dir / f"{digest(identity)}.json"
        async with semaphore:
            window = {"started_at": time.time()}
            if path.exists():
                record = json.loads(path.read_text())
                if record["identity"] != identity:
                    raise ValueError("Stored request identity does not match")
                if record["finished"]:
                    if record["attempts"] and record["attempts"][-1].get(
                        "http_status"
                    ) in (401, 403):
                        raise PermissionError(
                            "Saved authentication failure; fix the key and use a new record_dir"
                        )
                    return self.result(record, path)
                if record["attempts"] and record["attempts"][-1]["status"] == "running":
                    record["attempts"][-1]["status"] = "transport_failed"
                    record["attempts"][-1]["error"] = "ProcessInterrupted"
                # A resumed question gets a fresh deadline window; downtime is not
                # charged to the model. The interrupted attempt above still counts.
                record.setdefault(
                    "deadline_windows", [{"started_at": record["started_at"]}]
                ).append(window)
            else:
                record = {
                    "identity": identity,
                    "repeat_index": repeat,
                    "started_at": window["started_at"],
                    "deadline_windows": [window],
                    "status": "running",
                    "answer": "",
                    "attempts": [],
                    "elapsed_seconds": 0.0,
                    "finished": False,
                }
            key = self.get_api_key()
            retryable = {
                "transport_failed",
                "protocol_error",
                "empty",
                "generation_incomplete",
            }
            deadline = window["started_at"] + self.config["timeout"]
            while True:
                attempts = record["attempts"]
                consumed = sum(not attempt.get("budget_exempt") for attempt in attempts)
                previous = attempts[-1] if attempts else None
                if previous and previous["status"] not in retryable:
                    break
                if previous and previous.get("budget_exempt"):
                    streak = 0
                    for earlier in reversed(attempts):
                        if not earlier.get("budget_exempt"):
                            break
                        streak += 1
                    delay = retry_delay(previous.get("retry_after"), streak)
                elif consumed >= self.config["max_attempts"]:
                    break
                else:
                    delay = min(2**consumed, 30) if consumed else 0
                if deadline - time.time() <= delay:
                    record["status"] = "question_timeout"
                    break
                await asyncio.sleep(delay)
                attempt = {
                    "attempt": len(attempts) + 1,
                    "record_path": str(path),
                    "status": "running",
                    "started_at": time.time(),
                }
                record["attempts"].append(attempt)
                save_record(path, record)
                try:
                    await asyncio.wait_for(
                        self.request_once(session, payload, key, attempt),
                        timeout=max(0.001, deadline - time.time()),
                    )
                except asyncio.TimeoutError:
                    attempt["status"] = "question_timeout"
                finally:
                    record["elapsed_seconds"] = time.time() - record["started_at"]
                    record["status"] = attempt["status"]
                    record["answer"] = attempt.get("answer", "")
                    save_record(path, record)
                if attempt.get("http_status") in (401, 403):
                    record["finished"] = True
                    save_record(path, record)
                    raise PermissionError(
                        "API authentication/authorization failed; fix the key before resuming"
                    )
            if record["attempts"]:
                last = record["attempts"][-1]
                if record["status"] != "question_timeout":
                    record["status"] = last["status"]
                    record["answer"] = last.get("answer", "")
            record["elapsed_seconds"] = time.time() - record["started_at"]
            record["finished"] = True
            save_record(path, record)
            return self.result(record, path)

    def generate_until(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[GenerationResult]:
        import fcntl

        # Validate all inputs before starting any network calls.
        payloads = [self.create_payload(request) for request in requests]
        self.record_dir.mkdir(parents=True, exist_ok=True)
        with (self.record_dir / "runner.lock").open("w") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return asyncio.run(self.generate_batch(requests, payloads))

    async def generate_batch(
        self, requests: list[Instance], payloads: list[dict[str, Any]]
    ) -> list[GenerationResult]:
        from aiohttp import ClientSession, ClientTimeout, TCPConnector

        semaphore = asyncio.Semaphore(self.config["num_concurrent"])
        repeats: dict[tuple, int] = defaultdict(int)
        slots = []
        for request, payload in zip(requests, payloads):
            key = (request.task_name, request.doc_id, request.idx, digest(payload))
            slots.append((request, payload, repeats[key]))
            repeats[key] += 1
        async with ClientSession(
            connector=TCPConnector(limit=self.config["num_concurrent"]),
            timeout=ClientTimeout(total=None, sock_connect=30),
        ) as session:
            return await asyncio.gather(
                *(
                    self.generate_one(session, semaphore, request, payload, repeat)
                    for request, payload, repeat in slots
                )
            )
