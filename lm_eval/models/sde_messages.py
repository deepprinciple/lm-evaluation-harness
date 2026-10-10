"""Recorded no-tools Messages SSE using the shared SDE request lifecycle."""

import copy
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from lm_eval.api.instance import GenerationResult, Instance
from lm_eval.api.registry import register_model
from lm_eval.models.sde_chat import SDEChatCompletion
from lm_eval.models.sde_stream import capture_stream, sse_events, stream_sha256


def parse_stream(body: str, expected_model: str) -> dict[str, Any]:
    """Keep complete/partial segments; only supersede unfinished thinking."""
    message: dict[str, Any] = {}
    blocks: dict[int, dict[str, Any]] = {}
    closed: set[int] = set()
    counts: Counter[str] = Counter()
    segments: list[dict[str, Any]] = []
    stopped = False
    failure = None

    def snapshot() -> dict[str, Any]:
        response = copy.deepcopy(message)
        response["content"] = [copy.deepcopy(blocks[i]) for i in sorted(blocks)]
        return {
            "response": response,
            "message_stop": stopped,
            "blocks_closed": set(blocks) == closed,
        }

    try:
        for event in sse_events(body):
            kind = event["type"]
            counts[kind] += 1
            if kind == "ping":
                continue
            if kind == "error":
                if not isinstance(event.get("error"), dict) or not isinstance(
                    event["error"].get("type"), str
                ):
                    raise ValueError("invalid stream error")
                failure = event
                break
            if kind == "message_start":
                incoming = event["message"]
                if (
                    not isinstance(incoming, dict)
                    or incoming.get("role") != "assistant"
                    or not isinstance(incoming.get("id"), str)
                    or not incoming["id"]
                    or not isinstance(incoming.get("model"), str)
                    or incoming.get("content") != []
                    or not isinstance(incoming.get("usage", {}), dict)
                ):
                    raise ValueError("invalid message_start")
                if message:
                    ids = {segment["response"]["id"] for segment in segments}
                    ids.add(message["id"])
                    if (
                        stopped
                        or message.get("stop_reason") is not None
                        or message.get("model") != expected_model
                        or incoming["model"] != expected_model
                        or incoming["id"] in ids
                        or any(
                            block["type"] not in {"thinking", "redacted_thinking"}
                            for block in blocks.values()
                        )
                    ):
                        raise ValueError("ambiguous relay restart")
                    segments.append(snapshot())
                    blocks, closed, stopped = {}, set(), False
                message = copy.deepcopy(incoming)
                continue
            if not message or stopped:
                raise ValueError("event outside active message")
            if kind == "content_block_start":
                index = int(event["index"])
                if index < 0 or index in blocks:
                    raise ValueError("invalid or duplicate content block")
                content = event["content_block"]
                if not isinstance(content, dict) or not isinstance(
                    content.get("type"), str
                ):
                    raise ValueError("invalid content block")
                for field in ("text", "thinking", "signature"):
                    if field in content and not isinstance(content[field], str):
                        raise ValueError("non-text content")
                blocks[index] = copy.deepcopy(content)
            elif kind == "content_block_delta":
                index = int(event["index"])
                if index in closed:
                    raise ValueError("delta after block_stop")
                content, delta = blocks[index], event["delta"]
                fields = {
                    "text_delta": ("text", "text"),
                    "thinking_delta": ("thinking", "thinking"),
                    "signature_delta": ("thinking", "signature"),
                    "input_json_delta": ("tool_use", "partial_json"),
                }
                content_type, field = fields[delta["type"]]
                if content["type"] != content_type or not isinstance(delta[field], str):
                    raise ValueError("incompatible content delta")
                content[field] = content.get(field, "") + delta[field]
            elif kind == "content_block_stop":
                index = int(event["index"])
                if index not in blocks or index in closed:
                    raise ValueError("invalid block_stop")
                closed.add(index)
            elif kind == "message_delta":
                delta, usage = event.get("delta", {}), event.get("usage", {})
                if (
                    not isinstance(delta, dict)
                    or set(delta) - {"stop_reason", "stop_sequence"}
                    or not isinstance(usage, dict)
                ):
                    raise ValueError("invalid message_delta")
                message.update(delta)
                message.setdefault("usage", {}).update(usage)
            elif kind == "message_stop":
                stopped = True
            else:
                raise ValueError("unknown message event")
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        failure = {"type": "parser_error", "message": str(exc)}
    current = snapshot()
    return {
        **current,
        "relay_segments": [*segments, current],
        "event_counts": dict(counts),
        "stream_error": failure,
    }


def classify(parsed: dict[str, Any], expected_model: str) -> tuple[str, str]:
    response = parsed["response"]
    blocks = response["content"]
    answer = "".join(
        block.get("text", "") for block in blocks if block["type"] == "text"
    )
    failure = parsed["stream_error"]
    if failure:
        error_type = (failure.get("error") or {}).get("type")
        return (
            "transport_failed"
            if error_type in {"overloaded_error", "api_error", "rate_limit_error"}
            else "protocol_error",
            answer,
        )
    if not parsed["message_stop"] or not parsed["blocks_closed"]:
        return "transport_failed", answer
    if response.get("model") != expected_model:
        return "model_mismatch", answer
    if any(
        block["type"] not in {"text", "thinking", "redacted_thinking"}
        for block in blocks
    ):
        return "protocol_error", answer
    reason = response.get("stop_reason")
    if reason == "max_tokens":
        return "truncated", answer
    if reason == "refusal":
        return "refusal", answer
    if reason != "end_turn":
        return "generation_incomplete", answer
    return ("ok" if answer.strip() else "empty"), answer


@register_model("sde-messages")
class SDEMessages(SDEChatCompletion):
    """Opus Messages profile; reuse locking, deadlines, repeats and retries."""

    def __init__(
        self,
        model: str,
        max_tokens: int,
        record_dir: str,
        base_url: str = "https://api.anthropic.com/v1/messages",
        api_key_env: str = "ANTHROPIC_API_KEY",
        effort: str = "max",
        anthropic_version: str = "2023-06-01",
        num_concurrent: int = 4,
        timeout: float = 3600,
        max_attempts: int = 3,
        batch_size: int = 1,
        max_batch_size: Any = None,
        device: Any = None,
    ) -> None:
        if effort not in {"low", "medium", "high", "max"}:
            raise ValueError("Unsupported Messages effort")
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            record_dir=record_dir,
            base_url=base_url,
            api_key_env=api_key_env,
            num_concurrent=num_concurrent,
            timeout=timeout,
            max_attempts=max_attempts,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            device=device,
        )
        self.config.pop("reasoning_effort")
        self.config.pop("thinking")
        self.config["runtime_sha256"] = self.config.pop("adapter_sha256")
        self.config.update(
            api_format="messages",
            effort=effort,
            anthropic_version=anthropic_version,
            stream_sha256=stream_sha256(),
            adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )

    def create_payload(self, request: Instance) -> dict[str, Any]:
        chat, max_tokens = self.chat_input(request)
        messages = list(chat)
        system = []
        while messages and messages[0]["role"] == "system":
            system.append({"type": "text", "text": messages.pop(0)["content"]})
        if any(message["role"] == "system" for message in messages):
            raise ValueError("Messages system prompts must be leading messages")
        if (
            not messages
            or messages[0]["role"] != "user"
            or messages[-1]["role"] != "user"
        ):
            raise ValueError("No-tools Messages history must start and end with user")
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "output_config": {"effort": self.config["effort"]},
            "stream": True,
        }
        if system:
            payload["system"] = system
        return payload

    def result(self, record: dict[str, Any], path: Path) -> GenerationResult:
        output = super().result(record, path)
        output.metadata["stream_metadata"] = [
            {
                "body_path": attempt.get("body_path"),
                "message_stop": attempt.get("message_stop", False),
                "blocks_closed": attempt.get("blocks_closed", False),
                "event_counts": attempt.get("event_counts", {}),
                "stream_error": attempt.get("stream_error"),
                "segments": [
                    {
                        "message_id": segment["response"].get("id"),
                        "model": segment["response"].get("model"),
                        "stop_reason": segment["response"].get("stop_reason"),
                        "usage": segment["response"].get("usage", {}),
                        "message_stop": segment["message_stop"],
                    }
                    for segment in attempt.get("relay_segments", [])
                ],
            }
            for attempt in record["attempts"]
        ]
        return output

    async def request_once(
        self, session: Any, payload: dict[str, Any], key: str, attempt: dict[str, Any]
    ) -> None:
        try:
            await capture_stream(
                session,
                self.config["base_url"],
                payload,
                {
                    "x-api-key": key,
                    "anthropic-version": self.config["anthropic_version"],
                },
                key,
                attempt,
            )
        finally:
            if attempt.get("http_status") == 200:
                parsed = parse_stream(attempt["body"], self.model)
                attempt.update(parsed)
                status, answer = classify(parsed, self.model)
                attempt["answer"] = answer
                # Transport failure or cancellation takes precedence over any
                # completed-looking message in the captured partial HTTP body.
                if attempt["status"] == "running":
                    attempt["status"] = status
