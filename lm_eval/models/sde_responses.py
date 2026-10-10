"""Recorded no-tools Responses SSE for the existing Astra relay profile."""

import copy
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from lm_eval.api.instance import GenerationResult, Instance
from lm_eval.api.registry import register_model
from lm_eval.models.sde_chat import SDEChatCompletion
from lm_eval.models.sde_stream import capture_stream, sse_events, stream_sha256


def validate_response(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("id"), str)
        or not value["id"]
        or not isinstance(value.get("status"), str)
        or not isinstance(value.get("output"), list)
        or (value.get("model") is not None and not isinstance(value["model"], str))
    ):
        raise ValueError("invalid response object")
    for field in ("usage", "reasoning", "error", "incomplete_details"):
        if value.get(field) is not None and not isinstance(value[field], dict):
            raise ValueError(f"invalid response {field}")
    for item in value["output"]:
        if not isinstance(item, dict) or item.get("type") not in {
            "message",
            "reasoning",
        }:
            raise ValueError("unexpected no-tools output item")
        if item["type"] == "message":
            if item.get("role") != "assistant" or not isinstance(
                item.get("content"), list
            ):
                raise ValueError("invalid assistant message")
            for part in item["content"]:
                if not isinstance(part, dict) or part.get("type") not in {
                    "output_text",
                    "refusal",
                }:
                    raise ValueError("unexpected message content")
                field = "text" if part["type"] == "output_text" else "refusal"
                if not isinstance(part.get(field), str):
                    raise ValueError("non-text message content")
    return copy.deepcopy(value)


def parse_stream(body: str) -> dict[str, Any]:
    response: dict[str, Any] = {}
    items: dict[int, dict[str, Any]] = {}
    closed: set[int] = set()
    text_deltas: list[dict[str, Any]] = []
    reasoning_deltas: list[dict[str, Any]] = []
    refusal_deltas: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    terminal = None
    failure = None
    done = False
    sequence = -1
    delta_types = {
        "response.output_text.delta": text_deltas,
        "response.refusal.delta": refusal_deltas,
        "response.reasoning_text.delta": reasoning_deltas,
        "response.reasoning_summary_text.delta": reasoning_deltas,
    }
    try:
        for event in sse_events(body, allow_done=True):
            kind = event["type"]
            counts[kind] += 1
            if done:
                raise ValueError("event after stream sentinel")
            if kind == "[DONE]":
                done = True
                continue
            if "sequence_number" in event:
                number = event["sequence_number"]
                if type(number) is not int or number <= sequence:
                    raise ValueError("invalid or replayed sequence_number")
                sequence = number
            if kind == "ping":
                continue
            if terminal is not None:
                raise ValueError("event after terminal response")
            if kind == "error":
                if event.get("error") is not None and not isinstance(
                    event["error"], dict
                ):
                    raise ValueError("invalid stream error")
                failure = event
                break
            if kind in {
                "response.created",
                "response.in_progress",
                "response.queued",
                "response.completed",
                "response.incomplete",
                "response.failed",
            }:
                incoming = validate_response(event["response"])
                if response and (
                    kind == "response.created" or response["id"] != incoming["id"]
                ):
                    raise ValueError("response restart or changed identity")
                if (
                    response.get("model") is not None
                    and incoming.get("model") is not None
                    and response["model"] != incoming["model"]
                ):
                    raise ValueError("changed response model")
                expected_status = kind.split(".")[1]
                if kind == "response.created":
                    if incoming["status"] not in {"queued", "in_progress"}:
                        raise ValueError("invalid created response status")
                elif incoming["status"] != expected_status:
                    raise ValueError("event/response status mismatch")
                response = incoming
                if kind in {
                    "response.completed",
                    "response.incomplete",
                    "response.failed",
                }:
                    terminal = kind
            elif kind in {"response.output_item.added", "response.output_item.done"}:
                index = event["output_index"]
                item = event["item"]
                if (
                    type(index) is not int
                    or index < 0
                    or not isinstance(item, dict)
                    or item.get("type") not in {"message", "reasoning"}
                ):
                    raise ValueError("unexpected or invalid no-tools item")
                if index in closed or (kind.endswith("added") and index in items):
                    raise ValueError("duplicate output item")
                if index in items and items[index].get("id") != item.get("id"):
                    raise ValueError("changed output item identity")
                items[index] = copy.deepcopy(item)
                if kind.endswith("done"):
                    closed.add(index)
            elif kind in delta_types:
                if not isinstance(event.get("delta"), str):
                    raise ValueError("non-text delta")
                delta_types[kind].append(copy.deepcopy(event))
            elif kind in {
                "response.output_text.done",
                "response.reasoning_text.done",
                "response.reasoning_summary_text.done",
                "response.refusal.done",
            }:
                field = "refusal" if kind == "response.refusal.done" else "text"
                if not isinstance(event.get(field), str):
                    raise ValueError("invalid text done event")
            elif kind in {
                "response.content_part.added",
                "response.content_part.done",
                "response.reasoning_summary_part.added",
                "response.reasoning_summary_part.done",
            }:
                if not isinstance(event.get("part"), dict):
                    raise ValueError("invalid content part")
            elif kind != "response.output_text.annotation.added":
                raise ValueError("unexpected no-tools stream event")
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        failure = {"type": "parser_error", "message": str(exc)}
    return {
        "response": response,
        "terminal_event": terminal,
        "done_sentinel": done,
        "event_counts": dict(counts),
        "stream_error": failure,
        "output_items": [items[index] for index in sorted(items)],
        "text_deltas": text_deltas,
        "reasoning_deltas": reasoning_deltas,
        "refusal_deltas": refusal_deltas,
    }


def classify(
    parsed: dict[str, Any], expected_model: str, effort: str
) -> tuple[str, str]:
    response = parsed["response"]
    answer = "".join(
        part["text"] if part["type"] == "output_text" else part["refusal"]
        for item in response.get("output", [])
        if item["type"] == "message"
        for part in item["content"]
    )
    failure = parsed["stream_error"]
    if failure:
        return (
            "transport_failed" if failure["type"] == "error" else "protocol_error"
        ), answer
    terminal = parsed["terminal_event"]
    if terminal is None:
        return "transport_failed", answer
    if terminal == "response.failed":
        return "transport_failed", answer
    if response.get("model") != expected_model:
        return "model_mismatch", answer
    if terminal == "response.incomplete":
        reason = (response.get("incomplete_details") or {}).get("reason")
        if reason == "max_output_tokens":
            return "truncated", answer
        if reason == "content_filter":
            return "refusal", answer
        return "generation_incomplete", answer
    if (
        response.get("error")
        or (response.get("reasoning") or {}).get("effort") != effort
    ):
        return "protocol_error", answer
    messages = [item for item in response["output"] if item["type"] == "message"]
    if any(item.get("status", "completed") != "completed" for item in messages):
        return "protocol_error", answer
    if any(part["type"] == "refusal" for item in messages for part in item["content"]):
        return "refusal", answer
    # Only terminal output is scored. Deltas remain evidence, never an answer
    # fallback that could concatenate fragments from an interrupted stream.
    return ("ok" if answer.strip() else "empty"), answer


@register_model("sde-responses")
class SDEResponses(SDEChatCompletion):
    """Astra relay profile with the shared SDE lifecycle and SSE recorder."""

    def __init__(
        self,
        model: str,
        max_tokens: int,
        record_dir: str,
        base_url: str = "https://api.openai.com/v1/responses",
        api_key_env: str = "OPENAI_API_KEY",
        effort: str = "max",
        num_concurrent: int = 4,
        timeout: float = 3600,
        max_attempts: int = 3,
        batch_size: int = 1,
        max_batch_size: Any = None,
        device: Any = None,
    ) -> None:
        if effort not in {"minimal", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Unsupported Responses effort")
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
            api_format="responses",
            effort=effort,
            stream_sha256=stream_sha256(),
            adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )

    def create_payload(self, request: Instance) -> dict[str, Any]:
        messages, max_tokens = self.chat_input(request)
        return {
            "model": self.model,
            "input": messages,
            "reasoning": {"effort": self.config["effort"]},
            "max_output_tokens": max_tokens,
            "stream": True,
        }

    def result(self, record: dict[str, Any], path: Path) -> GenerationResult:
        output = super().result(record, path)
        last = record["attempts"][-1] if record["attempts"] else {}
        output.metadata["returned_effort"] = (
            (last.get("response") or {}).get("reasoning") or {}
        ).get("effort")
        output.metadata["stream_metadata"] = [
            {
                "body_path": attempt.get("body_path"),
                "response_id": (attempt.get("response") or {}).get("id"),
                "terminal_event": attempt.get("terminal_event"),
                "done_sentinel": attempt.get("done_sentinel", False),
                "event_counts": attempt.get("event_counts", {}),
                "stream_error": attempt.get("stream_error"),
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
                {"Authorization": f"Bearer {key}"},
                key,
                attempt,
            )
        finally:
            if attempt.get("http_status") == 200:
                parsed = parse_stream(attempt["body"])
                attempt.update(parsed)
                status, answer = classify(parsed, self.model, self.config["effort"])
                attempt["answer"] = answer
                if attempt["status"] == "running":
                    attempt["status"] = status
