"""Recorded Responses SSE with explicit native Agent and strict no-tools modes."""

import copy
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from lm_eval.api.instance import GenerationResult, Instance
from lm_eval.api.registry import register_model
from lm_eval.models.sde_chat import SDEChatCompletion
from lm_eval.models.sde_stream import capture_stream, sse_events, stream_sha256


# Protocol item types, not a tool-name allowlist. Function/custom namespaces and
# names are opaque and must be round-tripped unchanged.
NATIVE_ITEM_FIELDS: dict[str, dict[str, type]] = {
    "function_call": {"name": str, "call_id": str, "arguments": str},
    "custom_tool_call": {"name": str, "call_id": str, "input": str},
    "web_search_call": {"status": str, "id": str, "action": dict},
    "file_search_call": {"status": str, "id": str, "queries": list},
    "code_interpreter_call": {"status": str, "id": str, "container_id": str},
    "computer_call": {
        "status": str,
        "id": str,
        "call_id": str,
        "pending_safety_checks": list,
    },
    "local_shell_call": {"status": str, "id": str, "call_id": str, "action": dict},
    "shell_call": {"status": str, "id": str, "call_id": str, "action": dict},
    "apply_patch_call": {"status": str, "id": str, "call_id": str, "operation": dict},
    "image_generation_call": {"status": str, "id": str},
    "mcp_call": {"id": str, "name": str, "arguments": str, "server_label": str},
    "mcp_list_tools": {"id": str, "server_label": str, "tools": list},
    "mcp_approval_request": {
        "id": str,
        "name": str,
        "arguments": str,
        "server_label": str,
    },
    "mcp_approval_response": {"approval_request_id": str, "approve": bool},
    "tool_search_call": {"status": str, "id": str, "arguments": dict, "execution": str},
    "tool_search_output": {"status": str, "id": str, "tools": list, "execution": str},
    "function_call_output": {"call_id": str},
    "custom_tool_call_output": {"call_id": str},
    "computer_call_output": {"call_id": str, "output": dict},
    "local_shell_call_output": {"id": str, "output": str},
    "shell_call_output": {"call_id": str, "output": list},
    "apply_patch_call_output": {"call_id": str, "status": str},
    "program": {"id": str, "call_id": str, "code": str, "fingerprint": str},
    "program_output": {"id": str, "call_id": str, "result": str, "status": str},
    "additional_tools": {"id": str, "role": str, "tools": list},
    "compaction": {"id": str, "encrypted_content": str},
}


def validate_native_item(item: dict[str, Any], *, partial: bool = False) -> None:
    kind = item.get("type")
    if kind not in {"message", "reasoning", *NATIVE_ITEM_FIELDS}:
        raise ValueError("unknown Responses output item type")
    for field, field_type in NATIVE_ITEM_FIELDS.get(kind, {}).items():
        if (not partial or field in item) and not isinstance(
            item.get(field), field_type
        ):
            raise ValueError(f"invalid {kind} {field}")
    for field in ("id", "name", "call_id", "server_label"):
        if field in item and (not isinstance(item[field], str) or not item[field]):
            raise ValueError(f"invalid output item {field}")
    if item.get("namespace") is not None and not isinstance(item["namespace"], str):
        raise ValueError("invalid output item namespace")
    if "status" in item and item["status"] not in {
        "in_progress",
        "completed",
        "incomplete",
        "failed",
        "searching",
        "interpreting",
        "generating",
        "calling",
    }:
        raise ValueError("invalid output item status")
    if "execution" in item and item["execution"] not in {"server", "client"}:
        raise ValueError("invalid tool execution")
    for field in ("queries", "tools", "outputs", "pending_safety_checks"):
        if item.get(field) is not None:
            if not isinstance(item[field], list) or any(
                not isinstance(value, str if field == "queries" else dict)
                for value in item[field]
            ):
                raise ValueError(f"invalid tool {field}")
    if kind in {"function_call_output", "custom_tool_call_output"} and not isinstance(
        item.get("output"), (str, list)
    ):
        raise ValueError("invalid tool call output")
    if (
        kind == "code_interpreter_call"
        and item.get("code") is not None
        and not isinstance(item["code"], str)
    ):
        raise ValueError("invalid interpreter code")
    for field in ("action", "operation"):
        if field in item and not isinstance(item[field], dict):
            raise ValueError(f"invalid tool {field}")
    if (
        kind == "computer_call"
        and not partial
        and not (
            isinstance(item.get("action"), dict)
            or isinstance(item.get("actions"), list)
        )
    ):
        raise ValueError("invalid computer action")
    if kind == "web_search_call" and (not partial or "action" in item):
        action = item["action"]
        if action.get("type") not in {"search", "open_page", "find_in_page"}:
            raise ValueError("invalid search action")
        for field in ("query", "url", "pattern"):
            if action.get(field) is not None and not isinstance(action[field], str):
                raise ValueError(f"invalid search {field}")
        if action["type"] == "find_in_page" and any(
            not isinstance(action.get(field), str) for field in ("url", "pattern")
        ):
            raise ValueError("invalid find_in_page action")
        if "queries" in action and (
            not isinstance(action["queries"], list)
            or any(not isinstance(query, str) for query in action["queries"])
        ):
            raise ValueError("invalid search queries")
    if kind in {"shell_call", "local_shell_call"} and (not partial or "action" in item):
        commands = item["action"].get("commands" if kind == "shell_call" else "command")
        if not isinstance(commands, list) or any(
            not isinstance(command, str) for command in commands
        ):
            raise ValueError("invalid shell commands")


# Native tool progress events and their wire item types.
TOOL_PROGRESS = {
    "web_search_call": {"in_progress", "searching", "completed"},
    "file_search_call": {"in_progress", "searching", "completed"},
    "code_interpreter_call": {"in_progress", "interpreting", "completed"},
    "image_generation_call": {
        "in_progress",
        "generating",
        "completed",
        "partial_image",
    },
    "mcp_call": {"in_progress", "completed", "failed"},
    "mcp_list_tools": {"in_progress", "completed", "failed"},
    "compaction": {"compacting"},
}
TOOL_EVENT_FIELDS = {
    "function_call_arguments": ("function_call", "arguments"),
    "custom_tool_call_input": ("custom_tool_call", "input"),
    "mcp_call_arguments": ("mcp_call", "arguments"),
    "code_interpreter_call_code": ("code_interpreter_call", "code"),
    "shell_call_command": ("shell_call", "command"),
    "shell_call_output_content": ("shell_call_output", "output"),
}


def validate_tool_event(event: dict[str, Any]) -> str:
    _, family, phase = event["type"].split(".")
    kind = family
    if family in TOOL_EVENT_FIELDS:
        kind, field = TOOL_EVENT_FIELDS[family]
        phases = {"delta", "done"} | (
            {"added"} if family == "shell_call_command" else set()
        )
        if phase not in phases:
            raise ValueError("unknown native tool event")
        field = "delta" if phase == "delta" else field
        field_type = (
            (dict if phase == "delta" else list)
            if family == "shell_call_output_content"
            else str
        )
        if not isinstance(event.get(field), field_type):
            raise ValueError(f"invalid tool event {field}")
    elif phase not in TOOL_PROGRESS.get(family, set()):
        raise ValueError("unknown native tool event")
    if type(event.get("output_index")) is not int or event["output_index"] < 0:
        raise ValueError("invalid tool event output_index")
    if family != "shell_call_command" and (
        not isinstance(event.get("item_id"), str) or not event["item_id"]
    ):
        raise ValueError("invalid tool event item_id")
    if family in {"shell_call_command", "shell_call_output_content"} and (
        type(event.get("command_index")) is not int or event["command_index"] < 0
    ):
        raise ValueError("invalid command_index")
    if phase == "partial_image" and (
        type(event.get("partial_image_index")) is not int
        or event["partial_image_index"] < 0
        or not isinstance(event.get("partial_image_b64"), str)
    ):
        raise ValueError("invalid partial image")
    return kind


def validate_response(
    value: Any, allowed_tools: set[str] | None = None, *, agent_mode: bool = False
) -> dict[str, Any]:
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
    if value.get("max_output_tokens") is not None and (
        type(value["max_output_tokens"]) is not int or value["max_output_tokens"] <= 0
    ):
        raise ValueError("invalid response output budget")
    item_ids: set[str] = set()
    for item in value["output"]:
        if not isinstance(item, dict) or item.get("type") not in {
            "message",
            "reasoning",
            *(
                NATIVE_ITEM_FIELDS
                if agent_mode
                else ({"function_call"} if allowed_tools is not None else set())
            ),
        }:
            raise ValueError("unexpected no-tools output item")
        if agent_mode:
            validate_native_item(item)
            if item.get("id") is not None:
                if item["id"] in item_ids:
                    raise ValueError("duplicate output item identity")
                item_ids.add(item["id"])
        if item["type"] == "function_call" and not agent_mode:
            name = item.get("name", "")
            if item.get("namespace"):
                name = item["namespace"] + "__" + name
            if (
                allowed_tools is None
                or name not in allowed_tools
                or not isinstance(item.get("arguments"), str)
                or not isinstance(item.get("call_id"), str)
            ):
                raise ValueError("unexpected function call")
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


def parse_stream(
    body: str, allowed_tools: set[str] | None = None, *, agent_mode: bool = False
) -> dict[str, Any]:
    response: dict[str, Any] = {}
    items: dict[int, dict[str, Any]] = {}
    closed: set[int] = set()
    text_deltas: list[dict[str, Any]] = []
    reasoning_deltas: list[dict[str, Any]] = []
    refusal_deltas: list[dict[str, Any]] = []
    tool_events: list[dict[str, Any]] = []
    tool_references: list[tuple[int, str | None, str]] = []
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
            if kind in {"ping", "keepalive"}:
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
                incoming = validate_response(
                    event["response"], allowed_tools, agent_mode=agent_mode
                )
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
                if agent_mode and kind in {
                    "response.completed",
                    "response.incomplete",
                    "response.failed",
                }:
                    output = incoming["output"]
                    for index, item in items.items():
                        if index >= len(output) or any(
                            item.get(field) != output[index].get(field)
                            for field in ("id", "type", "name", "namespace", "call_id")
                        ):
                            raise ValueError("terminal output item identity mismatch")
                        if index in closed and any(
                            output[index].get(field) != value
                            for field, value in item.items()
                        ):
                            raise ValueError(
                                "terminal output differs from completed item"
                            )
                    for index, item_id, item_type in tool_references:
                        if (
                            index >= len(output)
                            or output[index].get("type") != item_type
                            or (
                                item_id is not None
                                and output[index].get("id") != item_id
                            )
                        ):
                            raise ValueError("tool event/terminal item mismatch")
                    if kind == "response.completed" and any(
                        item.get("status", "completed") != "completed"
                        for item in output
                    ):
                        raise ValueError("unfinished output in completed response")
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
                    or item.get("type")
                    not in (
                        {"message", "reasoning"}
                        | (
                            set(NATIVE_ITEM_FIELDS)
                            if agent_mode
                            else (
                                {"function_call"}
                                if allowed_tools is not None
                                else set()
                            )
                        )
                    )
                ):
                    raise ValueError("unexpected or invalid no-tools item")
                if agent_mode:
                    validate_native_item(item, partial=kind.endswith("added"))
                    if index in items and any(
                        items[index].get(field) != item.get(field)
                        for field in ("id", "type", "name", "namespace", "call_id")
                    ):
                        raise ValueError("changed output item identity")
                if index in closed or (kind.endswith("added") and index in items):
                    raise ValueError("duplicate output item")
                if index in items and items[index].get("id") != item.get("id"):
                    raise ValueError("changed output item identity")
                items[index] = copy.deepcopy(item)
                if kind.endswith("done"):
                    closed.add(index)
            elif (
                agent_mode
                and kind.startswith("response.")
                and (
                    kind.split(".")[1] in TOOL_EVENT_FIELDS
                    or kind.split(".")[1] in TOOL_PROGRESS
                )
            ):
                item_type = validate_tool_event(event)
                index = event["output_index"]
                item_id = event.get("item_id")
                if index in closed or (
                    index in items
                    and (
                        items[index].get("type") != item_type
                        or (item_id is not None and items[index].get("id") != item_id)
                    )
                ):
                    raise ValueError("tool event/output item mismatch")
                tool_references.append((index, item_id, item_type))
                tool_events.append(copy.deepcopy(event))
            elif (
                kind
                in {
                    "response.function_call_arguments.delta",
                    "response.function_call_arguments.done",
                }
                and allowed_tools is not None
            ):
                field = "delta" if kind.endswith("delta") else "arguments"
                if not isinstance(event.get(field), str):
                    raise ValueError("invalid function arguments")
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
        "tool_events": tool_events,
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
