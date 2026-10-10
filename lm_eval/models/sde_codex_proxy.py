"""Recorded Responses forwarding and explicit Codex tool/budget normalization."""

import copy
import json
import os
import uuid
from typing import Any

import aiohttp
from aiohttp import web

from lm_eval.models.sde_cc_proxy import ALLOWED_TOOLS, MessagesProxy, redact
from lm_eval.models.sde_chat import save_record
from lm_eval.models.sde_responses import parse_stream
from lm_eval.models.sde_tools import TOOLS

NAMESPACE = "mcp__sde_tools"
FUNCTIONS = [
    {
        "type": "function",
        "name": NAMESPACE + "__" + tool["name"],
        "description": tool["description"],
        "parameters": tool["inputSchema"],
    }
    for tool in TOOLS
]


class ToolProfileError(ValueError):
    pass


def guard_functions(value: Any) -> None:
    if isinstance(value, dict):
        kind = value.get("type", "")
        if kind == "function_call" and function_name(value) not in ALLOWED_TOOLS:
            raise ToolProfileError("Unapproved function call")
        if kind in {
            "custom_tool_call",
            "web_search_call",
            "computer_call",
            "code_interpreter_call",
            "tool_search_call",
            "mcp_call",
        }:
            raise ToolProfileError("Unapproved native tool")
        for child in value.values():
            guard_functions(child)
    elif isinstance(value, list):
        for child in value:
            guard_functions(child)


def function_name(item: dict[str, Any]) -> str:
    name = item.get("name", "")
    if not isinstance(name, str) or not isinstance(item.get("namespace", ""), str):
        return ""
    return item["namespace"] + "__" + name if item.get("namespace") else name


def map_functions(value: Any, to_cli: bool) -> None:
    """Bridge flat provider names to Codex namespaces; never rewrite arguments."""
    if isinstance(value, dict):
        if value.get("type") == "function_call":
            name = function_name(value)
            if name in ALLOWED_TOOLS:
                value["name"] = name.removeprefix(NAMESPACE + "__") if to_cli else name
                if to_cli:
                    value["namespace"] = NAMESPACE
                else:
                    value.pop("namespace", None)
        for child in value.values():
            map_functions(child, to_cli)
    elif isinstance(value, list):
        for child in value:
            map_functions(child, to_cli)


def mapped_line(line: bytes) -> bytes:
    if line.startswith(b"data:"):
        try:
            event = json.loads(line[5:])
        except (ValueError, UnicodeError, TypeError):
            return line
        guard_functions(event)
        map_functions(event, True)
        return b"data: " + json.dumps(event, ensure_ascii=False).encode()
    return line


def prepare_payload(body: dict[str, Any], model: str, effort: str, budget: int) -> dict:
    if body.get("model") != model:
        raise ValueError("model_mismatch")
    if (
        not isinstance(body.get("reasoning"), dict)
        or body["reasoning"].get("effort") != effort
    ):
        raise ValueError("budget_mismatch")
    if body.get("stream") is not True or not isinstance(body.get("input"), list):
        raise ValueError("protocol_error")
    value = copy.deepcopy(body)
    # Codex has no documented per-response output setting. Keep the original
    # request, then set and record the actual provider budget explicitly.
    value["max_output_tokens"] = budget
    value["tools"] = copy.deepcopy(FUNCTIONS)
    value["tool_choice"] = "auto"
    # No native tools, deferred discovery or setup generation is forwarded.
    history = []
    for item in value["input"]:
        if isinstance(item, dict) and item.get("type") == "additional_tools":
            if (
                item.get("role") != "developer"
                or not isinstance(item.get("tools"), list)
                or set(item) - {"type", "id", "role", "tools"}
            ):
                raise ValueError("tool_profile_mismatch")
            continue
        if not isinstance(item, dict) or item.get("type") not in {
            "message",
            "reasoning",
            "function_call",
            "function_call_output",
        }:
            raise ValueError("tool_profile_mismatch")
        if (
            item.get("type") == "function_call"
            and function_name(item) not in ALLOWED_TOOLS
        ):
            raise ValueError("tool_profile_mismatch")
        history.append(item)
    value["input"] = history
    map_functions(value["input"], False)
    return value


def inspect_stream(
    body: str, model: str, effort: str, budget: int | None = None
) -> dict[str, Any]:
    parsed = parse_stream(body, allowed_tools=ALLOWED_TOOLS)
    response = parsed["response"]
    status = "ok"
    terminal = parsed["terminal_event"]
    failure = parsed["stream_error"]
    output = response.get("output", [])
    text = "".join(
        part.get("text", part.get("refusal", ""))
        for item in output
        if item.get("type") == "message"
        for part in item.get("content", [])
    )
    if failure:
        status = (
            "transport_failed" if failure.get("type") == "error" else "protocol_error"
        )
    elif not terminal or terminal == "response.failed":
        status = "transport_failed"
    elif response.get("model") != model:
        status = "model_mismatch"
    elif (response.get("reasoning") or {}).get("effort") != effort or (
        budget is not None
        and response.get("max_output_tokens") is not None
        and response["max_output_tokens"] != budget
    ):
        status = "budget_mismatch"
    elif terminal == "response.incomplete":
        reason = (response.get("incomplete_details") or {}).get("reason")
        status = {
            "max_output_tokens": "truncated",
            "content_filter": "refusal",
            "context_window_exceeded": "context_limit",
        }.get(str(reason), "generation_incomplete")
    elif any(
        part.get("type") == "refusal"
        for item in output
        for part in item.get("content", [])
    ):
        status = "refusal"
    elif response.get("error") or any(
        item.get("status", "completed") != "completed" for item in output
    ):
        status = "protocol_error"
    elif not text.strip() and not any(
        item.get("type") == "function_call" for item in output
    ):
        status = "empty"
    return {
        "status": status,
        "answer": text,
        "output_budget_echo_matches": (
            response["max_output_tokens"] == budget
            if response.get("max_output_tokens") is not None
            else None
        ),
        **parsed,
    }


class ResponsesProxy(MessagesProxy):
    def __init__(self, *args, auth_mode: str = "api_key", **kwargs):
        super().__init__(*args, **kwargs)
        self.auth_mode = auth_mode
        self.controls: list[dict[str, Any]] = []
        self.previous_payload: dict[str, Any] | None = None
        self.last_response: dict[str, Any] | None = None

    async def __aenter__(self):
        app = web.Application(client_max_size=32 * 1024**2)
        app.router.add_post("/{path:.*}", self.forward)
        app.router.add_get("/{path:.*}", self.forward_ws)
        self.runner = web.AppRunner(app, shutdown_timeout=1)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{self.runner.addresses[0][1]}"
        return self

    async def forward_ws(self, request: web.Request) -> web.StreamResponse:
        if request.path != f"/{self.token}/responses" or self.failure.is_set():
            return web.json_response(
                {"error": {"message": "Local request rejected"}}, status=403
            )
        channel = web.WebSocketResponse(max_msg_size=32 * 1024**2)
        await channel.prepare(request)
        async for frame in channel:
            if frame.type != aiohttp.WSMsgType.TEXT:
                break
            try:
                body = json.loads(frame.data)
            except ValueError:
                await channel.send_json(
                    {"type": "error", "error": {"message": "Invalid WebSocket frame"}}
                )
                break
            if isinstance(body, dict) and body.get("generate") is False:
                # Codex prewarm is connection setup, not an inference. A local
                # empty acknowledgement must never create a billed model call
                # or supply a generated answer/tool result to the agent.
                response = {
                    "id": "prewarm_" + uuid.uuid4().hex,
                    "object": "response",
                    "model": self.api_model,
                    "status": "completed",
                    "output": [],
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                }
                control = {
                    "kind": "local_prewarm_acknowledgement",
                    "original_payload": body,
                    "response": response,
                    "upstream_forwarded": False,
                }
                self.controls.append(control)
                save_record(
                    self.directory / f"control_{len(self.controls):03}.json",
                    json.loads(
                        redact(json.dumps(control), self.secrets + [self.token])
                    ),
                )
                await channel.send_json(
                    {
                        "type": "response.created",
                        "response": {**response, "status": "in_progress"},
                    }
                )
                await channel.send_json(
                    {"type": "response.completed", "response": response}
                )
                continue
            result = await self.forward(request, body=body, channel=channel)
            if isinstance(result, web.Response):
                await channel.send_json(
                    {
                        "type": "error",
                        "error": {
                            "message": "Local experiment profile rejected",
                            "status": result.status,
                        },
                    }
                )
                break
            if self.failure.is_set():
                break
        await channel.close()
        return channel

    async def emit(
        self, channel: web.WebSocketResponse | None, downstream: Any, line: bytes
    ) -> None:
        mapped = mapped_line(line)
        if channel is None:
            await downstream.write(mapped + b"\n")
        elif mapped.startswith(b"data:"):
            if mapped[5:].strip() != b"[DONE]":
                await channel.send_json(json.loads(mapped[5:]))

    async def forward(
        self,
        request: web.Request,
        *,
        body: Any = None,
        channel: web.WebSocketResponse | None = None,
    ) -> web.StreamResponse:
        # Random path authenticates the loopback session even when Codex sends
        # subscription bearer credentials instead of a local API token.
        if request.path != f"/{self.token}/responses" or (
            self.auth_mode == "api_key"
            and request.headers.get("Authorization") != "Bearer " + self.token
        ):
            return web.json_response(
                {"error": {"message": "Local request rejected"}}, status=403
            )
        if self.failure.is_set():
            return web.json_response(
                {"error": {"message": "Attempt ended; no nested retry"}}, status=409
            )
        try:
            if body is None:
                body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("protocol_error")
        except (ValueError, TypeError):
            return web.json_response(
                {"error": {"message": "Malformed local payload"}}, status=400
            )
        call: dict[str, Any] = {
            "index": len(self.calls) + 1,
            "path": "/responses",
            "original_payload": body,
            "status": "running",
            "cli_transport": "websocket" if channel is not None else "http",
            "normalizations": [
                "fixed_output_budget",
                "shared_mcp_tools",
                "function_namespaces",
            ],
        }
        self.calls.append(call)
        raw = self.directory / f"api_{call['index']:03}.sse"
        call["raw_path"] = str(raw)
        try:
            incoming = copy.deepcopy(body)
            if channel is not None:
                if incoming.pop("type", None) != "response.create":
                    raise ValueError("protocol_error")
                call["normalizations"].append("websocket_to_http")
                previous_id = incoming.pop("previous_response_id", None)
                if previous_id:
                    if (
                        self.last_response is None
                        or self.previous_payload is None
                        or previous_id != self.last_response["id"]
                    ):
                        raise ValueError("protocol_error")
                    incoming["input"] = (
                        copy.deepcopy(self.previous_payload["input"])
                        + copy.deepcopy(self.last_response["output"])
                        + incoming.get("input", [])
                    )
                    call["normalizations"].append(
                        "materialized_previous_response_history"
                    )
                incoming["stream"] = True
            if incoming.get("generate") is not None:
                if incoming.pop("generate") is not True:
                    raise ValueError("protocol_error")
            call["payload"] = prepare_payload(
                incoming, self.api_model, self.effort, self.max_tokens
            )
            if any(
                item.get("type") == "additional_tools"
                for item in incoming["input"]
                if isinstance(item, dict)
            ):
                call["normalizations"].append("removed_inline_tool_declarations")
        except ValueError as exc:
            call["status"] = str(exc)
            self.persist(call)
            return web.json_response(
                {"error": {"message": "Experiment profile mismatch"}}, status=400
            )
        headers = {"Content-Type": "application/json"}
        if self.auth_mode == "api_key":
            headers["Authorization"] = "Bearer " + self.key
        else:
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or not auth[7:].strip():
                call["status"] = "http_error"
                call["http_status"] = 401
                self.persist(call)
                return web.json_response(
                    {"error": {"message": "Subscription authentication missing"}},
                    status=401,
                )
            self.secrets.append(auth[7:])
            headers["Authorization"] = auth
            if request.headers.get("ChatGPT-Account-ID"):
                headers["ChatGPT-Account-ID"] = request.headers["ChatGPT-Account-ID"]
            for name in ("OpenAI-Beta", "originator", "version", "session_id"):
                if request.headers.get(name):
                    headers[name] = request.headers[name]
        self.persist(call)
        chunks = []
        pending = b""
        downstream = None
        try:
            async with self.session.post(
                self.endpoint,
                json=call["payload"],
                headers=headers,
                allow_redirects=False,
            ) as response:
                call["http_status"] = response.status
                if channel is None:
                    downstream = web.StreamResponse(
                        status=response.status,
                        headers={
                            "Content-Type": response.headers.get(
                                "Content-Type", "application/json"
                            )
                        },
                    )
                    await downstream.prepare(request)
                else:
                    downstream = channel
                with raw.open("w") as out:
                    async for chunk in response.content.iter_chunked(65536):
                        chunks.append(chunk)
                        pending += chunk
                        while b"\n" in pending:
                            line, pending = pending.split(b"\n", 1)
                            out.write(
                                redact(
                                    line.decode("utf-8", "replace"),
                                    self.secrets + [self.token],
                                )
                                + "\n"
                            )
                            out.flush()
                            if channel is None or response.status == 200:
                                await self.emit(channel, downstream, line)
                    if pending:
                        out.write(
                            redact(
                                pending.decode("utf-8", "replace"),
                                self.secrets + [self.token],
                            )
                        )
                        if channel is None or response.status == 200:
                            await self.emit(channel, downstream, pending)
                        pending = b""
                    out.flush()
                    os.fsync(out.fileno())
                if response.status != 200:
                    text = b"".join(chunks).decode("utf-8", "replace").lower()
                    if self.auth_mode == "chatgpt" and any(
                        s in text
                        for s in (
                            "usage_limit_reached",
                            "insufficient_quota",
                            "usage limit exceeded",
                        )
                    ):
                        call["status"] = "quota_exhausted"
                    else:
                        call["status"] = (
                            "transport_failed"
                            if response.status in (408, 429) or response.status >= 500
                            else "http_error"
                        )
                else:
                    call.update(
                        inspect_stream(
                            b"".join(chunks).decode("utf-8"),
                            self.api_model,
                            self.effort,
                            self.max_tokens,
                        )
                    )
                if call["status"] == "ok":
                    self.previous_payload = copy.deepcopy(call["payload"])
                    self.last_response = copy.deepcopy(call["response"])
                if channel is None:
                    await downstream.write_eof()
                elif response.status != 200:
                    await channel.send_json(
                        {
                            "type": "error",
                            "error": {
                                "message": "Upstream request failed",
                                "status": response.status,
                            },
                        }
                    )
        except ToolProfileError as exc:
            call.update(status="tool_profile_mismatch", error=str(exc))
        except (aiohttp.ClientError, OSError, UnicodeError) as exc:
            call.update(status="transport_failed", error=type(exc).__name__)
        finally:
            if pending:
                with raw.open("a") as out:
                    out.write(
                        redact(
                            pending.decode("utf-8", "replace"),
                            self.secrets + [self.token],
                        )
                    )
                    out.flush()
                    os.fsync(out.fileno())
            if call["status"] == "running":
                call["status"] = "transport_failed"
            self.persist(call)
        return (
            downstream
            if downstream is not None
            else web.json_response(
                {"error": {"message": "Upstream transport failure"}}, status=502
            )
        )
