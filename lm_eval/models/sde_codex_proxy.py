"""Recorded native Responses forwarding with explicit output-budget handling."""

import copy
import json
import os
import uuid
from typing import Any

import aiohttp
from aiohttp import web

from lm_eval.models.sde_cc_proxy import MessagesProxy, redact
from lm_eval.models.sde_chat import save_record
from lm_eval.models.sde_responses import parse_stream, validate_native_item


SUBSCRIPTION_ENDPOINT = "https://chatgpt.com/backend-api/codex/responses"


class ToolProfileError(ValueError):
    pass


def guard_functions(value: Any) -> None:
    """Validate named-call fields without restricting tool names or namespaces."""
    if isinstance(value, dict):
        if value.get("type") in ("function_call", "custom_tool_call", "mcp_call"):
            if not isinstance(value.get("name"), str) or not value["name"]:
                raise ToolProfileError("Invalid tool name")
            if value.get("namespace") is not None and not isinstance(
                value["namespace"], str
            ):
                raise ToolProfileError("Invalid tool namespace")
        for child in value.values():
            guard_functions(child)
    elif isinstance(value, list):
        for child in value:
            guard_functions(child)


def mapped_line(line: bytes) -> bytes:
    """Preserve the provider's SSE bytes, including native tool namespaces."""
    if line.startswith(b"data:"):
        try:
            event = json.loads(line[5:])
        except (ValueError, UnicodeError, TypeError):
            return line
        guard_functions(event)
    return line


def prepare_payload(
    body: dict[str, Any],
    model: str,
    effort: str,
    budget: int,
    *,
    hosted_web_search: bool = False,
) -> dict:
    if body.get("model") != model:
        raise ValueError("model_mismatch")
    if (
        not isinstance(body.get("reasoning"), dict)
        or body["reasoning"].get("effort") != effort
    ):
        raise ValueError("budget_mismatch")
    if body.get("stream") is not True or not isinstance(body.get("input"), list):
        raise ValueError("protocol_error")
    if any(not isinstance(item, dict) for item in body["input"]):
        raise ValueError("protocol_error")
    for field in ("tools", "additional_tools"):
        if field in body and (
            not isinstance(body[field], list)
            or any(not isinstance(tool, dict) for tool in body[field])
        ):
            raise ValueError("protocol_error")
    for item in body["input"]:
        if "type" in item and not isinstance(item["type"], str):
            raise ValueError("protocol_error")
        if item.get("type") == "additional_tools" and (
            item.get("role") != "developer"
            or not isinstance(item.get("tools"), list)
            or any(not isinstance(tool, dict) for tool in item["tools"])
            or set(item) - {"type", "id", "role", "tools"}
            or ("id" in item and not isinstance(item["id"], str))
        ):
            raise ValueError("protocol_error")
        if item.get("type") in {
            "function_call",
            "custom_tool_call",
            "function_call_output",
            "custom_tool_call_output",
        }:
            try:
                validate_native_item(item)
            except (ValueError, TypeError) as exc:
                raise ValueError("protocol_error") from exc
    try:
        guard_functions(body["input"])
    except ToolProfileError as exc:
        raise ValueError("protocol_error") from exc
    value = copy.deepcopy(body)
    # Codex has no documented per-response output setting. Keep the original
    # request, then set and record the actual provider budget explicitly.
    value["max_output_tokens"] = budget
    if hosted_web_search and not any(
        tool.get("type") in {"web_search", "web_search_preview"}
        for tool in value.get("tools", [])
    ):
        value.setdefault("tools", []).append({"type": "web_search"})
    return value


def inspect_stream(
    body: str,
    model: str,
    effort: str,
    budget: int | None = None,
    *,
    allow_sparse_terminal: bool = False,
) -> dict[str, Any]:
    parsed = parse_stream(
        body, agent_mode=True, allow_sparse_terminal=allow_sparse_terminal
    )
    response = parsed["response"]
    status = "ok"
    terminal = parsed["terminal_event"]
    failure = parsed["stream_error"]
    output = response.get("output", [])
    final_messages = [
        item
        for item in output
        if item.get("type") == "message" and item.get("phase") == "final_answer"
    ]
    answer_items = final_messages or [
        item for item in output if item.get("phase") != "commentary"
    ]
    text = "".join(
        part.get("text", part.get("refusal", ""))
        for item in answer_items
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
        if item.get("type") == "message"
        for part in item.get("content", [])
    ):
        status = "refusal"
    elif (
        len(final_messages) > 1
        or response.get("error")
        or any(
            item.get("status", "completed") != "completed" or item.get("error")
            for item in output
        )
    ):
        status = "protocol_error"
    elif not text.strip() and not any(
        item.get("type")
        not in {"message", "reasoning", "compaction", "additional_tools"}
        for item in output
    ):
        status = "empty"
    return {
        "status": status,
        "answer": text,
        "answer_selection": "final_answer_phase"
        if final_messages
        else "non_commentary_messages",
        "output_budget_echo_matches": (
            response["max_output_tokens"] == budget
            if response.get("max_output_tokens") is not None
            else None
        ),
        **parsed,
    }


class ResponsesProxy(MessagesProxy):
    def __init__(
        self,
        *args,
        auth_mode: str = "api_key",
        hosted_web_search: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.auth_mode = auth_mode
        self.hosted_web_search = hosted_web_search
        self.controls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self.response_history: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}

    async def __aenter__(self):
        app = web.Application(client_max_size=32 * 1024**2)

        async def search(request: web.Request) -> web.StreamResponse:
            from lm_eval.models.sde_deepseek_search import forward_search

            return await forward_search(request, self)

        app.router.add_post(f"/{self.token}/deepseek-search", search)
        app.router.add_post(f"/{self.token}/alpha/search", self.forward_native_search)
        app.router.add_post("/{path:.*}", self.forward)
        app.router.add_get("/{path:.*}", self.forward_ws)
        self.runner = web.AppRunner(app, shutdown_timeout=1)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{self.runner.addresses[0][1]}"
        return self

    async def forward_native_search(self, request: web.Request) -> web.StreamResponse:
        """Preserve Codex's standalone subscription web tool and its own budget."""
        auth = request.headers.get("Authorization", "")
        if (
            self.auth_mode != "chatgpt"
            or self.endpoint != SUBSCRIPTION_ENDPOINT
            or not auth.startswith("Bearer ")
            or not auth[7:].strip()
        ):
            return web.json_response(
                {"error": {"message": "Native subscription search unavailable"}},
                status=403,
            )
        try:
            body = await request.json()
        except ValueError:
            return web.json_response({"error": "Invalid search request"}, status=400)
        if not isinstance(body, dict) or body.get("model") != self.api_model:
            return web.json_response({"error": "Search model mismatch"}, status=400)
        self.secrets.append(auth[7:])
        headers = {"Content-Type": "application/json", "Authorization": auth}
        for name in ("ChatGPT-Account-ID", "originator", "version", "session_id"):
            if request.headers.get(name):
                headers[name] = request.headers[name]
        prefix = self.directory / f"native_search_{len(self.search_calls) + 1:03}"
        call: dict[str, Any] = {
            "backend": "codex_subscription_native",
            "payload": copy.deepcopy(body),
            "status": "running",
            "response_path": str(prefix.with_suffix(".response.json")),
        }
        self.search_calls.append(call)
        try:
            async with self.session.post(
                SUBSCRIPTION_ENDPOINT.removesuffix("/responses") + "/alpha/search",
                json=body,
                headers=headers,
                allow_redirects=False,
            ) as response:
                wire = await response.read()
                prefix.with_suffix(".response.json").write_text(
                    redact(wire.decode("utf-8", "replace"), self.secrets + [self.token])
                )
                call.update(http_status=response.status, status="http_error")
                if response.status == 200:
                    result = json.loads(wire)
                    if (
                        not isinstance(result, dict)
                        or not isinstance(result.get("output"), str)
                        or not isinstance(result.get("results"), list)
                    ):
                        call["status"] = "protocol_error"
                        return web.json_response(
                            {"error": "Invalid search result"}, status=502
                        )
                    call["status"] = "ok"
                return web.Response(
                    body=wire,
                    status=response.status,
                    headers={
                        "Content-Type": response.headers.get(
                            "Content-Type", "application/json"
                        )
                    },
                )
        except (aiohttp.ClientError, OSError, ValueError) as exc:
            call.update(status="transport_failed", error=type(exc).__name__)
            return web.json_response(
                {"error": "Native search transport failed"}, status=502
            )
        finally:
            save_record(
                prefix.with_suffix(".json"),
                json.loads(redact(json.dumps(call), self.secrets + [self.token])),
            )

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
            "normalizations": ["fixed_output_budget"],
            "output_budget_limitation": (
                "Subscription output budget is a request parameter; enforcement "
                "is not independently verified, and absent echo is not confirmation."
                if self.auth_mode == "chatgpt"
                else None
            ),
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
                if previous_id is not None:
                    if (
                        not isinstance(previous_id, str)
                        or previous_id not in self.response_history
                    ):
                        raise ValueError("protocol_error")
                    if not isinstance(incoming.get("input"), list):
                        raise ValueError("protocol_error")
                    previous_payload, previous_response = self.response_history[
                        previous_id
                    ]
                    incoming["input"] = (
                        copy.deepcopy(previous_payload["input"])
                        + copy.deepcopy(previous_response["output"])
                        + incoming["input"]
                    )
                    call["normalizations"].append(
                        "materialized_previous_response_history"
                    )
                incoming["stream"] = True
            if incoming.get("generate") is not None:
                if incoming.pop("generate") is not True:
                    raise ValueError("protocol_error")
            call["payload"] = prepare_payload(
                incoming,
                self.api_model,
                self.effort,
                self.max_tokens,
                hosted_web_search=self.hosted_web_search,
            )
            if call["payload"].get("tools") != incoming.get("tools"):
                call["normalizations"].append("native_hosted_web_search_tool")
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
        terminal_forwarded = False
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
                                if (
                                    line.startswith(b"data:")
                                    and b"response.completed" in line
                                ):
                                    try:
                                        terminal_forwarded = (
                                            json.loads(line[5:]).get("type")
                                            == "response.completed"
                                        )
                                    except (ValueError, AttributeError):
                                        pass
                    if pending:
                        out.write(
                            redact(
                                pending.decode("utf-8", "replace"),
                                self.secrets + [self.token],
                            )
                        )
                        if channel is None or response.status == 200:
                            await self.emit(channel, downstream, pending)
                            if (
                                pending.startswith(b"data:")
                                and b"response.completed" in pending
                            ):
                                try:
                                    terminal_forwarded = (
                                        json.loads(pending[5:]).get("type")
                                        == "response.completed"
                                    )
                                except (ValueError, AttributeError):
                                    pass
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
                            allow_sparse_terminal=self.auth_mode == "chatgpt",
                        )
                    )
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
            call.update(status="protocol_error", error=str(exc))
        except aiohttp.ClientConnectionResetError as exc:
            parsed = inspect_stream(
                b"".join(chunks).decode("utf-8", errors="replace"),
                self.api_model,
                self.effort,
                self.max_tokens,
                allow_sparse_terminal=self.auth_mode == "chatgpt",
            )
            if terminal_forwarded and parsed["status"] == "ok":
                call.update(parsed, transport_tail_error=type(exc).__name__)
            else:
                call.update(status="transport_failed", error=type(exc).__name__)
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
            if call["status"] == "ok":
                self.response_history[call["response"]["id"]] = (
                    copy.deepcopy(call["payload"]),
                    copy.deepcopy(call["response"]),
                )
                if call.get("sparse_terminal_output"):
                    call["normalizations"].append(
                        "subscription_terminal_output_from_closed_items"
                    )
            self.persist(call)
        return (
            downstream
            if downstream is not None
            else web.json_response(
                {"error": {"message": "Upstream transport failure"}}, status=502
            )
        )
