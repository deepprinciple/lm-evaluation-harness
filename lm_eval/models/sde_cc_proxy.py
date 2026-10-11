"""Transparent local Messages capture for recorded Claude Code runs."""

import asyncio
import contextlib
import json
import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web

from lm_eval.models.sde_chat import save_record
from lm_eval.models.sde_messages import parse_stream
from lm_eval.models.sde_stream import (
    PRE_RESPONSE_RETRY_STATUSES,
    connect_failed,
    pre_response_retryable,
    retry_delay,
)


def redact(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<redacted>")
    return text


SUMMARY_FIELDS = (
    "index",
    "path",
    "kind",
    "backend",
    "request_kind",
    "profile_enforcement",
    "cli_transport",
    "status",
    "http_status",
    "fatal",
    "error",
    "stream_error",
    "transport_tail_error",
    "raw_path",
    "response_path",
    "response_format",
    "actual_profile",
    "profile_mismatches",
    "normalizations",
    "answer_selection",
    "output_budget_echo_matches",
    "terminal_event",
    "messages",
    "helper_evidence",
    "query",
    "upstream_forwarded",
    "elapsed_seconds",
)


def call_summary(call: dict[str, Any]) -> dict[str, Any]:
    """Sample-log view of a recorded call; payloads stay in the api_NNN files.

    Each agent request carries the whole conversation, so logging full calls
    grows quadratically with the number of turns.
    """
    summary = {field: call[field] for field in SUMMARY_FIELDS if field in call}
    response = call.get("response")
    if isinstance(response, dict):
        summary["model"] = response.get("model")
    usage = response.get("usage") if isinstance(response, dict) else None
    summary["usage"] = usage or call.get("usage") or {}
    if isinstance(call.get("upstream_retries"), list):
        summary["upstream_retries"] = len(call["upstream_retries"])
    if isinstance(call.get("results"), list):
        summary["result_count"] = len(call["results"])
    return summary


def inspect_stream(body: str, expected_model: str) -> dict[str, Any]:
    parsed = parse_stream(body, expected_model, agent_mode=True)
    segments = parsed["relay_segments"]
    response = parsed["response"]
    messages = [
        {
            "id": segment["response"].get("id"),
            "model": segment["response"].get("model"),
            "usage": segment["response"].get("usage", {}),
            "stop_reason": segment["response"].get("stop_reason"),
            "closed": segment["message_stop"],
        }
        for segment in segments
    ]
    failure = parsed["stream_error"]
    tool_calls = [
        block
        for segment in segments
        for block in segment["response"]["content"]
        if "tool" in block["type"]
    ]
    tool_errors = [
        block
        for block in tool_calls
        if block.get("is_error") is True
        or block["type"].endswith("_error")
        or isinstance(block.get("content"), dict)
        and str(block["content"].get("type", "")).endswith("_error")
    ]
    status = "ok"
    if failure:
        status = (
            "transport_failed" if failure.get("type") == "error" else "protocol_error"
        )
    elif response.get("model") != expected_model:
        status = "model_mismatch"
    elif not parsed["message_stop"] or not parsed["blocks_closed"]:
        status = "transport_failed"
    elif response.get("stop_reason") not in {
        "end_turn",
        "tool_use",
        "max_tokens",
        "refusal",
        "model_context_window_exceeded",
        "pause_turn",
    }:
        status = "protocol_error"
    elif tool_errors:
        status = "tool_error"
    return {
        "status": status,
        "messages": messages,
        "error": failure,
        "event_counts": parsed["event_counts"],
        "tool_calls": tool_calls,
        "tool_errors": tool_errors,
        "response": response,
    }


def inspect_message(body: str, expected_model: str) -> dict[str, Any]:
    """Validate atomic Messages responses using the same agent block parser."""
    try:
        message = json.loads(body)
        if not isinstance(message, dict) or not isinstance(
            message.get("content"), list
        ):
            raise TypeError("Invalid Messages response")
        events: list[dict[str, Any]] = [
            {"type": "message_start", "message": {**message, "content": []}}
        ]
        for index, block in enumerate(message["content"]):
            events.extend(
                [
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": block,
                    },
                    {"type": "content_block_stop", "index": index},
                ]
            )
        events.append({"type": "message_stop"})
        result = inspect_stream(
            "".join("data: " + json.dumps(event) + "\n\n" for event in events),
            expected_model,
        )
        # These are synthetic parser events, not events observed on the wire.
        result["event_counts"] = {}
        return result
    except (ValueError, TypeError) as exc:
        return {"status": "protocol_error", "error": str(exc)}


def is_message_stop(line: bytes) -> bool:
    if not line.startswith(b"data:") or b"message_stop" not in line:
        return False
    try:
        return json.loads(line[5:]).get("type") == "message_stop"
    except (ValueError, AttributeError):
        return False


def inspect_body(content_type: str, chunks: list[bytes], model: str) -> dict:
    json_body = "application/json" in content_type
    inspect = inspect_message if json_body else inspect_stream
    return {
        "response_format": "json" if json_body else "sse",
        **inspect(b"".join(chunks).decode("utf-8"), model),
    }


def helper_evidence(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Recognize isolated native web helpers from server tool declarations.

    Main agent requests also declare client tools. A budget/model difference or
    a search mention in the prompt alone is never evidence of a helper.
    """
    tools = body.get("tools")
    if not isinstance(tools, list) or not tools:
        return []
    if not all(
        isinstance(tool, dict)
        and isinstance(tool.get("type"), str)
        and tool["type"].startswith(("web_search_", "web_fetch_"))
        for tool in tools
    ):
        return []
    choice = body.get("tool_choice", {})
    if not isinstance(choice, dict) or choice.get("type") == "none":
        return []
    return [{"type": tool["type"], "name": tool.get("name")} for tool in tools]


class BufferedResponse:
    """An upstream error body already read for a retry decision, replayed once."""

    def __init__(self, status: int, headers: Any, body: bytes):
        self.status, self.headers, self.body = status, headers, body
        self.content = self

    async def iter_chunked(self, size: int) -> AsyncIterator[bytes]:
        if self.body:
            yield self.body


class MessagesProxy:
    """Capture unchanged request/response bodies; enforce the declared profile."""

    def __init__(
        self,
        session: Any,
        endpoint: str,
        api_model: str,
        effort: str,
        max_tokens: int,
        key: str,
        directory: Path,
        secrets: list[str],
    ):
        self.session, self.endpoint = session, endpoint.rstrip("/")
        self.api_model, self.effort, self.max_tokens = api_model, effort, max_tokens
        self.key, self.directory, self.secrets = key, directory, [key, *secrets]
        self.token = uuid.uuid4().hex
        self.calls: list[dict[str, Any]] = []
        self.failure = asyncio.Event()

    async def __aenter__(self):
        app = web.Application(client_max_size=32 * 1024**2)
        app.router.add_post("/{path:.*}", self.forward)
        self.runner = web.AppRunner(app, shutdown_timeout=1)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        addresses = self.runner.addresses
        self.url = f"http://127.0.0.1:{addresses[0][1]}"
        return self

    async def __aexit__(self, *args):
        await self.runner.cleanup()

    @contextlib.asynccontextmanager
    async def upstream(
        self, url: str, body: Any, headers: dict[str, str], call: dict[str, Any]
    ) -> AsyncIterator[Any]:
        """Retry failures before any upstream generation; the CLI sees one response.

        Only connection failures and 429/503/529 without a quota marker are
        retried, each recorded under upstream_retries. The question deadline
        cancels the attempt, and with it these retries.
        """
        retries = 0
        while True:
            async with contextlib.AsyncExitStack() as stack:
                try:
                    response = await stack.enter_async_context(
                        self.session.post(
                            url, json=body, headers=headers, allow_redirects=False
                        )
                    )
                except aiohttp.ClientError as exc:
                    if not connect_failed(exc):
                        raise
                    retry_after = None
                    failure: dict[str, Any] = {"error": type(exc).__name__}
                else:
                    if response.status not in PRE_RESPONSE_RETRY_STATUSES:
                        yield response
                        return
                    wire = b"".join(
                        [chunk async for chunk in response.content.iter_chunked(65536)]
                    )
                    text = redact(
                        wire.decode("utf-8", "replace"), self.secrets + [self.token]
                    )
                    if not pre_response_retryable(response.status, text):
                        yield BufferedResponse(response.status, response.headers, wire)
                        return
                    retry_after = response.headers.get("Retry-After")
                    failure = {
                        "http_status": response.status,
                        "retry_after": retry_after,
                        "body": text,
                    }
            retries += 1
            call.setdefault("upstream_retries", []).append(failure)
            self.persist(call)
            await asyncio.sleep(retry_delay(retry_after, retries))

    def persist(self, call: dict[str, Any]) -> None:
        call["fatal"] = call["status"] not in ("running", "ok") and (
            call.get("request_kind") != "tool_helper"
            or call.get("http_status") in (401, 403)
            or call["status"] in ("protocol_error", "model_mismatch")
        )
        clean = json.loads(
            redact(json.dumps(call, ensure_ascii=False), self.secrets + [self.token])
        )
        save_record(self.directory / f"api_{call['index']:03}.json", clean)
        if call["fatal"]:
            self.failure.set()

    async def forward(self, request: web.Request) -> web.StreamResponse:
        if request.path not in ("/v1/messages", "/v1/messages/count_tokens") or (
            request.headers.get("x-api-key") != self.token
            and request.headers.get("Authorization") != "Bearer " + self.token
        ):
            return web.json_response(
                {
                    "error": {
                        "type": "authentication_error",
                        "message": "Local request rejected",
                    }
                },
                status=403,
            )
        if self.failure.is_set():
            return web.json_response(
                {
                    "error": {
                        "message": "This question attempt has ended; no nested retry"
                    }
                },
                status=409,
            )
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise TypeError("Expected object payload")
        except (ValueError, TypeError):
            return web.json_response(
                {"error": {"message": "Malformed local payload"}}, status=400
            )
        call: dict[str, Any] = {
            "index": len(self.calls) + 1,
            "path": request.path,
            "payload": body,
            "status": "running",
        }
        self.calls.append(call)
        raw = self.directory / f"api_{call['index']:03}.sse"
        call["raw_path"] = str(raw)
        messages = request.path == "/v1/messages"
        evidence = helper_evidence(body)
        main = messages and not evidence
        call["request_kind"] = (
            "count_tokens" if not messages else "tool_helper" if evidence else "main"
        )
        initial_main = main and not any(
            previous.get("request_kind") == "main" for previous in self.calls[:-1]
        )
        # Followup purpose cannot be inferred reliably from the native wire format.
        call["profile_enforcement"] = (
            "initial_main_strict"
            if initial_main
            else "native_followup"
            if main
            else "isolated_web_helper"
            if messages
            else "count_tokens_model_only"
        )
        call["helper_evidence"] = evidence
        call["expected_profile"] = {
            "model": self.api_model,
            "max_tokens": self.max_tokens,
            "effort": self.effort,
        }
        output_config = body.get("output_config")
        call["actual_profile"] = {
            "model": body.get("model"),
            "max_tokens": body.get("max_tokens"),
            "effort": (
                output_config.get("effort") if isinstance(output_config, dict) else None
            ),
        }
        call["profile_mismatches"] = [
            field
            for field, value in call["expected_profile"].items()
            if call["actual_profile"][field] != value
        ]
        status = None
        if (
            not isinstance(body.get("model"), str)
            or not body["model"]
            or not evidence
            and body["model"] != self.api_model
        ):
            status = "model_mismatch"
        elif initial_main and (
            not isinstance(body.get("output_config"), dict)
            or body.get("max_tokens") != self.max_tokens
            or body.get("output_config", {}).get("effort") != self.effort
        ):
            status = "budget_mismatch"
        if status:
            call["status"] = status
            self.persist(call)
            return web.json_response(
                {
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Request does not match the recorded experiment profile",
                    }
                },
                status=400,
            )
        self.persist(call)
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.key,
            "anthropic-version": request.headers.get("anthropic-version", "2023-06-01"),
        }
        if request.headers.get("anthropic-beta"):
            headers["anthropic-beta"] = request.headers["anthropic-beta"]
        chunks: list[bytes] = []
        pending = b""
        downstream = None
        content_type = ""
        stop_seen = forwarded = False
        try:
            async with self.upstream(
                self.endpoint + request.path, body, headers, call
            ) as response:
                call["http_status"] = response.status
                content_type = response.headers.get("Content-Type", "")
                downstream = web.StreamResponse(
                    status=response.status,
                    headers={
                        "Content-Type": response.headers.get(
                            "Content-Type", "application/json"
                        )
                    },
                )
                await downstream.prepare(request)
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
                            stop_seen = stop_seen or is_message_stop(line)
                        await downstream.write(chunk)
                        forwarded = stop_seen
                    forwarded = True
                    if pending:
                        out.write(
                            redact(
                                pending.decode("utf-8", "replace"),
                                self.secrets + [self.token],
                            )
                        )
                        pending = b""
                    out.flush()
                    os.fsync(out.fileno())
                if response.status != 200:
                    call["status"] = (
                        "transport_failed"
                        if response.status in (408, 429) or response.status >= 500
                        else "http_error"
                    )
                elif messages:
                    call.update(inspect_body(content_type, chunks, body["model"]))
                else:
                    call["status"] = "ok"
                await downstream.write_eof()
        except aiohttp.ClientConnectionResetError as exc:
            # As in the Responses proxy, a client reset after the complete
            # upstream answer was forwarded is a tail event, not a failed call.
            try:
                if not forwarded or call.get("http_status") != 200:
                    raise ValueError("reset before the complete answer was forwarded")
                if messages:
                    call.update(inspect_body(content_type, chunks, body["model"]))
                else:
                    call["status"] = "ok"
                call["transport_tail_error"] = type(exc).__name__
            except ValueError:
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
            self.persist(call)
        return (
            downstream
            if downstream is not None
            else web.json_response(
                {"error": {"message": "Upstream transport failure"}}, status=502
            )
        )
