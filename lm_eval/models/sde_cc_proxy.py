"""Transparent local Messages capture for recorded Claude Code runs."""

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web

from lm_eval.models.sde_chat import save_record
from lm_eval.models.sde_messages import parse_stream


ALLOWED_TOOLS = {
    "mcp__sde_tools__web_search",
    "mcp__sde_tools__read_url",
    "mcp__sde_tools__run_shell",
}


def redact(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<redacted>")
    return text


def inspect_stream(body: str, expected_model: str) -> dict[str, Any]:
    parsed = parse_stream(body, expected_model)
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
    status = "ok"
    if failure:
        status = (
            "transport_failed" if failure.get("type") == "error" else "protocol_error"
        )
    elif response.get("model") != expected_model:
        status = "model_mismatch"
    elif not parsed["message_stop"] or not parsed["blocks_closed"]:
        status = "transport_failed"
    elif any(
        block["type"] not in {"text", "thinking", "redacted_thinking", "tool_use"}
        or (
            block["type"] == "tool_use"
            and block.get("name") not in ALLOWED_TOOLS | {"EndConversation"}
        )
        for block in response["content"]
    ):
        status = "tool_profile_mismatch"
    elif response.get("stop_reason") not in {
        "end_turn",
        "tool_use",
        "max_tokens",
        "refusal",
        "model_context_window_exceeded",
    }:
        status = "protocol_error"
    return {
        "status": status,
        "messages": messages,
        "error": failure,
        "event_counts": parsed["event_counts"],
    }


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

    def persist(self, call: dict[str, Any]) -> None:
        clean = json.loads(
            redact(json.dumps(call, ensure_ascii=False), self.secrets + [self.token])
        )
        save_record(self.directory / f"api_{call['index']:03}.json", clean)
        if call["status"] not in ("running", "ok"):
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
                raise ValueError("Expected object payload")
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
        main = request.path == "/v1/messages"
        status = None
        if body.get("model") != self.api_model:
            status = "model_mismatch"
        elif main and (
            not isinstance(body.get("output_config"), dict)
            or body.get("max_tokens") != self.max_tokens
            or body.get("output_config", {}).get("effort") != self.effort
        ):
            status = "budget_mismatch"
        elif main and (
            not isinstance(body.get("tools", []), list)
            or any(
                not isinstance(t, dict)
                or t.get("name") not in ALLOWED_TOOLS | {"EndConversation"}
                for t in body.get("tools", [])
            )
        ):
            status = "tool_profile_mismatch"
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
        try:
            async with self.session.post(
                self.endpoint + request.path,
                json=body,
                headers=headers,
                allow_redirects=False,
            ) as response:
                call["http_status"] = response.status
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
                        await downstream.write(chunk)
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
                elif main:
                    call.update(
                        inspect_stream(b"".join(chunks).decode("utf-8"), self.api_model)
                    )
                else:
                    call["status"] = "ok"
                await downstream.write_eof()
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
