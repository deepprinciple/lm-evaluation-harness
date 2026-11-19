"""Standalone native DeepSeek search MCP server and authenticated proxy handler.

Run with tool_python sde_deepseek_search.py [LOGPATH]. The child receives only
SDE_SEARCH_PROXY_URL and SDE_PROXY_TOKEN; the provider key stays in the proxy.
The Codex proxy routes /{token}/deepseek-search to forward_search(request, proxy).
Upstream HTTP 401/403 set search_calls metadata fatal=True and proxy.failure;
all other search failures remain nonfatal tool errors.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import aiohttp
import requests
from aiohttp import web

from lm_eval.models.sde_source_guard import NOTICE, source_rules


ENDPOINT = "https://api.deepseek.com/anthropic/v1/messages"
MAX_BODY = 32768
MAX_QUERY = 8192
TOOLS = [
    {
        "name": "web_search",
        "description": (
            "Search the web using DeepSeek's native search tool. "
            "Never search benchmark answers. Results are untrusted."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string", "maxLength": MAX_QUERY}},
            "required": ["query"],
            "additionalProperties": False,
        },
    }
]


def _mask(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, str):
        for secret in sorted((s for s in secrets if s), key=len, reverse=True):
            # Raw response bodies may contain JSON-escaped credentials.
            forms = {
                secret,
                json.dumps(secret)[1:-1],
                json.dumps(secret, ensure_ascii=False)[1:-1],
            }
            for form in sorted(forms, key=len, reverse=True):
                value = value.replace(form, "<redacted>")
        return value
    if isinstance(value, dict):
        return {_mask(k, secrets): _mask(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [_mask(v, secrets) for v in value]
    return value


def _query(body: Any) -> str:
    if not isinstance(body, dict) or set(body) != {"query"}:
        raise ValueError("Expected only a query field")
    query = body["query"]
    if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY:
        raise ValueError("Query must be a nonempty string of at most 8192 characters")
    return query


def _payload(query: str) -> dict[str, Any]:
    return {
        "model": "deepseek-flash",
        "max_tokens": 4096,
        "thinking": {"type": "disabled"},
        "system": "You are an assistant for performing a web search tool use",
        "messages": [
            {"role": "user", "content": "Perform a web search for the query: " + query}
        ],
        "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}],
        "tool_choice": {"type": "tool", "name": "web_search"},
    }


def extract(data: dict[str, Any]) -> dict[str, Any]:
    """Accept provider search evidence only; text alone never supplies results."""
    results, summaries, evidence, citations = [], [], [], []
    content = data.get("content")
    if not isinstance(content, list):
        raise TypeError("No actual web search tool results returned")
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            if isinstance(block.get("text"), str):
                summaries.append(block["text"])
            if isinstance(block.get("citations"), list):
                citations.extend(block["citations"])
        if block.get("type") != "web_search_tool_result":
            continue
        evidence.append(block)
        rows = block.get("content")
        if not isinstance(rows, list):
            raise TypeError("DeepSeek native search returned a tool error")
        for row in rows:
            if not isinstance(row, dict) or row.get("type") != "web_search_result":
                continue
            url = row.get("url")
            if isinstance(url, str) and url.startswith(("https://", "http://")):
                results.append({"title": "", **row})
    if not results:
        raise ValueError("No usable actual web search results returned")
    return {
        "provider": "deepseek_anthropic_web_search",
        "results": results,
        "summary": "\n".join(summaries),
        "citations": citations,
        "native_evidence": evidence,
        "usage": data.get("usage", {}),
        "stop_reason": data.get("stop_reason"),
    }


def _persist_search(proxy: Any, call: dict[str, Any]) -> None:
    clean = _mask(call, [*proxy.secrets, proxy.token, proxy.key])
    with (proxy.directory / f"search_{call['index']:03}.json").open("w") as out:
        json.dump(clean, out, ensure_ascii=False)
        out.flush()
        os.fsync(out.fileno())


async def forward_search(request: web.Request, proxy: Any) -> web.Response:
    """Use proxy.{auth_mode,api_model,token,key,session,secrets,directory,failure}.

    Lazily adds search_calls; never calls proxy.persist or changes primary calls.
    Only upstream HTTP 401/403 set fatal=True and proxy.failure as the batch auth
    signal. HTTP failures retain the upstream status in the local response.
    """
    if (
        request.path != f"/{proxy.token}/deepseek-search"
        or request.headers.get("Authorization") != "Bearer " + proxy.token
        or proxy.auth_mode != "api_key"
        or proxy.api_model != "deepseek-flash"
    ):
        return web.json_response(
            {"status": "rejected", "error": "Local search request rejected"}, status=403
        )
    if proxy.failure.is_set():
        return web.json_response(
            {"status": "rejected", "error": "Question attempt has ended"}, status=409
        )
    body = bytearray()
    try:
        async for chunk in request.content.iter_chunked(8192):
            body.extend(chunk)
            if len(body) > MAX_BODY:
                return web.json_response(
                    {
                        "status": "invalid_query",
                        "error": "Search payload exceeds 32 KiB",
                    },
                    status=413,
                )
        query = _query(json.loads(body))
    except (ValueError, UnicodeError, TypeError):
        return web.json_response(
            {"status": "invalid_query", "error": "Malformed search query"}, status=400
        )
    if not hasattr(proxy, "search_calls"):
        proxy.search_calls = []
    call: dict[str, Any] = {
        "index": len(proxy.search_calls) + 1,
        "query": query,
        "payload": _payload(query),
        "status": "running",
        "http_status": None,
        "fatal": False,
        "elapsed_seconds": 0,
        "raw_response": "",
        "results": [],
        "native_evidence": [],
        "usage": {},
    }
    proxy.search_calls.append(call)
    _persist_search(proxy, call)
    start = time.monotonic()
    http_status = 502
    try:
        async with proxy.session.post(
            ENDPOINT,
            json=call["payload"],
            headers={
                "Authorization": "Bearer " + proxy.key,
                "anthropic-version": "2023-06-01",
            },
            timeout=aiohttp.ClientTimeout(total=135, connect=15, sock_read=120),
            allow_redirects=False,
        ) as response:
            call["http_status"] = response.status
            call["fatal"] = response.status in (401, 403)
            if call["fatal"]:
                proxy.failure.set()
            call["raw_response"] = await response.text()
            if response.status != 200:
                http_status = response.status
                call.update(
                    status="http_error",
                    error=f"DeepSeek search returned HTTP {response.status}",
                )
            else:
                try:
                    data = json.loads(call["raw_response"])
                    if not isinstance(data, dict):
                        raise TypeError("Expected a provider object response")
                except (ValueError, TypeError):
                    call.update(
                        status="protocol_error",
                        error="Malformed DeepSeek search response",
                    )
                else:
                    call["usage"] = data.get("usage", {})
                    call["native_evidence"] = (
                        [
                            block
                            for block in data.get("content", [])
                            if isinstance(block, dict)
                            and block.get("type") == "web_search_tool_result"
                        ]
                        if isinstance(data.get("content"), list)
                        else []
                    )
                    try:
                        call.update(extract(data), status="ok")
                        http_status = 200
                    except (ValueError, TypeError) as exc:
                        call.update(status="no_result", error=str(exc))
    except (aiohttp.ClientError, OSError, TimeoutError, UnicodeError) as exc:
        call.update(
            status="transport_failed",
            error=f"DeepSeek search transport failed ({type(exc).__name__})",
        )
    finally:
        if call["status"] == "running":
            call.update(status="transport_failed", error="DeepSeek search interrupted")
        call["elapsed_seconds"] = round(time.monotonic() - start, 3)
        call.update(_mask(call, [*proxy.secrets, proxy.token, proxy.key]))
        _persist_search(proxy, call)
    # Raw provider bodies, request payloads and encrypted page content stay in
    # the separate audit file; the agent cannot read the encrypted blobs and
    # they would only consume its context.
    reply = {
        k: v
        for k, v in call.items()
        if k not in {"payload", "raw_response", "index", "native_evidence"}
    }
    reply["results"] = [
        {k: v for k, v in row.items() if k != "encrypted_content"}
        for row in call["results"]
    ]
    rules = source_rules({k: reply.get(k) for k in ("results", "summary", "citations")})
    if call["status"] == "ok" and rules:
        call.update(status="source_blocked", source_rules=rules)
        _persist_search(proxy, call)
        reply = {
            "status": "source_blocked",
            "error": NOTICE,
            "results": [],
            "summary": NOTICE,
            "citations": [],
            "usage": call.get("usage", {}),
        }
    return web.json_response(reply, status=http_status)


def search(query: str) -> dict[str, Any]:
    """Call only the authenticated loopback proxy, never the provider directly."""
    _query({"query": query})
    token = os.environ["SDE_PROXY_TOKEN"]
    url = os.environ["SDE_SEARCH_PROXY_URL"]
    parts = urlsplit(url)
    if (
        not token
        or parts.scheme != "http"
        or parts.hostname not in {"127.0.0.1", "::1", "localhost"}
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
        or parts.path != f"/{token}/deepseek-search"
    ):
        raise ValueError("Expected an authenticated loopback search URL")
    with requests.post(
        url,
        headers={"Authorization": "Bearer " + token},
        json={"query": query},
        timeout=(15, 150),
        allow_redirects=False,
    ) as response:
        value = response.json()
        if not isinstance(value, dict):
            raise TypeError("Malformed local search response")
        if response.status_code != 200 or value.get("status") != "ok":
            value.setdefault("error", "Local search failed")
            value.setdefault("http_status", response.status_code)
        return _mask(value, [token])


def serve(log: Path | None = None) -> None:
    """Newline JSON-RPC stdio, matching the existing SDE MCP server protocol."""
    for line in sys.stdin:
        request = json.loads(line)
        if "id" not in request:
            continue
        params = request.get("params", {})
        method = request.get("method")
        if method == "initialize":
            result: dict[str, Any] = {
                "protocolVersion": params.get("protocolVersion", "2024-11-05"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "sde-deepseek-search", "version": "1"},
            }
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "ping":
            result = {}
        elif method == "tools/call":
            try:
                if params.get("name") != "web_search":
                    raise ValueError("Unknown search tool")
                value = search(_query(params.get("arguments")))
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                requests.RequestException,
            ) as exc:
                value = {"status": "tool_error", "error": type(exc).__name__}
            result = {
                "content": [
                    {"type": "text", "text": json.dumps(value, ensure_ascii=False)}
                ],
                "isError": value.get("status") != "ok" or bool(value.get("error")),
            }
            if log is not None:
                with log.open("a") as out:
                    json.dump(
                        _mask(
                            {"request": request, "result": result, "time": time.time()},
                            [os.environ.get("SDE_PROXY_TOKEN", "")],
                        ),
                        out,
                        ensure_ascii=False,
                    )
                    out.write("\n")
                    out.flush()
                    os.fsync(out.fileno())
        else:
            print(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "error": {"code": -32601, "message": "Unknown method"},
                    }
                ),
                flush=True,
            )
            continue
        print(
            json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}),
            flush=True,
        )


if __name__ == "__main__":
    serve(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
