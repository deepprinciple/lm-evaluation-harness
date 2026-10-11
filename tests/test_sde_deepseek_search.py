from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from aiohttp import web


MODULE = Path(__file__).resolve().parents[1] / "lm_eval/models/sde_deepseek_search.py"
spec = importlib.util.spec_from_file_location("sde_deepseek_search", MODULE)
assert spec is not None and spec.loader is not None
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


def native_response() -> dict[str, Any]:
    return {
        "content": [
            {
                "type": "web_search_tool_result",
                "tool_use_id": "search1",
                "content": [
                    {
                        "type": "web_search_result",
                        "title": "Scientific source",
                        "url": "https://example.org/paper",
                        "snippet": "Observed evidence",
                        "encrypted_content": "provider evidence",
                    }
                ],
            },
            {
                "type": "text",
                "text": "Provider summary",
                "citations": [
                    {
                        "type": "web_search_result_location",
                        "url": "https://example.org/paper",
                        "title": "Scientific source",
                        "cited_text": "Cited page snippet",
                    }
                ],
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 4},
        "stop_reason": "end_turn",
    }


def proxy(tmp_path: Path, data: Any = None, status: int = 200) -> Any:
    async def text() -> str:
        await asyncio.sleep(0)
        return json.dumps(data or native_response())

    response = SimpleNamespace(
        status=status,
        text=AsyncMock(side_effect=text),
    )
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=response)
    context.__aexit__ = AsyncMock(return_value=False)
    return SimpleNamespace(
        token="random-loopback-token",
        key="fake-provider-key",
        secrets=["fake-provider-key", "another-secret"],
        auth_mode="api_key",
        api_model="deepseek-flash",
        failure=asyncio.Event(),
        calls=[{"status": "ok", "answer": "primary answer"}],
        directory=tmp_path,
        session=SimpleNamespace(post=MagicMock(return_value=context)),
    )


def request(p: Any, body: Any = None, *, raw: bytes | None = None) -> Any:
    encoded = (
        raw if raw is not None else json.dumps(body or {"query": "science"}).encode()
    )

    async def chunks(size: int) -> Any:
        for start in range(0, len(encoded), size):
            yield encoded[start : start + size]

    return SimpleNamespace(
        path=f"/{p.token}/deepseek-search",
        headers={"Authorization": "Bearer " + p.token},
        content=SimpleNamespace(iter_chunked=chunks),
    )


def result(response: web.Response) -> Any:
    return json.loads(response.text or "{}")


def test_native_payload_evidence_and_separate_records(tmp_path: Path) -> None:
    p = proxy(tmp_path)
    reply = asyncio.run(bridge.forward_search(request(p), p))
    assert reply.status == 200 and result(reply)["status"] == "ok"
    args, kwargs = p.session.post.call_args
    assert args == ("https://api.deepseek.com/anthropic/v1/messages",)
    assert kwargs["headers"] == {
        "Authorization": "Bearer fake-provider-key",
        "anthropic-version": "2023-06-01",
    }
    assert kwargs["allow_redirects"] is False
    assert kwargs["json"] == {
        "model": "deepseek-flash",
        "max_tokens": 4096,
        "thinking": {"type": "disabled"},
        "system": "You are an assistant for performing a web search tool use",
        "messages": [
            {"role": "user", "content": "Perform a web search for the query: science"}
        ],
        "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}],
        "tool_choice": {"type": "tool", "name": "web_search"},
    }
    saved = json.loads((tmp_path / "search_001.json").read_text())
    assert saved["status"] == "ok" and saved["http_status"] == 200
    assert saved["fatal"] is False and result(reply)["fatal"] is False
    assert saved["elapsed_seconds"] >= 0
    assert json.loads(saved["raw_response"]) == native_response()
    assert saved["usage"] == native_response()["usage"]
    assert saved["native_evidence"] == [native_response()["content"][0]]
    assert result(reply)["results"][0]["snippet"] == "Observed evidence"
    assert result(reply)["citations"][0]["cited_text"] == "Cited page snippet"
    assert "native_evidence" not in result(reply)
    assert "encrypted_content" not in result(reply)["results"][0]
    assert saved["results"][0]["encrypted_content"] == "provider evidence"
    assert p.calls == [{"status": "ok", "answer": "primary answer"}]
    assert not p.failure.is_set()


@pytest.mark.parametrize(
    "change", ["path", "bearer", "auth_mode", "api_model", "failure"]
)
def test_rejected_profile_never_dispatches(tmp_path: Path, change: str) -> None:
    p = proxy(tmp_path)
    req = request(p)
    if change == "path":
        req.path = "/wrong/deepseek-search"
    elif change == "bearer":
        req.headers = {"Authorization": "Bearer wrong"}
    elif change == "failure":
        p.failure.set()
    else:
        setattr(p, change, "chatgpt" if change == "auth_mode" else "deepseek-pro")
    reply = asyncio.run(bridge.forward_search(req, p))
    assert reply.status in (403, 409)
    assert result(reply)["error"]
    p.session.post.assert_not_called()
    assert not list(tmp_path.iterdir())
    assert not hasattr(p, "search_calls")
    assert p.failure.is_set() is (change == "failure")


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        b"[]",
        b"null",
        b"{}",
        b'{"query":1}',
        b'{"query":"   "}',
        b'{"query":"science","extra":1}',
        json.dumps({"query": "q" * 8193}).encode(),
        b" " * 32769,
    ],
)
def test_bounded_query_validation(tmp_path: Path, raw: bytes) -> None:
    p = proxy(tmp_path)
    reply = asyncio.run(bridge.forward_search(request(p, raw=raw), p))
    assert reply.status in (400, 413)
    p.session.post.assert_not_called()


@pytest.mark.parametrize(
    "content",
    [
        [{"type": "text", "text": "Here are invented search results"}],
        [{"type": "web_search_result", "url": "https://example.org"}],
        [
            {
                "type": "web_search_tool_result",
                "content": {
                    "type": "web_search_tool_result_error",
                    "error_code": "unavailable",
                },
            }
        ],
        [{"type": "web_search_tool_result", "content": []}],
        [
            {
                "type": "web_search_tool_result",
                "content": [{"type": "web_search_result", "url": ""}],
            }
        ],
    ],
)
def test_actual_tool_results_required(tmp_path: Path, content: Any) -> None:
    p = proxy(tmp_path, {"content": content})
    reply = asyncio.run(bridge.forward_search(request(p), p))
    assert reply.status == 502 and result(reply)["status"] == "no_result"
    assert result(reply)["error"] and not result(reply).get("results")
    assert p.search_calls[0]["status"] == "no_result"
    assert p.search_calls[0]["fatal"] is False and result(reply)["fatal"] is False
    assert not p.failure.is_set()


@pytest.mark.parametrize("status", [301, 401, 403, 429, 503])
def test_provider_http_errors_preserved(tmp_path: Path, status: int) -> None:
    p = proxy(tmp_path, {"error": {"message": "upstream failed"}}, status)
    reply = asyncio.run(bridge.forward_search(request(p), p))
    assert reply.status == status
    assert result(reply)["http_status"] == status
    assert result(reply)["error"] and result(reply)["status"] == "http_error"
    saved = json.loads((tmp_path / "search_001.json").read_text())
    assert json.loads(saved["raw_response"])["error"]["message"] == "upstream failed"
    fatal = status in (401, 403)
    assert p.failure.is_set() is fatal
    assert saved["fatal"] is fatal and result(reply)["fatal"] is fatal
    assert p.search_calls[0]["fatal"] is fatal
    assert p.calls == [{"status": "ok", "answer": "primary answer"}]
    p.session.post.return_value.__aenter__.return_value.status = 200
    p.session.post.return_value.__aenter__.return_value.text.side_effect = None
    p.session.post.return_value.__aenter__.return_value.text.return_value = json.dumps(
        native_response()
    )
    next_reply = asyncio.run(bridge.forward_search(request(p), p))
    assert next_reply.status == (409 if fatal else 200)
    assert p.session.post.call_count == (1 if fatal else 2)


@pytest.mark.parametrize(
    "failure", ["invalid_json", "bad_shape", "transport", "timeout"]
)
def test_protocol_and_transport_errors_are_tool_errors(
    tmp_path: Path, failure: str
) -> None:
    p = proxy(tmp_path)
    upstream = p.session.post.return_value.__aenter__.return_value
    if failure == "invalid_json":
        upstream.text.side_effect = None
        upstream.text.return_value = "not json"
    elif failure == "bad_shape":
        upstream.text.side_effect = None
        upstream.text.return_value = "[]"
    else:
        p.session.post.side_effect = (
            aiohttp.ClientError("fake-provider-key")
            if failure == "transport"
            else TimeoutError()
        )
    reply = asyncio.run(bridge.forward_search(request(p), p))
    assert reply.status == 502 and result(reply)["error"]
    assert "fake-provider-key" not in (reply.text or "")
    assert not p.failure.is_set()
    assert len(p.search_calls) == 1 and (tmp_path / "search_001.json").exists()
    assert p.search_calls[0]["fatal"] is False and result(reply)["fatal"] is False


def test_secret_masking_in_every_record_and_returned_result(tmp_path: Path) -> None:
    p = proxy(tmp_path)
    p.key = 'fake-"provider\\key'
    p.secrets = [p.key, "another-secret"]
    data = native_response()
    leaked = f"{p.key} {p.token} another-secret"
    data["content"][0]["content"][0]["title"] = leaked
    data["content"][1]["text"] = leaked
    p.session.post.return_value.__aenter__.return_value.text.side_effect = None
    p.session.post.return_value.__aenter__.return_value.text.return_value = json.dumps(
        data
    )
    reply = asyncio.run(bridge.forward_search(request(p, {"query": leaked}), p))
    saved = json.loads((tmp_path / "search_001.json").read_text())
    for value in [
        result(reply),
        saved,
        json.loads(saved["raw_response"]),
        p.search_calls,
    ]:
        encoded = json.dumps(value)
        for secret in [p.key, p.token, "another-secret"]:
            assert json.dumps(secret)[1:-1] not in encoded
    assert result(reply)["status"] == "ok"


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/source",
        "https://github.com/deepprinciple/source",
        "https://github.com/howiehwong/sde-harness",
        "https://arxiv.org/abs/2512.15567",
        "https://arxiv.org/pdf/2512.15567",
    ],
)
def test_native_search_does_not_filter_source_domains(tmp_path: Path, url: str) -> None:
    data = native_response()
    data["content"][0]["content"][0]["url"] = url
    p = proxy(tmp_path, data)
    reply = asyncio.run(bridge.forward_search(request(p), p))
    assert reply.status == 200 and result(reply)["results"][0]["url"] == url


def test_concurrent_searches_have_no_quota_or_primary_calls(tmp_path: Path) -> None:
    p = proxy(tmp_path)

    async def run() -> Any:
        return await asyncio.gather(
            *(bridge.forward_search(request(p), p) for _ in range(6))
        )

    replies = asyncio.run(run())
    assert all(reply.status == 200 for reply in replies)
    assert p.session.post.call_count == 6
    assert len(p.search_calls) == len(list(tmp_path.glob("search_*.json"))) == 6
    assert p.calls == [{"status": "ok", "answer": "primary answer"}]
    assert not p.failure.is_set()


def test_stdio_subprocess_loopback_and_flushed_log(tmp_path: Path) -> None:
    async def run() -> None:
        p = proxy(tmp_path)
        app = web.Application()

        async def forward(req: web.Request) -> web.Response:
            return await bridge.forward_search(req, p)

        app.router.add_post("/{path:.*}", forward)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        log = tmp_path / "mcp.jsonl"
        env = {
            "PATH": os.environ["PATH"],
            "SDE_SEARCH_PROXY_URL": f"http://127.0.0.1:{port}/{p.token}/deepseek-search",
            "SDE_PROXY_TOKEN": p.token,
        }
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(MODULE),
            str(log),
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdin is not None and proc.stdout is not None
        stdin, stdout = proc.stdin, proc.stdout

        async def send(method: str, params: Any = None, ident: int = 1) -> Any:
            stdin.write(
                (
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": ident,
                            "method": method,
                            "params": params or {},
                        }
                    )
                    + "\n"
                ).encode()
            )
            await stdin.drain()
            return json.loads(await asyncio.wait_for(stdout.readline(), 10))

        try:
            init = await send("initialize", {"protocolVersion": "2024-11-05"})
            assert init["result"]["protocolVersion"] == "2024-11-05"
            proc.stdin.write(
                b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n'
            )
            tools = await send("tools/list")
            assert [tool["name"] for tool in tools["result"]["tools"]] == ["web_search"]
            assert (await send("ping"))["result"] == {}
            for index in range(4):
                reply = await send(
                    "tools/call",
                    {"name": "web_search", "arguments": {"query": "science"}},
                    index,
                )
                assert reply["result"]["isError"] is False
                assert json.loads(reply["result"]["content"][0]["text"])["results"]
                assert len(log.read_text().splitlines()) == index + 1
            bad = await send(
                "tools/call",
                {"name": "read_url", "arguments": {"url": "https://example.org"}},
            )
            assert bad["result"]["isError"] is True
            p.session.post.return_value.__aenter__.return_value.status = 429
            error = await send(
                "tools/call", {"name": "web_search", "arguments": {"query": "science"}}
            )
            value = json.loads(error["result"]["content"][0]["text"])
            assert error["result"]["isError"] is True and value["http_status"] == 429
            unknown = await send("unknown")
            assert unknown["error"]["code"] == -32601
            assert p.token not in log.read_text() and p.key not in log.read_text()
            assert p.session.post.call_count == 5 and not p.failure.is_set()
        finally:
            proc.stdin.close()
            await asyncio.wait_for(proc.wait(), 10)
            await runner.cleanup()
        assert proc.returncode == 0
        assert proc.stderr is not None and not await proc.stderr.read()

    asyncio.run(run())


@pytest.mark.parametrize(
    "url",
    [
        "https://example.org/search",
        "http://example.org/search",
        "http://127.0.0.1:1234/wrong",
    ],
)
def test_stdio_rejects_non_loopback_or_wrong_token_path(
    url: str, monkeypatch: Any
) -> None:
    monkeypatch.setenv("SDE_SEARCH_PROXY_URL", url)
    monkeypatch.setenv("SDE_PROXY_TOKEN", "random-token")
    with pytest.raises(ValueError):
        bridge.search("science")
