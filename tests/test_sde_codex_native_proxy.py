"""Offline native Codex proxy protocol and transport regressions."""

import asyncio
import copy
import json
import sys
import types
from contextlib import asynccontextmanager
from typing import Any

import aiohttp
import pytest
from aiohttp import web

from lm_eval.models.sde_codex_proxy import (
    ResponsesProxy,
    ToolProfileError,
    guard_functions,
    inspect_stream,
    mapped_line,
    prepare_payload,
)
from tests.test_sde_responses import NATIVE_ITEMS, native_events, stream


def payload():
    return {
        "model": "test-model",
        "reasoning": {"effort": "max"},
        "stream": True,
        "input": [
            {"role": "user", "content": "scientific question"},
            {
                "type": "additional_tools",
                "id": "at1",
                "role": "developer",
                "tools": [
                    {
                        "type": "namespace",
                        "name": "plugin_namespace",
                        "tools": [{"type": "custom", "name": "apply_patch"}],
                    }
                ],
            },
            *copy.deepcopy(NATIVE_ITEMS),
            {"type": "function_call_output", "call_id": "c1", "output": "raw result"},
            {"type": "custom_tool_call_output", "call_id": "c2", "output": "patched"},
        ],
        "tools": [
            {"type": "web_search"},
            {
                "type": "namespace",
                "name": "functions",
                "tools": [
                    {"type": "function", "name": "exec_command", "parameters": {}}
                ],
            },
        ],
        "additional_tools": [{"type": "custom", "name": "unfamiliar_tool"}],
        "tool_choice": {
            "type": "function",
            "namespace": "functions",
            "name": "exec_command",
        },
        "parallel_tool_calls": True,
    }


@pytest.mark.parametrize("auth_mode", ["chatgpt", "api_key"])
def test_native_search_preserves_provider_request_and_raw_result(
    tmp_path, monkeypatch, auth_mode
):
    import lm_eval.models.sde_codex_proxy as module

    body = {
        "model": "test-model",
        "commands": {"search_query": [{"q": "Python hypot"}]},
        "settings": {"external_web_access": True},
        "max_output_tokens": 10000,
    }
    answer = {
        "output": "official docs",
        "encrypted_output": "opaque provider value",
        "results": [{"url": "https://docs.python.org/3/library/math.html"}],
    }
    seen = []

    async def run():
        app = web.Application()

        async def search(request):
            seen.append(await request.json())
            assert request.headers["Authorization"] == "Bearer fake-subscription"
            return web.json_response(answer)

        app.router.add_post("/alpha/search", search)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        endpoint = f"http://127.0.0.1:{runner.addresses[0][1]}/responses"
        monkeypatch.setattr(module, "SUBSCRIPTION_ENDPOINT", endpoint)
        try:
            async with (
                aiohttp.ClientSession() as session,
                ResponsesProxy(
                    session,
                    endpoint,
                    "test-model",
                    "max",
                    128000,
                    "fake-key",
                    tmp_path,
                    [],
                    auth_mode=auth_mode,
                ) as proxy,
            ):
                async with session.post(
                    proxy.url + "/" + proxy.token + "/alpha/search",
                    json=body,
                    headers={"Authorization": "Bearer fake-subscription"},
                ) as response:
                    assert response.status == (200 if auth_mode == "chatgpt" else 403)
                    if auth_mode == "chatgpt":
                        assert await response.json() == answer
                        assert proxy.search_calls[0]["payload"] == body
                        assert proxy.search_calls[0]["status"] == "ok"
                        raw = tmp_path / "native_search_001.response.json"
                        assert json.loads(raw.read_text()) == answer
        finally:
            await runner.cleanup()
        assert seen == ([body] if auth_mode == "chatgpt" else [])
        assert "fake-subscription" not in "".join(
            p.read_text() for p in tmp_path.glob("*.json")
        )

    asyncio.run(run())


@pytest.mark.parametrize("after_terminal", [False, True])
def test_downstream_reset_requires_a_valid_terminal(tmp_path, after_terminal):
    async def upstream(request):
        rows = native_events(NATIVE_ITEMS[1])
        for row in (rows[0], rows[-1]):
            row["response"]["model"] = "test-model"
        return web.Response(
            body=stream(rows, done=False) + "data: [DONE]\n\n",
            content_type="text/event-stream",
        )

    async def run():
        async with (
            local_provider(upstream) as endpoint,
            aiohttp.ClientSession() as session,
            ResponsesProxy(
                session, endpoint, "test-model", "max", 128000, "fake-key", tmp_path, []
            ) as proxy,
        ):
            emit = proxy.emit

            async def reset(channel, downstream, line):
                if (after_terminal and line == b"data: [DONE]") or (
                    not after_terminal and b"response.created" in line
                ):
                    raise aiohttp.ClientConnectionResetError("downstream closed")
                await emit(channel, downstream, line)

            proxy.emit = reset
            async with session.post(
                proxy.url + "/" + proxy.token + "/responses",
                json=payload(),
                headers={"Authorization": "Bearer " + proxy.token},
            ) as result:
                await result.read()
            call = proxy.calls[0]
            assert call["status"] == ("ok" if after_terminal else "transport_failed")
            assert proxy.failure.is_set() is not after_terminal
            if after_terminal:
                assert call["transport_tail_error"] == "ClientConnectionResetError"
                assert call["response"]["id"] in proxy.response_history

    asyncio.run(run())


def test_payload_preserves_tools_namespaces_choice_and_full_history():
    body = payload()
    original = copy.deepcopy(body)
    effective = prepare_payload(body, "test-model", "max", 128000)
    assert effective == {**original, "max_output_tokens": 128000}
    assert body == original
    effective["input"][0]["content"] = "changed"
    assert body == original


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("model", "other", "model_mismatch"),
        ("reasoning", {"effort": "high"}, "budget_mismatch"),
        ("stream", False, "protocol_error"),
        ("input", "bad", "protocol_error"),
        ("input", [None], "protocol_error"),
        ("tools", "bad", "protocol_error"),
    ],
)
def test_request_profile_and_protocol_checks_remain(field, value, error):
    body = payload()
    body[field] = value
    with pytest.raises(ValueError, match=error):
        prepare_payload(body, "test-model", "max", 128000)


@pytest.mark.parametrize("item", NATIVE_ITEMS, ids=lambda item: item["type"])
def test_native_lines_are_exact_bytes_with_no_namespace_mapping(item):
    event = {"type": "response.output_item.done", "item": item, "output_index": 0}
    line = (
        b"data:   "
        + json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode()
        + b"\r"
    )
    assert mapped_line(line) == line
    value = copy.deepcopy(event)
    guard_functions(value)
    assert value == event


@pytest.mark.parametrize(
    "name,namespace", [(None, "functions"), ("shell", 1), ("", "functions")]
)
def test_invalid_native_names_still_fail_closed(name, namespace):
    line = (
        b"data: "
        + json.dumps(
            {"item": {"type": "function_call", "name": name, "namespace": namespace}}
        ).encode()
    )
    with pytest.raises(ToolProfileError):
        mapped_line(line)


@pytest.mark.parametrize("item", NATIVE_ITEMS[:14], ids=lambda item: item["type"])
def test_proxy_accepts_native_tool_turns(item):
    assert (
        inspect_stream(stream(native_events(item)), "gpt-6-astra", "max", 128000)[
            "status"
        ]
        == "ok"
    )


@pytest.mark.parametrize(
    "change,status",
    [
        ("model", "model_mismatch"),
        ("effort", "budget_mismatch"),
        ("budget", "budget_mismatch"),
        ("bad_budget_type", "protocol_error"),
        ("item_status", "protocol_error"),
        ("unknown_item", "protocol_error"),
        ("interrupted", "transport_failed"),
        ("error", "transport_failed"),
    ],
)
def test_native_proxy_retains_failure_and_budget_validation(change, status):
    rows = native_events(copy.deepcopy(NATIVE_ITEMS[2]))
    final = rows[-1]["response"]
    if change == "model":
        for row in (rows[0], rows[-1]):
            row["response"]["model"] = "other"
    elif change == "effort":
        final["reasoning"] = {"effort": "high"}
    elif change in {"budget", "bad_budget_type"}:
        final["max_output_tokens"] = 5 if change == "budget" else "128000"
    elif change == "item_status":
        final["output"][0]["status"] = "searching"
    elif change == "unknown_item":
        final["output"] = [{"type": "unknown_item"}]
    elif change == "interrupted":
        rows.pop()
    elif change == "error":
        rows = [{"type": "error", "error": {"message": "busy"}}]
    assert (
        inspect_stream(stream(rows), "gpt-6-astra", "max", 128000)["status"] == status
    )


@asynccontextmanager
async def local_provider(handler):
    app = web.Application()
    app.router.add_post("/responses", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    try:
        yield f"http://127.0.0.1:{runner.addresses[0][1]}/responses"
    finally:
        await runner.cleanup()


def test_http_records_original_effective_requests_and_exact_native_sse(tmp_path):
    body = payload()
    rows = native_events(NATIVE_ITEMS[0])
    for row in (rows[0], rows[-1]):
        row["response"]["model"] = "test-model"
    wire = stream(rows, crlf=True).encode()
    seen = []

    async def upstream(request):
        seen.append(await request.json())
        return web.Response(body=wire, content_type="text/event-stream")

    async def run():
        async with (
            local_provider(upstream) as endpoint,
            aiohttp.ClientSession() as session,
            ResponsesProxy(
                session,
                endpoint,
                "test-model",
                "max",
                128000,
                "fake-provider-key",
                tmp_path,
                [],
            ) as proxy,
        ):
            async with session.post(
                proxy.url + "/" + proxy.token + "/responses",
                json=body,
                headers={"Authorization": "Bearer " + proxy.token},
            ) as result:
                assert result.status == 200
                assert await result.read() == wire
            call = json.loads((tmp_path / "api_001.json").read_text())
            assert call["status"] == "ok"
            assert call["original_payload"] == body
            assert call["payload"] == seen[0] == {**body, "max_output_tokens": 128000}
            assert call["normalizations"] == ["fixed_output_budget"]
            assert proxy.search_calls == []
            assert "fake-provider-key" not in (tmp_path / "api_001.json").read_text()
            assert not proxy.failure.is_set()

    asyncio.run(run())


@pytest.mark.parametrize("child_connection", [False, True])
def test_websocket_materializes_native_history_and_preserves_names(
    tmp_path, child_connection
):
    calls: list[dict[str, Any]] = []

    async def upstream(request):
        calls.append(await request.json())
        rows = native_events(NATIVE_ITEMS[1])
        for row in (rows[0], rows[-1]):
            row["response"].update(model="test-model", id=f"r{len(calls)}")
        return web.Response(body=stream(rows), content_type="text/event-stream")

    async def run():
        async with (
            local_provider(upstream) as endpoint,
            aiohttp.ClientSession() as session,
            ResponsesProxy(
                session, endpoint, "test-model", "max", 128000, "fake-key", tmp_path, []
            ) as proxy,
        ):
            async with session.ws_connect(
                proxy.url + "/" + proxy.token + "/responses",
                headers={"Authorization": "Bearer " + proxy.token},
            ) as ws:
                first = {**payload(), "type": "response.create"}
                await ws.send_json(first)
                received = []
                while True:
                    event = await ws.receive_json()
                    received.append(event)
                    if event["type"] == "response.completed":
                        break
                final = received[-1]["response"]
                assert final["output"][0] == NATIVE_ITEMS[1]
                # A child completes between the root's turns, on either the
                # same transport or a separate subagent WebSocket connection.
                child_request = {
                    **first,
                    "input": [{"role": "user", "content": "child question"}],
                }
                if child_connection:
                    async with session.ws_connect(
                        proxy.url + "/" + proxy.token + "/responses",
                        headers={"Authorization": "Bearer " + proxy.token},
                    ) as child:
                        await child.send_json(child_request)
                        while (await child.receive_json())[
                            "type"
                        ] != "response.completed":
                            pass
                else:
                    await ws.send_json(child_request)
                    while (await ws.receive_json())["type"] != "response.completed":
                        pass
                second_input = [
                    {
                        "type": "custom_tool_call_output",
                        "call_id": "c2",
                        "output": "done",
                    }
                ]
                await ws.send_json(
                    {**first, "previous_response_id": "r1", "input": second_input}
                )
                while (await ws.receive_json())["type"] != "response.completed":
                    pass
            assert calls[2]["input"] == first["input"] + final["output"] + second_input
            assert calls[2]["tools"] == first["tools"]
            assert proxy.calls[-1]["status"] == "ok"
            assert proxy.calls[-1]["original_payload"]["input"] == second_input

    asyncio.run(run())


def test_deepseek_search_route_is_separate_and_precedes_wildcard(monkeypatch, tmp_path):
    seen = []

    async def forward_search(request, proxy):
        seen.append((request.path, proxy))
        proxy.search_calls.append({"status": "ok"})
        return web.json_response({"search": "recorded"})

    helper = types.ModuleType("lm_eval.models.sde_deepseek_search")
    helper.__dict__["forward_search"] = forward_search
    monkeypatch.setitem(sys.modules, helper.__name__, helper)

    async def run():
        async with (
            aiohttp.ClientSession() as session,
            ResponsesProxy(
                session,
                "http://unused.invalid",
                "deepseek-flash",
                "max",
                128000,
                "fake-key",
                tmp_path,
                [],
            ) as proxy,
        ):
            async with session.post(
                proxy.url + "/" + proxy.token + "/deepseek-search",
                json={"query": "science"},
            ) as result:
                assert result.status == 200
                assert await result.json() == {"search": "recorded"}
            assert seen == [(f"/{proxy.token}/deepseek-search", proxy)]
            assert proxy.search_calls == [{"status": "ok"}]
            assert proxy.calls == []

    asyncio.run(run())


@pytest.mark.parametrize(
    "item",
    [
        {"type": "additional_tools", "role": "user", "tools": []},
        {"type": "additional_tools", "role": "developer", "tools": "bad"},
        {"type": "additional_tools", "role": "developer", "tools": [], "content": 5},
        {"type": "function_call", "name": "valid_name"},
        {"type": "custom_tool_call", "name": "valid_name", "call_id": "c1"},
    ],
)
def test_known_malformed_history_is_protocol_error(item):
    body = payload()
    body["input"] = [item]
    with pytest.raises(ValueError, match="protocol_error"):
        prepare_payload(body, "test-model", "max", 128000)


def test_inline_tools_without_optional_id_are_preserved():
    body = payload()
    body["input"] = [{"type": "additional_tools", "role": "developer", "tools": []}]
    assert prepare_payload(body, "test-model", "max", 128000)["input"] == body["input"]


def test_malformed_history_type_is_protocol_error():
    body = payload()
    body["input"] = [{"type": []}]
    with pytest.raises(ValueError, match="protocol_error"):
        prepare_payload(body, "test-model", "max", 128000)
