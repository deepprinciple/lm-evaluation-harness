import asyncio
import copy
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from datasets import Dataset

from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask
from lm_eval.evaluator import evaluate
from lm_eval.models import sde_codex as cx
from lm_eval.models.sde_codex_proxy import (
    ResponsesProxy,
    ToolProfileError,
    inspect_stream,
    mapped_line,
    prepare_payload,
)
from lm_eval.models.sde_responses import parse_stream


def stream(
    answer="<answer>A</answer>",
    *,
    function=None,
    model="test-model",
    effort="max",
    status="completed",
    reason=None,
    closed=True,
):
    output = [
        {
            "type": "message",
            "id": "msg1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": answer}],
        }
    ]
    if function:
        output = [
            {
                "type": "function_call",
                "id": "f1",
                "call_id": "c1",
                "name": function,
                "arguments": '{"query":"science"}',
                "status": "completed",
            }
        ]
    response = {
        "id": "r1",
        "model": model,
        "status": status,
        "output": output,
        "reasoning": {"effort": effort},
        "usage": {"input_tokens": 10, "output_tokens": 4},
    }
    if reason:
        response["incomplete_details"] = {"reason": reason}
    events = [
        {
            "type": "response.created",
            "response": {**response, "output": [], "status": "in_progress"},
        }
    ]
    if closed:
        events.append({"type": "response." + status, "response": response})
    return "".join("data: " + json.dumps(e) + "\n\n" for e in events)


def trace(answer="<answer>A</answer>", *, tool=None, error=None, completed=True):
    events: list[dict[str, Any]] = [
        {"type": "thread.started", "thread_id": "t1"},
        {"type": "turn.started"},
    ]
    if tool:
        events.append({"type": "item.completed", "item": tool})
    if error:
        events.append({"type": "turn.failed", "error": {"message": error}})
    if completed:
        events.extend(
            [
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": answer},
                },
                {"type": "turn.completed", "usage": {"output_tokens": 4}},
            ]
        )
    return "\n".join(json.dumps(e) for e in events) + "\n"


def payload():
    return {
        "model": "test-model",
        "reasoning": {"effort": "max"},
        "stream": True,
        "input": [
            {"type": "message", "role": "user", "content": "scientific question"}
        ],
        "instructions": "Frozen CLI system text",
        "tools": [{"type": "custom", "name": "apply_patch"}],
    }


@pytest.mark.parametrize("answer,status", [("wrong answer", "ok"), ("", "empty")])
def test_normal_answer_not_selected_by_score(answer, status):
    assert cx.summarize_trace(trace(answer), answer, 0)["status"] == status


@pytest.mark.parametrize(
    "body,answer,rc,status",
    [
        (trace(completed=False), "partial", 0, "transport_failed"),
        (trace(), "B", 0, "protocol_error"),
        (trace(), "<answer>A</answer>", 1, "transport_failed"),
        (trace() + "{bad", "A", 0, "protocol_error"),
        (trace() + '{"type":"turn.completed","usage":{}}', "A", 0, "protocol_error"),
        (trace(error="usage_limit_reached", completed=False), "", 1, "quota_exhausted"),
        (
            trace(tool={"type": "command_execution", "command": "echo A"}),
            "<answer>A</answer>",
            0,
            "ok",
        ),
        (
            trace(tool={"type": "web_search"}),
            "<answer>A</answer>",
            0,
            "ok",
        ),
        (
            trace(
                tool={"type": "mcp_tool_call", "server": "other", "tool": "run_shell"}
            ),
            "<answer>A</answer>",
            0,
            "ok",
        ),
    ],
)
def test_trace_failures(body, answer, rc, status):
    assert cx.summarize_trace(body, answer, rc)["status"] == status


def test_shared_mcp_trace_is_allowed():
    t = {
        "type": "mcp_tool_call",
        "server": "sde_tools",
        "tool": "run_shell",
        "result": {"text": "5"},
    }
    result = cx.summarize_trace(trace(tool=t), "<answer>A</answer>", 0)
    assert result["status"] == "ok" and result["tool_calls"] == [t]


@pytest.mark.parametrize(
    "body,status",
    [
        (stream(function="mcp__sde_tools__web_search"), "ok"),
        (stream(function="shell"), "ok"),
        (stream(model="other"), "model_mismatch"),
        (stream(effort="low"), "budget_mismatch"),
        (stream(status="incomplete", reason="max_output_tokens"), "truncated"),
        (stream(status="incomplete", reason="content_filter"), "refusal"),
        (
            stream(status="incomplete", reason="context_window_exceeded"),
            "context_limit",
        ),
        (stream(closed=False), "transport_failed"),
        ("data: {bad}\n\n", "protocol_error"),
    ],
)
def test_provider_completion_and_profile(body, status):
    assert inspect_stream(body, "test-model", "max")["status"] == status


def test_no_tools_responses_still_rejects_functions():
    body = stream(function="mcp__sde_tools__run_shell")
    assert parse_stream(body)["stream_error"] is not None
    assert (
        parse_stream(body, allowed_tools={"mcp__sde_tools__run_shell"})["stream_error"]
        is None
    )


def test_provider_output_budget_echo_is_not_assumed():
    result = inspect_stream(stream(), "test-model", "max", 128000)
    assert result["output_budget_echo_matches"] is None
    body = stream().replace(
        '"status": "completed", "output"',
        '"max_output_tokens": 555, "status": "completed", "output"',
    )
    assert (
        inspect_stream(body, "test-model", "max", 128000)["status"] == "budget_mismatch"
    )


def test_refusal_content_is_retained_but_not_normal():
    body = stream().replace(
        '"type": "output_text", "text": "<answer>A</answer>"',
        '"type": "refusal", "refusal": "cannot answer"',
    )
    result = inspect_stream(body, "test-model", "max")
    assert result["status"] == "refusal" and result["answer"] == "cannot answer"


def test_budget_and_tools_normalization_preserves_prompt_and_history():
    body = payload()
    body["input"].extend(
        [
            {
                "type": "function_call",
                "namespace": "mcp__sde_tools",
                "name": "web_search",
                "call_id": "c1",
                "arguments": '{"query":"x"}',
            },
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": "raw tool result",
            },
        ]
    )
    original = copy.deepcopy(body)
    value = prepare_payload(body, "test-model", "max", 128000)
    assert body == original
    assert value["instructions"] == body["instructions"]
    assert value["input"][0] == body["input"][0]
    assert value["input"][-1] == body["input"][-1]
    assert value["input"][1] == original["input"][1]
    assert value["input"][1]["arguments"] == '{"query":"x"}'
    assert value["max_output_tokens"] == 128000 and value["tools"] == original["tools"]


def test_inline_tool_declarations_are_preserved_without_changing_history():
    body = payload()
    declaration = {
        "type": "additional_tools",
        "id": "cli-tools",
        "role": "developer",
        "tools": [{"type": "namespace", "name": "functions", "tools": []}],
    }
    body["input"].insert(0, declaration)
    original = copy.deepcopy(body)
    value = prepare_payload(body, "test-model", "max", 128000)
    assert body == original
    assert value["input"] == original["input"]
    assert value["tools"] == original["tools"]


@pytest.mark.parametrize(
    "field,value,status",
    [
        ("model", "other", "model_mismatch"),
        ("reasoning", {"effort": "high"}, "budget_mismatch"),
        ("stream", False, "protocol_error"),
        (
            "input",
            [{"type": "additional_tools", "role": "user", "tools": []}],
            "protocol_error",
        ),
        (
            "input",
            [{"type": "additional_tools", "role": "developer", "tools": "bad"}],
            "protocol_error",
        ),
        (
            "input",
            [
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [],
                    "content": 5,
                }
            ],
            "protocol_error",
        ),
        (
            "input",
            [{"type": "function_call", "name": "shell"}],
            "protocol_error",
        ),
    ],
)
def test_unexpected_profile_never_forwarded(field, value, status):
    body = payload()
    body[field] = value
    with pytest.raises(ValueError, match=status):
        prepare_payload(body, "test-model", "max", 128000)


def test_namespace_roundtrip_and_unicode_line():
    item = {
        "type": "function_call",
        "name": "mcp__sde_tools__read_url",
        "call_id": "c1",
        "arguments": '{"url":"中文"}',
    }
    original = copy.deepcopy(item)
    line = mapped_line(b"data: " + json.dumps({"item": item}).encode())
    value = json.loads(line[5:])["item"]
    assert value == original
    assert value == original


@pytest.mark.parametrize(
    "kind", ["function_call", "custom_tool_call", "web_search_call"]
)
def test_native_tool_event_preserved_before_cli_receives_it(kind):
    event = {
        "type": "response.output_item.added",
        "item": {"type": kind, "name": "apply_patch"},
    }
    raw = b"data: " + json.dumps(event).encode()
    assert mapped_line(raw) == raw


@pytest.mark.parametrize("name,namespace", [(None, "mcp__sde_tools"), ("run_shell", 1)])
def test_malformed_function_name_fails_closed(name, namespace):
    item = {"type": "function_call", "name": name, "namespace": namespace}
    with pytest.raises(ToolProfileError):
        mapped_line(b"data: " + json.dumps({"item": item}).encode())


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


async def ws_response(channel):
    events: list[dict[str, Any]] = []
    while True:
        frame = await channel.receive(timeout=5)
        if frame.type != aiohttp.WSMsgType.TEXT:
            return events
        event = json.loads(frame.data)
        events.append(event)
        if event["type"] in {
            "response.completed",
            "response.incomplete",
            "response.failed",
            "error",
        }:
            return events


def test_websocket_tool_history_materialized_and_recorded(tmp_path):
    async def run():
        seen = []

        async def upstream(req):
            seen.append(await req.json())
            body = (
                stream(function="mcp__sde_tools__web_search")
                if len(seen) == 1
                else stream().replace('"r1"', '"r2"')
            )
            return web.Response(text=body, content_type="text/event-stream")

        async with (
            local_provider(upstream) as endpoint,
            aiohttp.ClientSession() as session,
            ResponsesProxy(
                session, endpoint, "test-model", "max", 128000, "fake", tmp_path, []
            ) as proxy,
        ):
            async with session.ws_connect(
                proxy.url + "/" + proxy.token + "/responses",
                headers={"Authorization": "Bearer " + proxy.token},
            ) as channel:
                prewarm = {
                    **payload(),
                    "type": "response.create",
                    "generate": False,
                }
                await channel.send_json(prewarm)
                events = await ws_response(channel)
                assert not seen and not proxy.calls
                assert events[-1]["response"]["output"] == []
                assert proxy.controls[0]["upstream_forwarded"] is False
                saved = json.loads((tmp_path / "control_001.json").read_text())
                assert saved["original_payload"] == prewarm
                first = {**payload(), "type": "response.create"}
                first.pop("stream")
                await channel.send_json(first)
                events = await ws_response(channel)
                call = events[-1]["response"]["output"][0]
                assert call["name"] == "mcp__sde_tools__web_search"
                second = {
                    **first,
                    "previous_response_id": "r1",
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": "c1",
                            "output": "5",
                        }
                    ],
                }
                await channel.send_json(second)
                events = await ws_response(channel)
                assert events[-1]["response"]["id"] == "r2"
            assert [c["status"] for c in proxy.calls] == ["ok", "ok"]
            assert proxy.calls[1]["original_payload"] == second
            assert (
                "materialized_previous_response_history"
                in proxy.calls[1]["normalizations"]
            )
            assert all(c["cli_transport"] == "websocket" for c in proxy.calls)
            assert seen[1]["input"][0] == payload()["input"][0]
            assert seen[1]["input"][1]["name"] == "mcp__sde_tools__web_search"
            assert seen[1]["input"][-1] == second["input"][0]
            assert "previous_response_id" not in seen[1]
            assert all(
                p["max_output_tokens"] == 128000 and p["tools"] == payload()["tools"]
                for p in seen
            )

    asyncio.run(run())


def test_websocket_unknown_previous_response_not_forwarded(tmp_path):
    async def run():
        async def upstream(req):
            pytest.fail("Unknown response ID reached provider")

        async with (
            local_provider(upstream) as endpoint,
            aiohttp.ClientSession() as session,
            ResponsesProxy(
                session, endpoint, "test-model", "max", 128000, "fake", tmp_path, []
            ) as proxy,
        ):
            async with session.ws_connect(
                proxy.url + "/" + proxy.token + "/responses",
                headers={"Authorization": "Bearer " + proxy.token},
            ) as channel:
                await channel.send_json(
                    {
                        **payload(),
                        "type": "response.create",
                        "previous_response_id": "unknown",
                    }
                )
                assert (await ws_response(channel))[-1]["type"] == "error"
            assert (
                proxy.failure.is_set() and proxy.calls[0]["status"] == "protocol_error"
            )

    asyncio.run(run())


@pytest.mark.parametrize(
    "status,body,expected",
    [
        (401, "unauthorized", "http_error"),
        (429, "usage_limit_reached", "quota_exhausted"),
        (524, "timeout", "transport_failed"),
    ],
)
def test_websocket_failure_saved_and_nested_retry_blocked(
    tmp_path, status, body, expected
):
    async def run():
        seen = []

        async def upstream(req):
            seen.append(await req.json())
            return web.Response(status=status, text=body)

        async with (
            local_provider(upstream) as endpoint,
            aiohttp.ClientSession() as session,
            ResponsesProxy(
                session,
                endpoint,
                "test-model",
                "max",
                128000,
                "fake",
                tmp_path,
                [],
                auth_mode="chatgpt",
            ) as proxy,
        ):
            url = proxy.url + "/" + proxy.token + "/responses"
            headers = {"Authorization": "Bearer fake-access"}
            async with session.ws_connect(url, headers=headers) as channel:
                await channel.send_json({**payload(), "type": "response.create"})
                assert (await ws_response(channel))[-1]["type"] == "error"
            async with session.post(url, headers=headers, json=payload()) as response:
                assert response.status == 409
            assert len(seen) == 1 and proxy.calls[0]["status"] == expected
            assert proxy.calls[0]["http_status"] == status
            saved = json.loads((tmp_path / "api_001.json").read_text())
            assert saved["status"] == expected
            assert "fake-access" not in json.dumps(saved)

    asyncio.run(run())


@pytest.mark.parametrize("websocket", [False, True])
def test_proxy_preserves_native_tool_before_output_is_forwarded(tmp_path, websocket):
    async def run():
        async def upstream(req):
            return web.Response(
                text=stream(function="apply_patch"), content_type="text/event-stream"
            )

        async with (
            local_provider(upstream) as endpoint,
            aiohttp.ClientSession() as session,
            ResponsesProxy(
                session, endpoint, "test-model", "max", 128000, "fake", tmp_path, []
            ) as proxy,
        ):
            url = proxy.url + "/" + proxy.token + "/responses"
            headers = {"Authorization": "Bearer " + proxy.token}
            if websocket:
                async with session.ws_connect(url, headers=headers) as channel:
                    await channel.send_json({**payload(), "type": "response.create"})
                    received = json.dumps(await ws_response(channel))
            else:
                async with session.post(
                    url, headers=headers, json=payload()
                ) as response:
                    received = await response.text()
            assert "apply_patch" in received
            assert proxy.calls[0]["status"] == "ok"
            assert "apply_patch" in (tmp_path / "api_001.sse").read_text()

    asyncio.run(run())


@pytest.fixture
def model(tmp_path, monkeypatch):
    for name in ("DEEPSEEK_API_KEY", "SERPAPI_API_KEY", "JINA_API_KEY"):
        monkeypatch.setenv(name, "private-" + name)
    monkeypatch.setattr(
        cx,
        "validate_runtime",
        lambda *args: {"cli_version": "codex-cli 0.test", "compute_packages": {}},
    )
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "slug": "test-model",
                        "supported_reasoning_levels": [{"effort": "max"}],
                        "base_instructions": "Frozen test instructions",
                    }
                ]
            }
        )
    )

    def make(**kwargs):
        args: dict[str, Any] = dict(
            model="test-model",
            max_tokens=128000,
            record_dir=str(tmp_path / "records"),
            codex_bin=sys.executable,
            cli_version="0.test",
            tool_python=sys.executable,
            model_catalog=str(catalog),
            base_url="http://127.0.0.1:1/responses",
        )
        args.update(kwargs)
        if args["model"] == "test-model" and args.get("auth_mode") != "chatgpt":
            args.setdefault("search_backend", "native_responses")
        return cx.SDECodex(**args)

    return make


def request(doc=0, answer="A"):
    return Instance(
        request_type="generate_until",
        doc={"answer": answer},
        arguments=(
            json.dumps([{"role": "user", "content": "Question " + str(doc)}]),
            {},
        ),
        idx=0,
        metadata=("test-task", doc, 1),
    )


def authfile(path, account="account1"):
    path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": "private-access",
                    "refresh_token": "private-refresh",
                    "id_token": "private-id",
                    "account_id": account,
                },
            }
        )
    )
    return path


def test_api_profile_does_not_read_personal_auth(model, monkeypatch):
    monkeypatch.setattr(
        cx,
        "read_subscription",
        lambda path: pytest.fail("API path read subscription auth"),
    )
    m = model()
    assert m.get_api_key() == "private-DEEPSEEK_API_KEY"


def test_explicit_resume_preserves_agent_attempt_directories(model, monkeypatch):
    calls = []

    async def request_once(session, payload, key, attempt):
        calls.append(attempt["attempt"])
        directory = Path(attempt["record_path"]).with_suffix(
            f".attempt{attempt['attempt']:02}.codex"
        )
        directory.mkdir()
        (directory / "trace.jsonl").write_text(f"trace {attempt['attempt']}")
        attempt.update(
            status="transport_failed" if len(calls) == 1 else "ok",
            answer="<answer>A</answer>",
            response={"usage": {"output_tokens": 7}},
        )

    first_model = model(max_attempts=1)
    monkeypatch.setattr(first_model, "request_once", request_once)
    first = first_model.generate_until([request()])[0]
    path = Path(first.metadata["record_path"])
    before = json.loads(path.read_text())
    resumed = model(resume=True, timeout=120, max_attempts=1)
    monkeypatch.setattr(resumed, "request_once", request_once)
    after = resumed.generate_until([request()])[0]
    assert after.metadata["status"] == "ok"
    assert after.metadata["record_path"] == str(path)
    assert json.loads(path.read_text())["attempts"][:1] == before["attempts"]
    assert (
        path.with_suffix(".attempt01.codex").joinpath("trace.jsonl").read_text()
        == "trace 1"
    )
    assert (
        path.with_suffix(".attempt02.codex").joinpath("trace.jsonl").read_text()
        == "trace 2"
    )
    assert calls == [1, 2]


def test_native_tools_do_not_require_third_party_search_keys(model, monkeypatch):
    monkeypatch.delenv("SERPAPI_API_KEY")
    monkeypatch.delenv("JINA_API_KEY")
    m = model()
    assert m.config["tools_policy"] == "all_available"
    assert m.config["search_attempt_limit"] is None
    assert m.config["native_tools"] is True


@pytest.mark.parametrize(
    "item",
    [
        {"type": "command_execution", "command": "python -c 'print(5)'"},
        {"type": "web_search", "query": "Python documentation"},
        {"type": "file_change", "changes": [{"path": "result.txt"}]},
        {"type": "mcp_tool_call", "server": "provider", "tool": "search"},
        {"type": "collab_agent_tool_call", "tool": "spawn_agent"},
    ],
)
def test_native_tool_trace_keeps_complete_answer(item):
    result = cx.summarize_trace(trace(tool=item), "<answer>A</answer>", 0)
    assert result["status"] == "ok"
    assert result["tool_calls"] == [item]


def test_native_cli_command_opens_tools_without_loading_personal_settings(
    model, tmp_path
):
    m = model()
    proxy = ResponsesProxy(
        None, "http://127.0.0.1:1", "test-model", "max", 128000, "fake", tmp_path, []
    )
    proxy.url = "http://127.0.0.1:2"
    command = m.command(tmp_path, "prompt", proxy, tmp_path)
    cfg = {
        command[i + 1].split("=", 1)[0]: json.loads(command[i + 1].split("=", 1)[1])
        for i, arg in enumerate(command[:-1])
        if arg == "-c"
    }
    assert cfg["web_search"] == "live"
    assert cfg["agents.enabled"] is True
    assert "--dangerously-bypass-approvals-and-sandbox" in command
    assert "--sandbox" not in command
    assert not any("mcp_servers.sde_tools" in arg for arg in command)
    disabled = {
        command[i + 1] for i, arg in enumerate(command[:-1]) if arg == "--disable"
    }
    assert not disabled & {"shell_tool", "unified_exec", "code_mode", "multi_agent"}
    assert "--ignore-user-config" in command


def test_final_api_answer_is_attributed_to_root_not_last_child():
    calls = [
        {
            "index": 1,
            "status": "ok",
            "answer": "parent answer",
            "original_payload": {"client_metadata": {"thread_id": "parent"}},
        },
        {
            "index": 2,
            "status": "ok",
            "answer": "child answer",
            "original_payload": {"client_metadata": {"thread_id": "child"}},
        },
    ]
    call, attribution = cx.final_api_call(calls, "parent", "parent answer")
    assert call is calls[0] and attribution == "thread_id"
    assert cx.final_api_call(calls, "parent", "child answer")[0] is None


def test_cli_cwd_matches_temporary_workspace(model, tmp_path):
    m = model()
    home = tmp_path / "temporary-home"
    work = home / "work"
    work.mkdir(parents=True)
    records = tmp_path / "retained-results"
    records.mkdir()
    proxy = ResponsesProxy(
        None, "http://127.0.0.1:1", "test-model", "max", 128000, "fake", records, []
    )
    proxy.url = "http://127.0.0.1:2"
    command = m.command(records, "prompt", proxy, home)
    assert Path(command[command.index("-C") + 1]) == work
    assert Path(command[command.index("-C") + 1]).is_dir()


def test_flash_uses_provider_native_search_bridge_without_external_search_keys(
    model, tmp_path, monkeypatch
):
    catalog = tmp_path / "catalog.json"
    body = json.loads(catalog.read_text())
    body["models"][0]["slug"] = "deepseek-flash"
    catalog.write_text(json.dumps(body))
    monkeypatch.delenv("SERPAPI_API_KEY")
    monkeypatch.delenv("JINA_API_KEY")
    m = model(model="deepseek-flash")
    proxy = ResponsesProxy(
        None,
        "http://127.0.0.1:1",
        "deepseek-flash",
        "max",
        128000,
        "fake",
        tmp_path,
        [],
    )
    proxy.url = "http://127.0.0.1:2"
    command = m.command(tmp_path, "prompt", proxy, tmp_path)
    cfg = {
        command[i + 1].split("=", 1)[0]: json.loads(command[i + 1].split("=", 1)[1])
        for i, arg in enumerate(command[:-1])
        if arg == "-c"
    }
    assert cfg["web_search"] == "disabled"
    assert cfg["mcp_servers.deepseek_search.args"][0].endswith("sde_deepseek_search.py")
    assert cfg["mcp_servers.deepseek_search.env_vars"] == [
        "SDE_SEARCH_PROXY_URL",
        "SDE_PROXY_TOKEN",
    ]
    assert m.config["search_backend"] == "deepseek_anthropic_bridge"
    assert m.config["search_attempt_limit"] is None
    assert m.config["agent_turn_limit"] is None
    assert "private-DEEPSEEK_API_KEY" not in json.dumps(cfg)


def test_unattributed_api_answer_is_recorded_without_claiming_root_identity():
    calls = [{"index": 1, "status": "ok", "answer": "final"}]
    call, attribution = cx.final_api_call(calls, "root", "final")
    assert call is calls[0] and attribution == "answer_match_without_thread_identity"
    assert cx.final_api_call(calls, "root", "fabricated")[0] is None


def test_subscription_auth_requires_explicit_source_and_official_endpoint(
    model, tmp_path, monkeypatch
):
    with pytest.raises(ValueError, match="explicit auth_file"):
        model(auth_mode="chatgpt", base_url=cx.SUBSCRIPTION_ENDPOINT)
    auth = authfile(tmp_path / "auth.json")
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    m = model(
        auth_mode="chatgpt", auth_file=str(auth), base_url=cx.SUBSCRIPTION_ENDPOINT
    )
    assert m.get_api_key() == "" and m.account_identity
    assert "private-access" not in json.dumps(m.config)
    with pytest.raises(ValueError, match="subscription endpoint"):
        model(auth_mode="chatgpt", auth_file=str(auth))
    with pytest.raises(ValueError, match="must not read"):
        model(auth_file=str(auth))


def test_subscription_rejects_account_change_and_api_auth(tmp_path):
    path = authfile(tmp_path / "auth.json")
    _, identity, secrets = cx.read_subscription(path)
    assert "private-refresh" in secrets
    authfile(path, account="different")
    assert cx.read_subscription(path)[1] != identity
    path.write_text('{"auth_mode":"api_key","OPENAI_API_KEY":"fake"}')
    with pytest.raises(ValueError, match="ChatGPT subscription"):
        cx.read_subscription(path)


def test_subscription_account_change_rejected_by_adapter(model, tmp_path):
    auth = authfile(tmp_path / "auth.json")
    m = model(
        auth_mode="chatgpt", auth_file=str(auth), base_url=cx.SUBSCRIPTION_ENDPOINT
    )
    authfile(auth, account="changed")
    with pytest.raises(ValueError, match="account changed"):
        m.validate_auth("unused")


def test_code_mode_only_catalog_preserves_native_tools_and_instructions(
    model, tmp_path
):
    path = tmp_path / "catalog.json"
    original = json.loads(path.read_text())
    original["models"][0]["tool_mode"] = "code_mode_only"
    path.write_text(json.dumps(original))
    m = model()
    assert json.loads(path.read_text()) == original
    assert m.catalog["models"][0]["tool_mode"] == "code_mode_only"
    assert (
        m.catalog["models"][0]["base_instructions"]
        == original["models"][0]["base_instructions"]
    )
    assert m.config["pinned_tool_mode"] == "code_mode_only"
    assert m.config["catalog_normalizations"] == []


def test_native_model_messages_are_passed_to_cli_without_legacy_override(
    model, tmp_path
):
    path = tmp_path / "catalog.json"
    original = json.loads(path.read_text())
    entry = original["models"][0]
    entry.pop("base_instructions")
    entry["model_messages"] = {
        "instructions_template": "Synthetic native template",
        "persistent_instructions": "Synthetic optional mode",
    }
    path.write_text(json.dumps(original))
    m = model()
    proxy = ResponsesProxy(
        None, "http://127.0.0.1:1", "test-model", "max", 128000, "fake", tmp_path, []
    )
    proxy.url = "http://127.0.0.1:2"
    m.command(tmp_path, "evaluation instructions", proxy, tmp_path)
    saved = json.loads(path.read_text())
    assert saved == original
    assert "base_instructions" not in m.catalog["models"][0]
    assert m.catalog["models"][0]["model_messages"] == entry["model_messages"]


def test_prompt_and_effective_command(model, tmp_path):
    m = model()
    r = request()
    r.arguments = (
        json.dumps(
            [
                {"role": "system", "content": "extra instruction"},
                {"role": "user", "content": "verbatim question"},
            ]
        ),
        {"max_tokens": 128000},
    )
    body = m.create_payload(r)
    assert body["prompt"] == "verbatim question" and body["max_tokens"] == 128000
    assert body["system_append"].endswith("extra instruction")
    proxy = ResponsesProxy(
        None, "http://127.0.0.1:1", "test-model", "max", 128000, "fake", tmp_path, []
    )
    proxy.url = "http://127.0.0.1:2"
    cmd = m.command(
        tmp_path,
        body["system_append"],
        proxy,
        tmp_path / "home",
    )
    assert cmd[-1] == "-" and "--ignore-user-config" in cmd and "--ignore-rules" in cmd
    assert "--ephemeral" in cmd and "--dangerously-bypass-approvals-and-sandbox" in cmd
    assert "hooks" in cmd and "unbounded_connection_retries" in cmd
    assert "shell_tool" not in cmd
    cfg = {
        cmd[i + 1].split("=", 1)[0]: cmd[i + 1].split("=", 1)[1]
        for i, arg in enumerate(cmd[:-1])
        if arg == "-c"
    }
    instructions = json.loads(cfg["developer_instructions"])
    assert instructions == body["system_append"]
    assert (
        "all tool processes and threads share a total budget of "
        "8 vCPUs and 16 GiB of RAM."
    ) in instructions
    assert not any("mcp_servers.sde_tools" in key for key in cfg)
    assert cfg["web_search"] == '"live"'
    assert cfg["project_doc_max_bytes"] == "0"
    assert cfg["model_providers.sde_recorded.request_max_retries"] == "0"
    assert not any("max_turn" in arg or "fallback_model" in arg for arg in cmd)


def test_fewshot_role_history_recorded(model):
    m = model()
    r = request()
    messages = [
        {"role": "user", "content": "example"},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": "question"},
    ]
    r.arguments = (json.dumps(messages), {})
    body = m.create_payload(r)
    assert body["input_messages"] == messages and '"assistant"' in body["prompt"]


def install_response(m, monkeypatch, responses):
    count: list[dict[str, Any]] = []

    async def once(session, body, key, attempt):
        status, answer = responses[min(len(count), len(responses) - 1)]
        count.append(attempt)
        attempt.update(
            status=status,
            answer=answer,
            response={"model": "test-model", "usage": {"output_tokens": 4}},
        )

    monkeypatch.setattr(m, "request_once", once)
    real_sleep = asyncio.sleep

    async def sleep(seconds):
        await real_sleep(0 if seconds >= 1 else seconds)

    monkeypatch.setattr(cx.asyncio, "sleep", sleep)
    return count


@pytest.mark.parametrize(
    "status",
    [
        "refusal",
        "truncated",
        "context_limit",
        "budget_mismatch",
        "tool_profile_mismatch",
    ],
)
def test_terminal_statuses_not_retried(model, monkeypatch, status):
    m = model()
    count = install_response(m, monkeypatch, [(status, "partial")])
    result = m.generate_until([request()])[0]
    assert result.text == "" and result.metadata["status"] == status and len(count) == 1


def test_retries_independent_draws_and_resume(model, monkeypatch):
    m = model()
    count = install_response(
        m, monkeypatch, [("transport_failed", ""), ("ok", "wrong")]
    )
    r = request()
    values = m.generate_until([r, r])
    assert [v.metadata["repeat_index"] for v in values] == [0, 1]
    assert all(v.text == "wrong" for v in values)
    before = len(count)
    m.generate_until([r, r])
    assert len(count) == before


def test_quota_exhaustion_saved_and_batch_stopped(model, monkeypatch):
    m = model()
    count = install_response(m, monkeypatch, [("quota_exhausted", "")])
    with pytest.raises(RuntimeError, match="quota exhausted"):
        m.generate_until([request()])
    saved = list(m.record_dir.glob("*.json"))
    assert (
        len(saved) == 1
        and json.loads(saved[0].read_text())["status"] == "quota_exhausted"
    )
    with pytest.raises(RuntimeError):
        m.generate_until([request()])
    assert len(count) == 1


def test_explicit_resume_reopens_quota_and_keeps_normal_answer(model, monkeypatch):
    m = model(num_concurrent=1, max_attempts=1)
    install_response(m, monkeypatch, [("ok", "saved answer"), ("quota_exhausted", "")])
    requests = [request(doc=0), request(doc=1)]
    with pytest.raises(RuntimeError, match="quota exhausted"):
        m.generate_until(requests)
    records = {
        path: json.loads(path.read_text()) for path in m.record_dir.glob("*.json")
    }
    normal_path = next(
        path for path, record in records.items() if record["status"] == "ok"
    )
    quota_path = next(
        path
        for path, record in records.items()
        if record["status"] == "quota_exhausted"
    )
    normal_before = normal_path.read_bytes()
    quota_before = records[quota_path]["attempts"]
    resumed = model(resume=True, num_concurrent=1, max_attempts=1)
    calls = install_response(resumed, monkeypatch, [("ok", "new answer")])
    results = resumed.generate_until(requests)
    assert [result.text for result in results] == ["saved answer", "new answer"]
    assert len(calls) == 1
    assert normal_path.read_bytes() == normal_before
    repaired = json.loads(quota_path.read_text())
    assert repaired["attempts"][:-1] == quota_before
    assert [attempt["status"] for attempt in repaired["attempts"]] == [
        "quota_exhausted",
        "ok",
    ]
    assert repaired["deadline_windows"][-1]["attempt_start"] == len(quota_before)
    assert repaired["finished"] is True


@pytest.mark.parametrize(
    "runtime_sha256",
    [
        "f9cdac7e3d3c6a65fd503c18a6a1998fbf1816f78bac77da7a83084af291daad",
        "ef22aedfa4e9f6233429740cfc74f06724c6aedd6b9824816315ce4d22a14221",
    ],
)
def test_resume_keeps_records_from_before_quota_fix(model, monkeypatch, runtime_sha256):
    m = model()
    install_response(m, monkeypatch, [("ok", "saved answer")])
    result = m.generate_until([request()])[0]
    path = Path(result.metadata["record_path"])
    record = json.loads(path.read_text())
    record["identity"]["config"]["runtime_sha256"] = runtime_sha256
    path.write_text(json.dumps(record))
    before = path.read_bytes()
    resumed = model(resume=True)
    calls = install_response(resumed, monkeypatch, [("ok", "replacement")])
    again = resumed.generate_until([request()])[0]
    assert again.text == "saved answer"
    assert again.metadata["record_path"] == str(path)
    assert path.read_bytes() == before
    assert calls == []


@pytest.mark.parametrize("code", [401, 403])
def test_auth_failure_is_saved_and_stops_batch(model, monkeypatch, code):
    m = model()

    async def once(session, body, key, attempt):
        attempt.update(status="http_error", http_status=code)

    monkeypatch.setattr(m, "request_once", once)
    with pytest.raises(PermissionError):
        m.generate_until([request()])
    saved = list(m.record_dir.glob("*.json"))
    assert len(saved) == 1
    assert json.loads(saved[0].read_text())["attempts"][0]["http_status"] == code
    with pytest.raises(PermissionError, match="Saved authentication"):
        m.generate_until([request()])


def test_original_filters_and_metrics_keep_string_input(model, monkeypatch):
    m = model()
    install_response(m, monkeypatch, [("ok", "<answer>A</answer>")])
    data = Dataset.from_list([{"question": str(i), "answer": "A"} for i in range(10)])

    class ToyTask(ConfigurableTask):
        def download(self, *args, **kwargs):
            self.dataset = {"test": data}

    task = ToyTask(
        config={
            "task": "tiny",
            "dataset_path": "unused",
            "output_type": "generate_until",
            "test_split": "test",
            "doc_to_text": "question",
            "doc_to_target": "answer",
            "num_fewshot": 0,
            "generation_kwargs": {"max_tokens": 128000, "until": []},
            "filter_list": [
                {
                    "name": "xml",
                    "filter": [
                        {
                            "function": "regex",
                            "regex_pattern": "<answer>(.*?)</answer>",
                        },
                        {"function": "take_first"},
                    ],
                }
            ],
            "metric_list": [
                {
                    "metric": "exact_match",
                    "aggregation": "mean",
                    "higher_is_better": True,
                }
            ],
        },
    )
    result = evaluate(
        m, {"tiny": task}, apply_chat_template=True, log_samples=True, bootstrap_iters=0
    )
    assert result["results"]["tiny"]["exact_match,xml"] == 1
    assert len(result["samples"]["tiny"]) == 10
    assert all(
        row["response_metadata"][0][0]["status"] == "ok"
        for row in result["samples"]["tiny"]
    )


def test_proxy_loopback_records_original_and_effective_requests(tmp_path):
    async def run():
        seen = []

        async def upstream(req):
            seen.append(await req.json())
            return web.Response(text=stream(), content_type="text/event-stream")

        app = web.Application()
        app.router.add_post("/responses", upstream)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        try:
            async with (
                aiohttp.ClientSession() as session,
                ResponsesProxy(
                    session,
                    f"http://127.0.0.1:{runner.addresses[0][1]}/responses",
                    "test-model",
                    "max",
                    128000,
                    "private-provider",
                    tmp_path,
                    [],
                ) as proxy,
            ):
                async with session.post(
                    proxy.url + "/" + proxy.token + "/responses",
                    json=payload(),
                    headers={"Authorization": "Bearer " + proxy.token},
                ) as response:
                    assert response.status == 200
                    await response.text()
                assert proxy.calls[0]["status"] == "ok"
                assert (
                    proxy.calls[0]["original_payload"]["tools"][0]["name"]
                    == "apply_patch"
                )
                assert (
                    seen[0]["tools"] == payload()["tools"]
                    and seen[0]["max_output_tokens"] == 128000
                )
                assert seen[0]["input"] == payload()["input"]
        finally:
            await runner.cleanup()

    asyncio.run(run())


def test_proxy_failure_blocks_nested_retry_and_redacts_fragmented_secrets(tmp_path):
    async def run():
        seen = []

        async def upstream(req):
            seen.append(await req.json())
            response = web.StreamResponse(status=524)
            await response.prepare(req)
            for part in ("错误 private-", "provider\n"):
                await response.write(part.encode())
            await response.write_eof()
            return response

        app = web.Application()
        app.router.add_post("/responses", upstream)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        try:
            async with (
                aiohttp.ClientSession() as session,
                ResponsesProxy(
                    session,
                    f"http://127.0.0.1:{runner.addresses[0][1]}/responses",
                    "test-model",
                    "max",
                    128000,
                    "private-provider",
                    tmp_path,
                    [],
                ) as proxy,
            ):
                for expected in (524, 409):
                    async with session.post(
                        proxy.url + "/" + proxy.token + "/responses",
                        json=payload(),
                        headers={"Authorization": "Bearer " + proxy.token},
                    ) as response:
                        assert response.status == expected
                        await response.text()
                assert len(seen) == 1 and proxy.failure.is_set()
                assert "private-provider" not in (tmp_path / "api_001.sse").read_text()
                assert "错误 <redacted>" in (tmp_path / "api_001.sse").read_text()
        finally:
            await runner.cleanup()

    asyncio.run(run())


def fake_cli(path, *, hang=False, subscription=False):
    code = """#!/usr/bin/env python3
import json, os, sys, urllib.request, time
from pathlib import Path
cfg={}
for i,arg in enumerate(sys.argv[:-1]):
 if arg=='-c':
  k,v=sys.argv[i+1].split('=',1); cfg[k]=json.loads(v)
home=Path(os.environ['CODEX_HOME'])
token=os.environ['SDE_PROXY_TOKEN']
if SUBSCRIPTION:
 assert (home/'auth.json').stat().st_mode & 0o777 == 0o600
 token=json.loads((home/'auth.json').read_text())['tokens']['access_token']
 assert 'DEEPSEEK_API_KEY' not in os.environ
else:
 assert not (home/'auth.json').exists()
url=(cfg['openai_base_url'] if SUBSCRIPTION else cfg['model_providers.'+cfg['model_provider']+'.base_url'])+'/responses'
prompt=sys.stdin.read()
print(json.dumps({'type':'thread.started','thread_id':'t1'}),flush=True)
print(json.dumps({'type':'turn.started'}),flush=True)
if HANG:
 print('partial trace',file=sys.stderr,flush=True);time.sleep(10)
body={'model':'test-model','stream':True,'reasoning':{'effort':'max'},'instructions':'actual CLI system','input':[{'type':'message','role':'user','content':prompt}],'tools':[{'type':'custom','name':'apply_patch'}]}
req=urllib.request.Request(url,json.dumps(body).encode(),headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
with urllib.request.urlopen(req) as response: response.read()
answer='<answer>A</answer>'
Path(sys.argv[sys.argv.index('-o')+1]).write_text(answer+'\\n')
print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':answer}}),flush=True)
print(json.dumps({'type':'turn.completed','usage':{'output_tokens':4}}),flush=True)
""".replace("SUBSCRIPTION", str(subscription)).replace("HANG", str(hang))
    path.write_text(code)
    path.chmod(0o700)
    return path


@pytest.mark.parametrize("subscription", [False, True])
def test_real_subprocess_to_loopback_proxy_with_isolated_auth(
    model, tmp_path, subscription
):
    cli = fake_cli(tmp_path / "fake-codex", subscription=subscription)
    auth = authfile(tmp_path / "auth.json")
    before = auth.read_bytes()
    m = model(
        codex_bin=str(cli),
        **(
            {
                "auth_mode": "chatgpt",
                "auth_file": str(auth),
                "base_url": cx.SUBSCRIPTION_ENDPOINT,
            }
            if subscription
            else {}
        ),
    )
    homes = []
    original = m.command

    def command(*args):
        homes.append(args[-1])
        return original(*args)

    m.command = command

    async def run():
        async def upstream(req):
            assert (await req.json())["tools"] == payload()["tools"]
            assert req.headers["Authorization"] == "Bearer " + (
                "private-access" if subscription else "private-DEEPSEEK_API_KEY"
            )
            return web.Response(text=stream(), content_type="text/event-stream")

        app = web.Application()
        app.router.add_post("/responses", upstream)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        m.config["base_url"] = f"http://127.0.0.1:{runner.addresses[0][1]}/responses"
        try:
            m.record_dir.mkdir()
            return await m.generate_batch([request()], [m.create_payload(request())])
        finally:
            await runner.cleanup()

    values = asyncio.run(run())
    assert (
        values[0].text == "<answer>A</answer>" and values[0].metadata["status"] == "ok"
    )
    assert homes and all(not home.exists() for home in homes)
    assert auth.read_bytes() == before
    for path in m.record_dir.rglob("*"):
        if path.is_file():
            assert "private-access" not in path.read_text()
            assert "private-refresh" not in path.read_text()


def test_timeout_terminates_subprocess_and_retains_partial_trace(model, tmp_path):
    cli = fake_cli(tmp_path / "fake-codex", hang=True)
    # Leave time for the fake Python CLI to start before testing termination.
    m = model(codex_bin=str(cli), timeout=3.0)
    result = m.generate_until([request()])[0]
    assert result.metadata["status"] == "question_timeout"
    path = Path(result.metadata["record_path"])
    raw = json.loads(path.read_text())
    pid = raw["attempts"][0]["pid"]
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert "partial trace" in Path(raw["attempts"][0]["stderr_path"]).read_text()


def test_responses_proxy_retries_503_before_stream(tmp_path, monkeypatch):
    from lm_eval.models import sde_cc_proxy

    monkeypatch.setattr(sde_cc_proxy, "retry_delay", lambda *args: 0)

    async def run():
        seen = []

        async def upstream(req):
            seen.append(await req.json())
            if len(seen) == 1:
                return web.json_response({"error": "unavailable"}, status=503)
            return web.Response(text=stream(), content_type="text/event-stream")

        async with (
            local_provider(upstream) as endpoint,
            aiohttp.ClientSession() as session,
            ResponsesProxy(
                session, endpoint, "test-model", "max", 128000, "fake", tmp_path, []
            ) as proxy,
        ):
            async with session.post(
                proxy.url + "/" + proxy.token + "/responses",
                headers={"Authorization": "Bearer " + proxy.token},
                json=payload(),
            ) as response:
                assert response.status == 200
                await response.read()
            call = proxy.calls[0]
            assert call["status"] == "ok"
            assert call["upstream_retries"][0]["http_status"] == 503
            assert len(seen) == 2 and seen[0] == seen[1]

    asyncio.run(run())


def test_unknown_api_model_requires_explicit_search_backend(model, tmp_path):
    with pytest.raises(ValueError, match="search_backend"):
        model(search_backend=None)
    with pytest.raises(ValueError, match="search_backend"):
        model(search_backend="serpapi")
    with pytest.raises(ValueError, match="pinned to deepseek-flash"):
        model(search_backend="deepseek_anthropic_bridge")
    hosted = model(search_backend="native_responses_hosted")
    assert hosted.hosted_web_search and not hosted.deepseek_search
    assert hosted.config["search_backend"] == "native_responses_hosted"
    auth = authfile(tmp_path / "auth.json")
    with pytest.raises(ValueError, match="native_subscription_web"):
        model(
            auth_mode="chatgpt",
            auth_file=str(auth),
            base_url=cx.SUBSCRIPTION_ENDPOINT,
            search_backend="native_responses",
        )


def test_auth_profile_mismatch_stops_batch(model, monkeypatch):
    m = model()
    count = install_response(m, monkeypatch, [("auth_profile_mismatch", "")])
    with pytest.raises(RuntimeError, match="credentials changed"):
        m.generate_until([request()])
    assert len(count) == 1


def test_changed_subscription_file_is_an_auth_mismatch(model, tmp_path):
    auth = authfile(tmp_path / "auth.json")
    m = model(
        auth_mode="chatgpt", auth_file=str(auth), base_url=cx.SUBSCRIPTION_ENDPOINT
    )
    m.record_dir.mkdir(parents=True)
    authfile(auth, account="changed")
    attempt = {
        "record_path": str(m.record_dir / "record.json"),
        "attempt": 1,
        "started_at": 0,
        "status": "running",
    }
    asyncio.run(m.request_once(None, m.create_payload(request()), "", attempt))
    assert attempt["status"] == "auth_profile_mismatch"
    assert "account changed" in attempt["error"]


def test_reasoning_summary_is_requested_unless_disabled(model, tmp_path):
    def cfg(m):
        proxy = ResponsesProxy(
            None,
            "http://127.0.0.1:1",
            "test-model",
            "max",
            128000,
            "fake",
            tmp_path,
            [],
        )
        proxy.url = "http://127.0.0.1:2"
        command = m.command(tmp_path, "prompt", proxy, tmp_path)
        return {
            command[i + 1].split("=", 1)[0]: json.loads(command[i + 1].split("=", 1)[1])
            for i, arg in enumerate(command[:-1])
            if arg == "-c"
        }

    default = model()
    assert cfg(default)["model_reasoning_summary"] == "auto"
    assert default.config["reasoning_summary"] == "auto"
    assert "model_reasoning_summary" not in cfg(model(reasoning_summary="none"))
    assert cfg(model(reasoning_summary="detailed"))["model_reasoning_summary"] == (
        "detailed"
    )
    with pytest.raises(ValueError, match="reasoning_summary"):
        model(reasoning_summary="raw")
