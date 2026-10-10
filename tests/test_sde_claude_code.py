import asyncio
import concurrent.futures
import json
import os
import sys
from pathlib import Path

import aiohttp
import pytest
from datasets import Dataset

from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask
from lm_eval.evaluator import evaluate
from lm_eval.models import sde_claude_code as cc
from lm_eval.models import sde_tools as tools
from lm_eval.models.sde_cc_proxy import ALLOWED_TOOLS, MessagesProxy, inspect_stream


def event(kind, **kwargs):
    return {"type": kind, **kwargs}


def trace(
    answer="<answer>A</answer>",
    *,
    tool="mcp__sde_tools__run_shell",
    stop="end_turn",
    final=True,
):
    events = [
        event(
            "system", subtype="init", model="test-model", tools=sorted(ALLOWED_TOOLS)
        ),
        event(
            "assistant",
            message={
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": tool,
                        "input": {"command": "2+3"},
                    }
                ]
            },
        ),
        event(
            "stream_event",
            event={"type": "message_delta", "delta": {"stop_reason": stop}},
        ),
    ]
    if final:
        events.append(
            event(
                "result", subtype="success", result=answer, usage={"output_tokens": 7}
            )
        )
    return "\n".join(json.dumps(e) for e in events) + "\n"


def stream(model="test-model", stop="end_turn", closed=True):
    events = [
        event(
            "message_start",
            message={
                "id": "m1",
                "model": model,
                "role": "assistant",
                "content": [],
                "usage": {"input_tokens": 8},
            },
        ),
        event("message_delta", delta={"stop_reason": stop}, usage={"output_tokens": 3}),
    ]
    if closed:
        events.append(event("message_stop"))
    return "".join("data: " + json.dumps(e) + "\n\n" for e in events)


@pytest.mark.parametrize(
    "answer,stop,status",
    [
        ("wrong answer", "end_turn", "ok"),
        ("", "end_turn", "empty"),
        ("denied", "refusal", "refusal"),
        ("partial", "max_tokens", "truncated"),
        ("partial", "model_context_window_exceeded", "context_limit"),
    ],
)
def test_cli_classification_never_consults_scientific_score(answer, stop, status):
    assert cc.summarize_trace(trace(answer, stop=stop), 0)["status"] == status


@pytest.mark.parametrize(
    "body,status",
    [
        (trace(final=False), "transport_failed"),
        (trace(tool="Bash"), "tool_profile_mismatch"),
        (trace() + "{broken", "protocol_error"),
        (
            trace() + json.dumps(event("result", subtype="success", result="B")),
            "protocol_error",
        ),
    ],
)
def test_incomplete_invalid_and_extra_tool_traces(body, status):
    assert cc.summarize_trace(body, 0)["status"] == status


def test_failure_and_turn_limit_are_not_normal_answers():
    body = trace().replace('"subtype": "success"', '"subtype": "error_max_turns"')
    assert cc.summarize_trace(body, 1)["status"] == "turn_limit"
    assert cc.summarize_trace(trace(), 1)["status"] == "transport_failed"


@pytest.mark.parametrize(
    "body,status",
    [
        (stream(stop="tool_use"), "ok"),
        (stream(closed=False), "transport_failed"),
        (stream(model="other"), "model_mismatch"),
        ("data: {bad}\n\n", "protocol_error"),
    ],
)
def test_proxy_closure_and_model(body, status):
    assert inspect_stream(body, "test-model")["status"] == status


def test_search_quota_is_atomic_and_survives_server_restarts(tmp_path):
    path = tmp_path / "question.json"

    def reserve(_):
        try:
            return tools.reserve_search(path)
        except ValueError:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        values = list(pool.map(reserve, range(20)))
    assert sorted(v for v in values if v is not None) == [1, 2, 3]
    assert json.loads(path.read_text()) == {"backend_attempts": 3}
    with pytest.raises(ValueError, match="exhausted"):
        tools.reserve_search(path)


def test_failed_search_counts_against_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "fake")

    def fail(*args, **kwargs):
        raise ValueError("HTTP 524")

    monkeypatch.setattr(tools, "api_get", fail)
    for _ in range(3):
        with pytest.raises(ValueError, match="524"):
            tools.dispatch(
                tmp_path, tmp_path / "quota.json", "web_search", {"query": "science"}
            )
    with pytest.raises(ValueError, match="exhausted"):
        tools.dispatch(
            tmp_path, tmp_path / "quota.json", "web_search", {"query": "science"}
        )


def test_search_and_jina_payloads_and_bounds(tmp_path, monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "fake-search")
    monkeypatch.setenv("JINA_API_KEY", "fake-read")
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        if "serpapi" in url:
            return json.dumps(
                {
                    "organic_results": [
                        {
                            "title": "a",
                            "link": f"https://example.org/{i}",
                            "snippet": "b",
                        }
                        for i in range(20)
                    ]
                    + [{"link": "https://huggingface.co/answers"}]
                }
            ).encode()
        return b"x" * 40000

    monkeypatch.setattr(tools, "api_get", get)
    monkeypatch.setattr(tools, "public_url", lambda url: url)
    result = tools.dispatch(
        tmp_path, tmp_path / "quota.json", "web_search", {"query": "wave"}
    )
    assert len(result["results"]) == 8
    assert set(result["results"][0]) == {"title", "url", "snippet"}
    assert calls[0][1]["params"]["engine"] == "google"
    assert calls[0][1]["params"]["api_key"] == "fake-search"
    page = tools.dispatch(
        tmp_path, tmp_path / "quota.json", "read_url", {"url": "https://example.org"}
    )
    assert len(page["text"]) == 32000
    assert calls[1][1]["headers"]["Authorization"] == "Bearer fake-read"


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "https://user:pass@example.org",
        "https://huggingface.co/answers",
        "http://localhost",
    ],
)
def test_public_read_excludes_labels_and_private_hosts(url, monkeypatch):
    monkeypatch.setattr(
        tools.socket,
        "getaddrinfo",
        lambda *a, **k: [(None, None, None, None, ("127.0.0.1", 80))],
    )
    with pytest.raises(ValueError):
        tools.public_url(url)


def test_mcp_logs_results_without_keys(tmp_path, monkeypatch, capsys):
    import io

    monkeypatch.setenv("SERPAPI_API_KEY", "fake-secret")
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "web_search", "arguments": {"query": "x"}},
                }
            )
            + "\n"
        ),
    )
    monkeypatch.setattr(tools, "dispatch", lambda *a: {"results": "fake-secret"})
    tools.serve(tmp_path, tmp_path / "quota.json", tmp_path / "tools.jsonl")
    assert "fake-secret" not in capsys.readouterr().out
    assert "fake-secret" not in (tmp_path / "tools.jsonl").read_text()
    assert json.loads((tmp_path / "tools.jsonl").read_text())["result"]["content"]


@pytest.fixture
def model_setup(tmp_path, monkeypatch):
    for name in ("TEST_CC_KEY", "SERPAPI_API_KEY", "JINA_API_KEY"):
        monkeypatch.setenv(name, "test-" + name)
    monkeypatch.setattr(
        cc,
        "validate_runtime",
        lambda *a: {"cli_version": "0.0.test", "tool_python_version": "test"},
    )
    original_sleep = asyncio.sleep

    async def no_sleep(delay):
        await original_sleep(0 if delay >= 1 else delay)

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    def make(**kwargs):
        return cc.SDEClaudeCode(
            model="test-model",
            api_model="test-model",
            max_tokens=128000,
            record_dir=str(tmp_path / "records"),
            claude_bin=sys.executable,
            cli_version="0.0.test",
            tool_python=sys.executable,
            base_url="https://provider.invalid",
            api_key_env="TEST_CC_KEY",
            **kwargs,
        )

    return make


def question(model, doc_id=0, kwargs=None):
    return Instance(
        "generate_until",
        {"question": "2+3?"},
        (
            model.apply_chat_template([{"role": "user", "content": "2+3?"}]),
            kwargs or {},
        ),
        0,
        ("toy", doc_id, 1),
    )


def test_command_disables_native_tools_and_turn_limit(model_setup, tmp_path):
    model = model_setup()
    cmd = model.command(tmp_path, tmp_path / "quota.json", "system")
    assert cmd[cmd.index("--tools") + 1] == ""
    assert set(cmd[cmd.index("--allowedTools") + 1].split(",")) == ALLOWED_TOOLS
    assert "--max-turns" not in cmd and "--fallback-model" not in cmd
    assert "--strict-mcp-config" in cmd and "--restricted" in cmd
    assert cmd[cmd.index("--setting-sources") + 1] == ""
    mcp = json.loads(cmd[cmd.index("--mcp-config") + 1])
    assert set(mcp["mcpServers"]) == {"sde_tools"}
    assert str(tmp_path / "quota.json") in mcp["mcpServers"]["sde_tools"]["args"]
    assert (
        model.create_payload(question(model, kwargs={"max_tokens": 500}))["max_tokens"]
        == 500
    )


def test_fewshot_and_system_contents_are_preserved(model_setup):
    model = model_setup()
    messages = [
        {"role": "system", "content": "rule"},
        {"role": "user", "content": "example"},
        {"role": "assistant", "content": "example answer"},
        {"role": "user", "content": "question"},
    ]
    req = Instance(
        "generate_until",
        {},
        (model.apply_chat_template(messages), {}),
        0,
        ("toy", 0, 1),
    )
    payload = model.create_payload(req)
    assert payload["input_messages"] == messages
    assert json.loads(payload["prompt"].split("\n", 1)[1]) == messages[1:]
    assert payload["system_append"].endswith("rule")
    req.arguments = (model.apply_chat_template(messages[1:] + [messages[0]]), {})
    with pytest.raises(ValueError, match="precede"):
        model.create_payload(req)


@pytest.mark.parametrize(
    "status",
    [
        "ok",
        "refusal",
        "truncated",
        "turn_limit",
        "context_limit",
        "budget_mismatch",
        "tool_profile_mismatch",
    ],
)
def test_terminal_statuses_never_retry(model_setup, monkeypatch, status):
    model = model_setup()
    calls = []

    async def request_once(session, payload, key, attempt):
        calls.append(attempt)
        attempt.update(
            status=status,
            answer="scientifically wrong",
            response={"model": "test-model", "usage": {"output_tokens": 5}},
        )

    monkeypatch.setattr(model, "request_once", request_once)
    result = model.generate_until([question(model)])[0]
    assert len(calls) == 1
    assert result.metadata["status"] == status
    assert result.text == ("scientifically wrong" if status == "ok" else "")
    model.generate_until([question(model)])
    assert len(calls) == 1


def test_technical_retry_repeat_and_resume_records(model_setup, monkeypatch):
    model = model_setup()
    calls = []

    async def request_once(session, payload, key, attempt):
        calls.append(attempt)
        attempt.update(
            status="transport_failed" if attempt["attempt"] == 1 else "ok",
            answer="A",
            response={"usage": {"output_tokens": 7}},
        )

    monkeypatch.setattr(model, "request_once", request_once)
    result = model.generate_until([question(model), question(model)])
    assert len(calls) == 4
    assert [r.metadata["repeat_index"] for r in result] == [0, 1]
    assert result[0].metadata["record_path"] != result[1].metadata["record_path"]
    assert all(r.text == "A" for r in result)
    assert model.generate_until([question(model), question(model)]) == result
    assert len(calls) == 4


def test_question_deadline_includes_all_attempts(model_setup, monkeypatch):
    model = model_setup(timeout=0.02)

    async def request_once(session, payload, key, attempt):
        await asyncio.Event().wait()

    monkeypatch.setattr(model, "request_once", request_once)
    result = model.generate_until([question(model)])[0]
    assert result.metadata["status"] == "question_timeout"
    assert result.metadata["attempts"] == 1


@pytest.mark.parametrize("code", [401, 403])
def test_auth_failures_stop_batch_and_resume(model_setup, monkeypatch, code):
    model = model_setup()

    async def request_once(session, payload, key, attempt):
        attempt.update(status="http_error", http_status=code)

    monkeypatch.setattr(model, "request_once", request_once)
    for _ in range(2):
        with pytest.raises(PermissionError):
            model.generate_until([question(model)])


class FakeContent:
    def __init__(self, body):
        self.body = body

    async def iter_chunked(self, size):
        # Deliberately split Unicode/key boundaries; response bytes stay unchanged.
        for b in self.body:
            yield bytes([b])


class FakeHTTPResponse:
    def __init__(self, status=200, body=None):
        self.status = status
        self.headers = {"Content-Type": "text/event-stream"}
        self.content = FakeContent((body or stream()).encode())

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeSession:
    def __init__(self, response=None):
        self.response = response or FakeHTTPResponse()
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def test_loopback_proxy_preserves_payload_and_redacts_raw(tmp_path):
    async def check():
        session = FakeSession(
            FakeHTTPResponse(
                body=stream() + 'data: {"type":"ping", "text":"中文 fake-key"}\n\n'
            )
        )
        body = {
            "model": "test-model",
            "max_tokens": 128000,
            "output_config": {"effort": "max"},
            "messages": [{"role": "user", "content": "question"}],
            "system": "exact system",
        }
        async with MessagesProxy(
            session,
            "https://provider.invalid/anthropic",
            "test-model",
            "max",
            128000,
            "fake-key",
            tmp_path,
            [],
        ) as proxy:
            async with aiohttp.ClientSession() as client:
                async with client.post(
                    proxy.url + "/v1/messages",
                    json=body,
                    headers={"x-api-key": proxy.token},
                ) as response:
                    raw = await response.text()
                    assert response.status == 200 and "fake-key" in raw
            assert (
                session.calls[0][0] == "https://provider.invalid/anthropic/v1/messages"
            )
            assert session.calls[0][1]["json"] == body
            assert session.calls[0][1]["headers"]["x-api-key"] == "fake-key"
            assert proxy.calls[0]["status"] == "ok"
        captured = (tmp_path / "api_001.sse").read_text()
        assert "中文" in captured and "fake-key" not in captured
        assert "exact system" in (tmp_path / "api_001.json").read_text()

    asyncio.run(check())


@pytest.mark.parametrize(
    "field,value,status",
    [
        ("model", "other", "model_mismatch"),
        ("max_tokens", 256, "budget_mismatch"),
        ("output_config", {"effort": "high"}, "budget_mismatch"),
        ("tools", [{"name": "Bash"}], "tool_profile_mismatch"),
    ],
)
def test_proxy_rejects_profile_mismatch_without_upstream(
    tmp_path, field, value, status
):
    async def check():
        session = FakeSession()
        body = {
            "model": "test-model",
            "max_tokens": 128000,
            "output_config": {"effort": "max"},
            field: value,
        }
        async with MessagesProxy(
            session,
            "https://provider.invalid",
            "test-model",
            "max",
            128000,
            "fake-key",
            tmp_path,
            [],
        ) as proxy:
            async with aiohttp.ClientSession() as client:
                async with client.post(
                    proxy.url + "/v1/messages",
                    json=body,
                    headers={"x-api-key": proxy.token},
                ) as response:
                    assert response.status == 400
            assert not session.calls
            assert proxy.calls[0]["status"] == status
            assert proxy.failure.is_set()

    asyncio.run(check())


def test_failed_upstream_cannot_trigger_hidden_cli_retry(tmp_path):
    async def check():
        session = FakeSession(FakeHTTPResponse(status=524))
        async with MessagesProxy(
            session,
            "https://provider.invalid",
            "test-model",
            "max",
            128000,
            "fake-key",
            tmp_path,
            [],
        ) as proxy:
            body = {
                "model": "test-model",
                "max_tokens": 128000,
                "output_config": {"effort": "max"},
            }
            async with aiohttp.ClientSession() as client:
                for expected in (524, 409):
                    async with client.post(
                        proxy.url + "/v1/messages",
                        json=body,
                        headers={"x-api-key": proxy.token},
                    ) as response:
                        await response.read()
                        assert response.status == expected
            assert len(session.calls) == 1

    asyncio.run(check())


def test_isolated_cli_subprocess_capture_and_environment(model_setup, monkeypatch):
    model = model_setup()

    async def enter(proxy):
        proxy.url = "http://127.0.0.1:12345"
        return proxy

    async def exit(proxy, *args):
        pass

    monkeypatch.setattr(MessagesProxy, "__aenter__", enter)
    monkeypatch.setattr(MessagesProxy, "__aexit__", exit)
    script = (
        "import os,sys,json; sys.stdin.read(); print("
        + repr(trace())
        + ",end=''); print(os.environ.get('DONT_LEAK','isolated'),file=sys.stderr)"
    )
    monkeypatch.setattr(model, "command", lambda *a: [sys.executable, "-c", script])
    monkeypatch.setenv("DONT_LEAK", "inherited-secret")
    original = MessagesProxy.__init__

    def init(proxy, *args, **kwargs):
        original(proxy, *args, **kwargs)
        proxy.calls.append(
            {"path": "/v1/messages", "status": "ok", "payload": {"model": "test-model"}}
        )

    monkeypatch.setattr(MessagesProxy, "__init__", init)
    result = model.generate_until([question(model)])[0]
    assert result.text == "<answer>A</answer>"
    meta = result.metadata["agent_metadata"][0]
    assert "inherited-secret" not in Path(meta["stderr_path"]).read_text()
    assert "isolated" in Path(meta["stderr_path"]).read_text()
    assert Path(meta["trace_path"]).exists()


def test_cli_cancellation_kills_process_and_keeps_partial_trace(
    model_setup, monkeypatch
):
    model = model_setup(timeout=0.1)

    async def enter(proxy):
        proxy.url = "http://127.0.0.1:12345"
        return proxy

    async def exit(proxy, *args):
        pass

    monkeypatch.setattr(MessagesProxy, "__aenter__", enter)
    monkeypatch.setattr(MessagesProxy, "__aexit__", exit)
    script = "import os,time; print('partial',flush=True); time.sleep(100)"
    monkeypatch.setattr(model, "command", lambda *a: [sys.executable, "-c", script])
    result = model.generate_until([question(model)])[0]
    assert result.metadata["status"] == "question_timeout"
    record = json.loads(Path(result.metadata["record_path"]).read_text())
    pid = record["attempts"][0]["pid"]
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert "partial" in Path(record["attempts"][0]["trace_path"]).read_text()


def test_original_harness_filters_and_metadata(model_setup, monkeypatch):
    class ToyTask(ConfigurableTask):
        def download(self, *a, **k):
            self.dataset = {
                "test": Dataset.from_list(
                    [{"question": f"Question {i}", "answer": "A"} for i in range(10)]
                )
            }

    task = ToyTask(
        config={
            "task": "toy",
            "test_split": "test",
            "output_type": "generate_until",
            "doc_to_text": "question",
            "doc_to_target": "answer",
            "num_fewshot": 0,
            "generation_kwargs": {"max_gen_toks": 128000, "until": []},
            "metric_list": [
                {
                    "metric": "exact_match",
                    "aggregation": "mean",
                    "higher_is_better": True,
                }
            ],
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
        }
    )
    model = model_setup()

    async def run(session, payload, key, attempt):
        attempt.update(
            status="ok",
            answer="<answer>A</answer>",
            response={"usage": {"output_tokens": 4}},
        )

    monkeypatch.setattr(model, "request_once", run)
    result = evaluate(
        lm=model,
        task_dict={"toy": task},
        bootstrap_iters=0,
        apply_chat_template=True,
        log_samples=True,
    )
    assert result["results"]["toy"]["exact_match,xml"] == 1
    assert len(result["samples"]["toy"]) == 10
    assert result["samples"]["toy"][0]["response_metadata"][0][0]["status"] == "ok"


def test_unclosed_content_block_is_not_complete():
    body = stream(closed=False)
    body += 'data: {"type":"content_block_start", "index":0, "content_block":{"type":"text", "text":"partial"}}\n\n'
    body += 'data: {"type":"message_stop"}\n\n'
    assert inspect_stream(body, "test-model")["status"] == "transport_failed"


def test_proxy_rejects_native_tool_response():
    body = stream(closed=False)
    body += 'data: {"type":"content_block_start", "index":0, "content_block":{"type":"tool_use", "id":"t1", "name":"WebSearch", "input":{}}}\n\n'
    body += 'data: {"type":"content_block_stop", "index":0}\n\n'
    body += 'data: {"type":"message_stop"}\n\n'
    assert inspect_stream(body, "test-model")["status"] == "tool_profile_mismatch"


def test_interrupted_attempt_consumes_budget_and_preserves_search_count(
    model_setup, monkeypatch
):
    model = model_setup(max_attempts=2)

    async def complete(session, payload, key, attempt):
        attempt.update(status="ok", answer="A")

    monkeypatch.setattr(model, "request_once", complete)
    result = model.generate_until([question(model)])[0]
    path = Path(result.metadata["record_path"])
    record = json.loads(path.read_text())
    record["finished"] = False
    record["status"] = record["attempts"][0]["status"] = "running"
    record["answer"] = ""
    path.write_text(json.dumps(record))
    quota = path.with_suffix(".search.json")
    for _ in range(2):
        tools.reserve_search(quota)
    result = model.generate_until([question(model)])[0]
    assert result.metadata["attempts"] == 2
    assert json.loads(path.read_text())["attempts"][0]["error"] == "ProcessInterrupted"
    assert tools.reserve_search(quota) == 3
    with pytest.raises(ValueError):
        tools.reserve_search(quota)


def test_config_has_cli_tools_and_parser_identity(model_setup):
    model = model_setup()
    for name in ("sde_claude_code", "sde_cc_proxy", "sde_tools", "sde_sandbox"):
        assert len(model.config[name + "_sha256"]) == 64
    assert len(model.config["stream_sha256"]) == 64
    assert len(model.config["messages_parser_sha256"]) == 64
    assert model.config["agent_turn_limit"] is None
    assert model.config["native_tools"] is False
    assert model.config["search_attempt_limit"] == 3


def test_runtime_version_and_isolation_are_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(cc.subprocess, "check_output", lambda *a, **k: "other-version")
    with pytest.raises(ValueError, match="version"):
        cc.validate_runtime(
            Path(sys.executable), "desired-version", Path(sys.executable)
        )
    monkeypatch.setattr(
        cc.subprocess, "check_output", lambda *a, **k: "desired-version"
    )

    class Failure:
        returncode = 1

    monkeypatch.setattr(cc.subprocess, "run", lambda *a, **k: Failure())
    with pytest.raises(ValueError, match="isolation"):
        cc.validate_runtime(
            Path(sys.executable), "desired-version", Path(sys.executable)
        )


def test_compute_output_caps_and_clean_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "do-not-inherit")
    captured = {}

    class Process:
        pid = 123456
        returncode = 0

        def __init__(self, command, **kwargs):
            captured.update(kwargs)
            kwargs["stdout"].write(("中文" * 20000).encode())
            kwargs["stderr"].write(("错误" * 10000).encode())

        def wait(self, **kwargs):
            return 0

    monkeypatch.setattr(tools.subprocess, "Popen", Process)
    monkeypatch.setattr(tools.os, "killpg", lambda *a: None)
    value = tools.execute(tmp_path, "python -c 'print(1)'")
    assert len(value["stdout"]) == 24000
    assert len(value["stderr"]) == 8000
    assert "ANTHROPIC_API_KEY" not in captured["env"]
    assert tools.ACTIVE_PROCESS is None


def test_mcp_shutdown_stops_active_compute_group(monkeypatch):
    class Process:
        pid = 123456

    calls = []
    monkeypatch.setattr(tools, "ACTIVE_PROCESS", Process())
    monkeypatch.setattr(tools.os, "killpg", lambda *a: calls.append(a))
    with pytest.raises(SystemExit):
        tools.stop_active_process(tools.signal.SIGTERM, None)
    assert calls == [(Process.pid, tools.signal.SIGKILL)]


def test_full_cli_and_proxy_roundtrip_without_provider(model_setup, monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append(json.loads(body))
            data = stream().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    model = model_setup()
    model.config["base_url"] = f"http://127.0.0.1:{server.server_port}"
    script = """import json,os,sys,urllib.request
prompt=sys.stdin.read()
body={"model":"test-model","max_tokens":128000,"output_config":{"effort":"max"},"stream":True,"messages":[{"role":"user","content":prompt}],"system":"captured system"}
request=urllib.request.Request(os.environ["ANTHROPIC_BASE_URL"]+"/v1/messages",data=json.dumps(body).encode(),headers={"x-api-key":os.environ["ANTHROPIC_AUTH_TOKEN"],"Content-Type":"application/json"})
with urllib.request.urlopen(request,timeout=3) as response:
 response.read()
print(TRACE,end="")
""".replace("TRACE", repr(trace()))
    monkeypatch.setattr(model, "command", lambda *a: [sys.executable, "-c", script])
    try:
        results = model.generate_until([question(model, i) for i in range(10)])
        assert len(requests) == 10
        assert all(r.text == "<answer>A</answer>" for r in results)
        for result in results:
            attempt = json.loads(Path(result.metadata["record_path"]).read_text())[
                "attempts"
            ][0]
            assert attempt["api_calls"][0]["status"] == "ok"
            assert (
                "message_stop" in Path(attempt["api_calls"][0]["raw_path"]).read_text()
            )
            assert attempt["api_calls"][0]["payload"]["system"] == "captured system"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
