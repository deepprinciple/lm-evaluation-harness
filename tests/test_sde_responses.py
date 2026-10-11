import asyncio
import copy
import json
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from datasets import Dataset

from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask
from lm_eval.evaluator import evaluate
from lm_eval.models.sde_responses import SDEResponses, classify, parse_stream


def test_subscription_sparse_terminal_uses_only_closed_stream_items():
    item = {
        "id": "c1",
        "type": "custom_tool_call",
        "name": "exec",
        "call_id": "call1",
        "input": "text(1)",
        "status": "completed",
    }
    rows = native_events(item)
    rows[-1]["response"]["output"] = []
    parsed = parse_stream(stream(rows), agent_mode=True, allow_sparse_terminal=True)
    assert parsed["stream_error"] is None
    assert parsed["response"]["output"] == [item]
    assert parsed["sparse_terminal_output"] is True
    assert parsed["terminal_event"] == "response.completed"
    assert parse_stream(stream(rows), agent_mode=True)["stream_error"] is not None
    unclosed = [row for row in rows if row["type"] != "response.output_item.done"]
    assert (
        parse_stream(stream(unclosed), agent_mode=True, allow_sparse_terminal=True)[
            "stream_error"
        ]
        is not None
    )


def response(
    answer="<answer>A</answer>",
    status="completed",
    effort="max",
    model="gpt-6-astra",
    response_id="resp_1",
):
    return {
        "id": response_id,
        "object": "response",
        "status": status,
        "model": model,
        "reasoning": {"effort": effort},
        "error": None,
        "incomplete_details": None,
        "output": [
            {
                "id": "rs_1",
                "type": "reasoning",
                "content": [],
                "summary": [{"type": "summary_text", "text": "reasoning summary"}],
                "encrypted_content": "opaque",
            },
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": answer}],
            },
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 25,
            "output_tokens_details": {"reasoning_tokens": 20},
            "total_tokens": 35,
        },
    }


def events(final=None):
    final = response() if final is None else final
    initial = copy.deepcopy(final)
    initial.update(
        status="in_progress", output=[], usage=None, error=None, incomplete_details=None
    )
    return [
        {"type": "response.created", "response": initial},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"id": "rs_1", "type": "reasoning", "summary": []},
        },
        {
            "type": "response.reasoning_summary_text.delta",
            "output_index": 0,
            "summary_index": 0,
            "item_id": "rs_1",
            "delta": "summary fragment",
        },
        {
            "type": "response.reasoning_text.delta",
            "output_index": 0,
            "content_index": 0,
            "item_id": "rs_1",
            "delta": "returned reasoning text",
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": final["output"][0],
        },
        {
            "type": "response.output_text.delta",
            "output_index": 1,
            "content_index": 0,
            "item_id": "msg_1",
            "delta": "partial answer",
        },
        {"type": f"response.{final['status']}", "response": final},
    ]


def stream(items, done=True, crlf=False, multiline=False):
    rows = [": keepalive\n\n"]
    for number, item in enumerate(items):
        value = {**item, "sequence_number": number}
        body = json.dumps(value, ensure_ascii=False, indent=2 if multiline else None)
        data = "\n".join("data: " + line for line in body.splitlines())
        rows.append(f"event: {item['type']}\n{data}\n\n")
    if done:
        rows.append("data: [DONE]\n\n")
    result = "".join(rows)
    return result.replace("\n", "\r\n") if crlf else result


class FakeContent:
    def __init__(self, body, error=None, blocked=False, fragment=False):
        self.body, self.error, self.blocked, self.fragment = (
            body,
            error,
            blocked,
            fragment,
        )

    async def iter_chunked(self, size):
        for part in (
            [bytes([value]) for value in self.body] if self.fragment else [self.body]
        ):
            yield part
            await asyncio.sleep(0)
        if self.error:
            raise self.error
        if self.blocked:
            await asyncio.Event().wait()


class FakeResponse:
    def __init__(self, body=None, status=200, **kwargs):
        self.status = status
        self.headers = {}
        self.content = FakeContent(
            stream(events()).encode() if body is None else body, **kwargs
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class FakeSession:
    def __init__(self, responses):
        self.responses, self.payloads, self.headers = iter(responses), [], []

    def post(self, url, *, json, headers):
        self.payloads.append(json)
        self.headers.append(headers)
        return next(self.responses)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_RESPONSES_KEY", "fake-secret")
    original_sleep = asyncio.sleep

    async def no_backoff(delay):
        await original_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", no_backoff)

    def make(responses, **kwargs):
        session = FakeSession(responses)
        monkeypatch.setattr(aiohttp, "ClientSession", lambda **kw: session)
        model = SDEResponses(
            model="gpt-6-astra",
            max_tokens=128000,
            record_dir=str(tmp_path),
            api_key_env="TEST_RESPONSES_KEY",
            **kwargs,
        )
        return model, session

    return make


def request(model, doc_id=0, messages=None, kwargs=None):
    return Instance(
        "generate_until",
        {"question": f"Question {doc_id}"},
        (
            model.apply_chat_template(
                messages or [{"role": "user", "content": f"Question {doc_id}"}]
            ),
            kwargs or {},
        ),
        0,
        ("toy_responses", doc_id, 1),
    )


def record(output):
    return json.loads(Path(output.metadata["record_path"]).read_text())


def test_terminal_response_is_authoritative_and_reasoning_is_kept():
    parsed = parse_stream(stream(events(), crlf=True, multiline=True))
    assert classify(parsed, "gpt-6-astra", "max") == ("ok", "<answer>A</answer>")
    assert parsed["response"]["output"][0]["encrypted_content"] == "opaque"
    assert parsed["reasoning_deltas"][0]["delta"] == "summary fragment"
    assert parsed["reasoning_deltas"][1]["delta"] == "returned reasoning text"
    assert parsed["text_deltas"][0]["delta"] == "partial answer"
    assert (
        parsed["response"]["usage"]["output_tokens_details"]["reasoning_tokens"] == 20
    )


def test_done_sentinel_is_optional_but_not_a_completion_marker():
    assert (
        classify(parse_stream(stream(events(), done=False)), "gpt-6-astra", "max")[0]
        == "ok"
    )
    assert (
        classify(parse_stream("data: [DONE]\n\n"), "gpt-6-astra", "max")[0]
        == "transport_failed"
    )
    assert (
        classify(parse_stream(stream(events()[:-1])), "gpt-6-astra", "max")[0]
        == "transport_failed"
    )


def test_final_only_relay_response_does_not_require_missing_optional_events():
    body = stream([{"type": "response.completed", "response": response()}])
    assert classify(parse_stream(body), "gpt-6-astra", "max")[0] == "ok"


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_terminal",
        "restart",
        "changed_id",
        "replayed_sequence",
        "malformed_json",
        "unfinished_frame",
        "after_done",
        "bad_usage",
        "bad_text",
        "unknown_event",
        "event_mismatch",
    ],
)
def test_ambiguous_or_malformed_stream_is_not_a_normal_answer(change):
    items = events()
    if change == "duplicate_terminal":
        items.append(items[-1])
    elif change == "restart":
        items.insert(2, items[0])
    elif change == "changed_id":
        items[-1]["response"]["id"] = "resp_other"
    elif change == "bad_usage":
        items[-1]["response"]["usage"] = "bad usage"
    elif change == "bad_text":
        items[-1]["response"]["output"][1]["content"][0]["text"] = None
    elif change == "unknown_event":
        items.insert(3, {"type": "response.unknown"})
    body = stream(items, done=change != "unfinished_frame")
    if change == "replayed_sequence":
        body = body.replace('"sequence_number": 1', '"sequence_number": 0')
    elif change == "malformed_json":
        body += "data: {bad}\n\n"
    elif change == "unfinished_frame":
        body += 'data: {"type":"ping"}'
    elif change == "after_done":
        body += stream([items[-1]], done=False)
    elif change == "event_mismatch":
        body = body.replace("event: response.completed", "event: response.failed")
    assert classify(parse_stream(body), "gpt-6-astra", "max")[0] == "protocol_error"


@pytest.mark.parametrize(
    "reason,status",
    [
        ("max_output_tokens", "truncated"),
        ("content_filter", "refusal"),
        ("server_timeout", "generation_incomplete"),
    ],
)
def test_incomplete_responses_keep_reason_and_partial_output(reason, status):
    final = response("partial", status="incomplete")
    final["incomplete_details"] = {"reason": reason}
    parsed = parse_stream(stream(events(final)))
    assert classify(parsed, "gpt-6-astra", "max") == (status, "partial")


@pytest.mark.parametrize(
    "change,status",
    [
        ("model", "model_mismatch"),
        ("effort", "protocol_error"),
        ("no_effort", "protocol_error"),
        ("refusal", "refusal"),
        ("tool", "protocol_error"),
        ("empty", "empty"),
        ("wrong", "ok"),
    ],
)
def test_completion_checks_model_effort_refusal_and_no_tools(change, status):
    final = response()
    if change == "model":
        final["model"] = "other-model"
    elif change == "effort":
        final["reasoning"]["effort"] = "high"
    elif change == "no_effort":
        final.pop("reasoning")
    elif change == "refusal":
        final["output"][1]["content"] = [
            {"type": "refusal", "refusal": "I cannot help"}
        ]
    elif change == "tool":
        final["output"].append(
            {"type": "function_call", "name": "shell", "arguments": "{}"}
        )
    elif change == "empty":
        final["output"][1]["content"][0]["text"] = ""
    elif change == "wrong":
        final["output"][1]["content"][0]["text"] = "wrong scientific answer"
    assert (
        classify(parse_stream(stream(events(final))), "gpt-6-astra", "max")[0] == status
    )


def test_server_error_events_and_failed_responses_are_technical_failures():
    final = response(status="failed")
    final["error"] = {"code": "server_error", "message": "busy"}
    assert (
        classify(parse_stream(stream(events(final))), "gpt-6-astra", "max")[0]
        == "transport_failed"
    )
    for error in (
        {"type": "error", "code": "server_is_overloaded", "message": "busy"},
        {"type": "error", "error": {"code": "server_error"}},
    ):
        assert (
            classify(parse_stream(stream([error])), "gpt-6-astra", "max")[0]
            == "transport_failed"
        )


def test_relay_keepalive_is_not_a_completion_or_parser_error():
    rows = events()
    rows.insert(2, {"type": "keepalive"})
    result = parse_stream(stream(rows))
    assert result["stream_error"] is None
    assert result["terminal_event"] == "response.completed"
    assert result["event_counts"]["keepalive"] == 1
    incomplete = parse_stream(stream(rows[:-1]))
    assert classify(incomplete, "gpt-6-astra", "max")[0] == "transport_failed"


def test_payload_maps_budget_and_preserves_system_and_fewshot_messages(setup):
    model, _ = setup([])
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "example"},
        {"role": "assistant", "content": "example answer"},
        {"role": "user", "content": "question"},
    ]
    kwargs = {
        "max_tokens": 256000,
        "temperature": 0,
        "until": ["Q:"],
        "do_sample": False,
    }
    payload = model.create_payload(request(model, messages=messages, kwargs=kwargs))
    assert payload == {
        "model": "gpt-6-astra",
        "input": messages,
        "reasoning": {"effort": "max"},
        "max_output_tokens": 256000,
        "stream": True,
    }
    assert kwargs["max_tokens"] == 256000
    assert model.config["api_format"] == "responses"
    assert (
        not {"temperature", "seed", "stop", "tools", "max_tokens", "thinking"}
        & payload.keys()
    )
    with pytest.raises(ValueError, match="Unsupported"):
        model.create_payload(request(model, kwargs={"tools": []}))


def test_capture_preserves_unicode_redacts_secrets_and_resumes_without_network(setup):
    model, session = setup(
        [
            FakeResponse(
                body=stream(events(response("fake-secret 中文"))).encode(),
                fragment=True,
            )
        ]
    )
    req = request(model)
    output = model.generate_until([req])[0]
    attempt = record(output)["attempts"][0]
    assert output.text == "<redacted> 中文"
    assert output.metadata["returned_effort"] == "max"
    assert output.metadata["usage"]["output_tokens_details"]["reasoning_tokens"] == 20
    assert Path(attempt["body_path"]).read_text() == attempt["body"]
    assert "fake-secret" not in json.dumps(record(output))
    assert session.headers[0] == {"Authorization": "Bearer fake-secret"}
    assert model.generate_until([req]) == [output]
    assert len(session.payloads) == 1


def test_technical_errors_retry_first_complete_response_wins(setup):
    model, session = setup(
        [
            FakeResponse(status=524, body=b"timeout"),
            FakeResponse(body=stream(events()[:-1]).encode()),
            FakeResponse(),
        ]
    )
    output = model.generate_until([request(model)])[0]
    assert output.metadata["status"] == "ok"
    assert output.metadata["attempts"] == len(session.payloads) == 3


@pytest.mark.parametrize(
    "kind,status", [("refusal", "refusal"), ("truncated", "truncated"), ("wrong", "ok")]
)
def test_refusal_truncation_and_wrong_answers_do_not_retry(setup, kind, status):
    final = response("wrong scientific answer")
    if kind == "truncated":
        final.update(
            status="incomplete", incomplete_details={"reason": "max_output_tokens"}
        )
    elif kind == "refusal":
        final["output"][1]["content"] = [{"type": "refusal", "refusal": "Cannot help"}]
    model, session = setup([FakeResponse(body=stream(events(final)).encode())])
    output = model.generate_until([request(model)])[0]
    assert output.metadata["status"] == status
    assert output.text == ("wrong scientific answer" if status == "ok" else "")
    assert len(session.payloads) == 1


@pytest.mark.parametrize("blocked", [False, True])
def test_failed_transport_keeps_partial_reasoning_and_does_not_score_it(setup, blocked):
    model, _ = setup(
        [
            FakeResponse(
                body=stream(events()[:-1], done=False).encode(),
                blocked=blocked,
                error=None if blocked else aiohttp.ClientPayloadError("drop"),
            )
        ],
        max_attempts=1,
        timeout=0.03 if blocked else 3600,
    )
    output = model.generate_until([request(model)])[0]
    assert output.metadata["status"] == (
        "question_timeout" if blocked else "transport_failed"
    )
    assert output.text == ""
    attempt = record(output)["attempts"][0]
    assert attempt["reasoning_deltas"][1]["delta"] == "returned reasoning text"
    assert attempt["output_items"][0]["encrypted_content"] == "opaque"
    assert Path(attempt["body_path"]).read_text() == attempt["body"]


@pytest.mark.parametrize("code", [401, 403])
def test_auth_failure_stops_batch_without_retry(setup, code):
    model, session = setup([FakeResponse(status=code, body=b"bad token")])
    with pytest.raises(PermissionError):
        model.generate_until([request(model)])
    assert len(session.payloads) == 1


def test_repeats_have_independent_raw_records(setup):
    model, session = setup(
        [
            FakeResponse(
                body=stream(events(response("first", response_id="a"))).encode()
            ),
            FakeResponse(
                body=stream(events(response("second", response_id="b"))).encode()
            ),
        ]
    )
    req = request(model)
    outputs = model.generate_until([req, req])
    assert [output.text for output in outputs] == ["first", "second"]
    assert [output.metadata["repeat_index"] for output in outputs] == [0, 1]
    assert outputs[0].metadata["record_path"] != outputs[1].metadata["record_path"]
    assert model.generate_until([req, req]) == outputs
    assert len(session.payloads) == 2


def test_cancellation_after_completion_marker_does_not_mean_transport_success(
    setup, tmp_path
):
    model, session = setup([FakeResponse(blocked=True)])
    attempt: dict[str, Any] = {
        "attempt": 1,
        "status": "running",
        "record_path": str(tmp_path / "record.json"),
    }

    async def run():
        task = asyncio.create_task(
            model.request_once(
                session, model.create_payload(request(model)), "fake-secret", attempt
            )
        )
        for _ in range(10):
            await asyncio.sleep(0)
            if (
                attempt.get("body_path")
                and Path(attempt["body_path"]).exists()
                and Path(attempt["body_path"]).stat().st_size
            ):
                break
        assert Path(attempt["body_path"]).read_text() == stream(events())
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert attempt["terminal_event"] == "response.completed"
    assert attempt["status"] == "transport_failed"


def test_harness_filters_only_answer_text_and_preserves_metadata(setup):
    model, session = setup([FakeResponse() for _ in range(10)])

    class ToyTask(ConfigurableTask):
        def download(self, *args, **kwargs):
            self.dataset = {
                "test": Dataset.from_list(
                    [{"question": str(i), "answer": "A"} for i in range(10)]
                )
            }

    task = ToyTask(
        config={
            "task": "toy_responses",
            "output_type": "generate_until",
            "test_split": "test",
            "doc_to_text": "question",
            "doc_to_target": "answer",
            "generation_kwargs": {"temperature": 0, "max_gen_toks": 4096},
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
        }
    )
    evaluated = evaluate(
        model, {"toy_responses": task}, log_samples=True, apply_chat_template=True
    )
    assert evaluated["results"]["toy_responses"]["exact_match,xml"] == 1.0
    assert len(session.payloads) == 10
    assert all(payload["max_output_tokens"] == 4096 for payload in session.payloads)
    assert all(
        sample["response_metadata"][0][0]["returned_effort"] == "max"
        for sample in evaluated["samples"]["toy_responses"]
    )


@pytest.mark.parametrize(
    "change",
    [
        "changed_model",
        "bad_status",
        "bad_error",
        "negative_sequence",
        "bad_delta",
        "bad_reasoning",
    ],
)
def test_malformed_lifecycle_cannot_overwrite_validated_control_fields(change):
    items = events()
    if change == "changed_model":
        items[0]["response"]["model"] = "different-model"
    elif change == "bad_status":
        items[-1]["response"]["status"] = "in_progress"
    elif change == "bad_error":
        items = [{"type": "error", "error": "bad error"}]
    elif change == "bad_delta":
        items[3]["delta"] = {"text": "not a string"}
    elif change == "bad_reasoning":
        items[-1]["response"]["reasoning"] = ["max"]
    body = stream(items)
    if change == "negative_sequence":
        body = body.replace('"sequence_number": 0', '"sequence_number": -1')
    assert classify(parse_stream(body), "gpt-6-astra", "max")[0] == "protocol_error"


def test_effort_mismatch_is_recorded_and_only_bounded_technical_retries_apply(setup):
    model, session = setup(
        [
            FakeResponse(body=stream(events(response(effort="high"))).encode()),
            FakeResponse(),
        ]
    )
    output = model.generate_until([request(model)])[0]
    assert output.metadata["status"] == "ok"
    assert len(session.payloads) == 2
    assert record(output)["attempts"][0]["response"]["reasoning"]["effort"] == "high"
    assert record(output)["attempts"][0]["status"] == "protocol_error"


NATIVE_ITEMS: list[dict[str, Any]] = [
    {
        "type": "function_call",
        "id": "f1",
        "call_id": "c1",
        "name": "exec_command",
        "namespace": "functions",
        "arguments": '{"cmd":"pwd"}',
        "status": "completed",
    },
    {
        "type": "custom_tool_call",
        "id": "p1",
        "call_id": "c2",
        "name": "apply_patch",
        "namespace": "another_plugin",
        "input": "*** Begin Patch\n*** End Patch",
    },
    {
        "type": "web_search_call",
        "id": "w1",
        "status": "completed",
        "action": {"type": "search", "queries": ["science"]},
    },
    {
        "type": "code_interpreter_call",
        "id": "ci1",
        "status": "completed",
        "container_id": "container_1",
        "code": "print(2)",
        "outputs": [{"type": "logs", "logs": "2"}],
    },
    {
        "type": "file_search_call",
        "id": "fs1",
        "status": "completed",
        "queries": ["science"],
        "results": [],
    },
    {
        "type": "mcp_call",
        "id": "m1",
        "name": "unfamiliar_valid_tool",
        "server_label": "other_server",
        "arguments": "{}",
        "output": "result",
        "status": "completed",
    },
    {
        "type": "shell_call",
        "id": "s1",
        "call_id": "c3",
        "status": "completed",
        "action": {"commands": ["pwd"]},
    },
    {
        "type": "computer_call",
        "id": "pc1",
        "call_id": "c4",
        "status": "completed",
        "action": {"type": "screenshot"},
        "pending_safety_checks": [],
    },
    {
        "type": "tool_search_call",
        "id": "ts1",
        "status": "completed",
        "execution": "server",
        "arguments": {"query": "science"},
    },
    {
        "type": "tool_search_output",
        "id": "to1",
        "status": "completed",
        "execution": "server",
        "tools": [{"type": "function", "name": "any_name", "parameters": {}}],
    },
    {
        "type": "image_generation_call",
        "id": "im1",
        "status": "completed",
        "result": "base64",
    },
    {
        "type": "apply_patch_call",
        "id": "ap1",
        "call_id": "c5",
        "status": "completed",
        "operation": {"type": "delete_file", "path": "temporary.txt"},
    },
    {
        "type": "program",
        "id": "prog1",
        "call_id": "c6",
        "code": "await tools.any_name()",
        "fingerprint": "opaque",
    },
    {
        "type": "program_output",
        "id": "po1",
        "call_id": "c6",
        "status": "completed",
        "result": "value",
    },
    {
        "type": "additional_tools",
        "id": "at1",
        "role": "developer",
        "tools": [{"type": "namespace", "name": "functions", "tools": []}],
    },
]


def native_events(item, extra=()):
    final = response()
    final["output"] = [copy.deepcopy(item)]
    initial = {**final, "status": "in_progress", "output": []}
    added = copy.deepcopy(item)
    if "status" in added:
        added["status"] = "in_progress"
    if added["type"] in {"function_call", "mcp_call"}:
        added["arguments"] = ""
    if added["type"] == "custom_tool_call":
        added["input"] = ""
    return [
        {"type": "response.created", "response": initial},
        {"type": "response.output_item.added", "output_index": 0, "item": added},
        *extra,
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        {"type": "response.completed", "response": final},
    ]


@pytest.mark.parametrize("item", NATIVE_ITEMS, ids=lambda item: item["type"])
def test_agent_mode_accepts_native_items_without_relaxing_no_tools(item):
    body = stream(native_events(item))
    parsed = parse_stream(body, agent_mode=True)
    assert parsed["stream_error"] is None
    assert parsed["terminal_event"] == "response.completed"
    assert parsed["response"]["output"] == [item]
    assert parsed["output_items"] == [item]
    assert parse_stream(body)["stream_error"] is not None


@pytest.mark.parametrize(
    "item,event",
    [
        (
            NATIVE_ITEMS[0],
            {"type": "response.function_call_arguments.delta", "delta": '{"cmd":'},
        ),
        (
            NATIVE_ITEMS[0],
            {
                "type": "response.function_call_arguments.done",
                "arguments": '{"cmd":"pwd"}',
            },
        ),
        (
            NATIVE_ITEMS[1],
            {
                "type": "response.custom_tool_call_input.delta",
                "delta": "*** Begin Patch",
            },
        ),
        (
            NATIVE_ITEMS[1],
            {
                "type": "response.custom_tool_call_input.done",
                "input": "*** Begin Patch\n*** End Patch",
            },
        ),
        (NATIVE_ITEMS[2], {"type": "response.web_search_call.in_progress"}),
        (NATIVE_ITEMS[2], {"type": "response.web_search_call.searching"}),
        (NATIVE_ITEMS[2], {"type": "response.web_search_call.completed"}),
        (
            NATIVE_ITEMS[3],
            {"type": "response.code_interpreter_call_code.delta", "delta": "print(2)"},
        ),
        (
            NATIVE_ITEMS[3],
            {"type": "response.code_interpreter_call_code.done", "code": "print(2)"},
        ),
        (NATIVE_ITEMS[3], {"type": "response.code_interpreter_call.interpreting"}),
        (NATIVE_ITEMS[4], {"type": "response.file_search_call.searching"}),
        (
            NATIVE_ITEMS[5],
            {"type": "response.mcp_call_arguments.done", "arguments": "{}"},
        ),
        (NATIVE_ITEMS[5], {"type": "response.mcp_call.completed"}),
        (
            NATIVE_ITEMS[6],
            {
                "type": "response.shell_call_command.delta",
                "command_index": 0,
                "delta": "pwd",
            },
        ),
        (
            NATIVE_ITEMS[10],
            {
                "type": "response.image_generation_call.partial_image",
                "partial_image_index": 0,
                "partial_image_b64": "base64",
            },
        ),
    ],
    ids=lambda value: value["type"],
)
def test_agent_mode_records_real_native_events(item, event):
    event = {"output_index": 0, "item_id": item["id"], **event}
    parsed = parse_stream(stream(native_events(item, [event])), agent_mode=True)
    assert parsed["stream_error"] is None
    assert parsed["event_counts"][event["type"]] == 1
    assert parsed["tool_events"][0]["type"] == event["type"]
    assert parsed["tool_events"][0]["item_id"] == item["id"]


@pytest.mark.parametrize(
    "change",
    [
        "bad_name",
        "bad_namespace",
        "bad_arguments",
        "missing_call_id",
        "bad_status",
        "changed_item_id",
        "changed_type",
        "wrong_terminal_item",
        "wrong_terminal_arguments",
        "bad_index",
        "wrong_event_item",
        "bad_delta",
        "unknown_event",
        "replayed_sequence",
        "changed_response_id",
        "bad_response_status",
        "after_terminal",
        "after_done",
        "unknown_output",
    ],
)
def test_agent_mode_rejects_malformed_protocol(change):
    item = copy.deepcopy(NATIVE_ITEMS[0])
    extra = {
        "type": "response.function_call_arguments.delta",
        "item_id": item["id"],
        "output_index": 0,
        "delta": "{}",
    }
    rows = native_events(item, [extra])
    if change == "bad_name":
        item["name"] = None
    elif change == "bad_namespace":
        item["namespace"] = 42
    elif change == "bad_arguments":
        item["arguments"] = {}
    elif change == "missing_call_id":
        item.pop("call_id")
    elif change == "bad_status":
        item["status"] = "invented"
    elif change == "changed_item_id":
        rows[3]["item"] = {**item, "id": "other"}
    elif change == "changed_type":
        rows[3]["item"] = NATIVE_ITEMS[1]
    elif change == "wrong_terminal_item":
        rows[-1]["response"]["output"] = [NATIVE_ITEMS[1]]
    elif change == "wrong_terminal_arguments":
        rows[-1]["response"]["output"][0]["arguments"] = "different"
    elif change == "bad_index":
        extra["output_index"] = True
    elif change == "wrong_event_item":
        extra["item_id"] = "other"
    elif change == "bad_delta":
        extra["delta"] = {}
    elif change == "unknown_event":
        extra["type"] = "response.unknown_tool_event"
    elif change == "changed_response_id":
        rows[-1]["response"]["id"] = "other"
    elif change == "bad_response_status":
        rows[-1]["response"]["status"] = "in_progress"
    elif change == "after_terminal":
        rows.append(extra)
    elif change == "unknown_output":
        rows[-1]["response"]["output"] = [{"type": "invented_call", "id": "x"}]
    body = stream(rows)
    if change == "after_done":
        body += stream([extra])
    elif change == "replayed_sequence":
        body = body.replace('"sequence_number": 2', '"sequence_number": 1')
    assert parse_stream(body, agent_mode=True)["stream_error"] is not None


def test_agent_mode_interrupted_tool_stream_is_not_completion():
    parsed = parse_stream(stream(native_events(NATIVE_ITEMS[2])[:-1]), agent_mode=True)
    assert parsed["stream_error"] is None
    assert parsed["terminal_event"] is None


@pytest.mark.parametrize(
    "change", ["missing_status", "missing_action", "bad_action", "duplicate_item_id"]
)
def test_malformed_native_output_is_not_accepted(change):
    final = response()
    item = copy.deepcopy(NATIVE_ITEMS[2])
    if change == "missing_status":
        item.pop("status")
    elif change == "missing_action":
        item.pop("action")
    elif change == "bad_action":
        item["action"] = {"type": "find_in_page", "url": None, "pattern": 42}
    final["output"] = [item]
    if change == "duplicate_item_id":
        final["output"].append(copy.deepcopy(item))
    parsed = parse_stream(
        stream([{"type": "response.completed", "response": final}]), agent_mode=True
    )
    assert parsed["stream_error"] is not None


@pytest.mark.parametrize("item", NATIVE_ITEMS[2:5])
def test_native_added_items_can_be_partial_before_tool_execution(item):
    rows = native_events(item)
    rows[1]["item"] = {"type": item["type"], "id": item["id"], "status": "in_progress"}
    parsed = parse_stream(stream(rows), agent_mode=True)
    assert parsed["stream_error"] is None
