import asyncio
import json
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from datasets import Dataset

from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask
from lm_eval.evaluator import evaluate
from lm_eval.models.sde_messages import SDEMessages, classify, parse_stream


def start(message_id="msg_1", model="claude-opus-5-5"):
    return {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "usage": {"input_tokens": 12, "output_tokens": 1},
        },
    }


def block(index, kind, field, value):
    return [
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": kind, field: ""},
        },
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": f"{kind}_delta", field: value},
        },
        {"type": "content_block_stop", "index": index},
    ]


def events(answer="<answer>A</answer>", reason="end_turn", message_id="msg_1"):
    return [
        start(message_id),
        *block(0, "thinking", "thinking", "推理文本"),
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "redacted_thinking", "data": "opaque"},
        },
        {"type": "content_block_stop", "index": 1},
        *block(2, "text", "text", answer),
        {
            "type": "message_delta",
            "delta": {"stop_reason": reason},
            "usage": {"output_tokens": 33},
        },
        {"type": "message_stop"},
    ]


def stream(items, crlf=False):
    body = ": keepalive\n\n" + "".join(
        f"event: {item['type']}\ndata: {json.dumps(item, ensure_ascii=False)}\n\n"
        for item in items
    )
    return body.replace("\n", "\r\n") if crlf else body


def test_agent_native_citation_delta_is_preserved():
    citation = {
        "type": "web_search_result_location",
        "url": "https://docs.python.org/3/library/math.html",
        "title": "math",
        "cited_text": "Euclidean norm",
        "encrypted_index": "opaque",
    }
    items = [
        start(),
        *block(0, "text", "text", "hypot computes a norm"),
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}},
        {"type": "message_stop"},
    ]
    items.insert(
        3,
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "citations_delta", "citation": citation},
        },
    )
    parsed = parse_stream(stream(items), "claude-opus-5-5", agent_mode=True)
    assert parsed["stream_error"] is None
    assert parsed["message_stop"] and parsed["blocks_closed"]
    assert parsed["response"]["content"][0]["citations"] == [citation]
    assert parse_stream(stream(items), "claude-opus-5-5")["stream_error"] is not None


@pytest.mark.parametrize("citation", [None, "wrong", []])
def test_agent_malformed_citation_delta_is_rejected(citation):
    items = [start(), *block(0, "text", "text", "hypot computes a norm")]
    items.insert(
        3,
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "citations_delta", "citation": citation},
        },
    )
    assert (
        parse_stream(stream(items), "claude-opus-5-5", agent_mode=True)["stream_error"]
        is not None
    )


class FakeContent:
    def __init__(self, body, error=None, blocked=False, fragment=False):
        self.body = body
        self.error = error
        self.blocked = blocked
        self.fragment = fragment

    async def iter_chunked(self, size):
        parts = [bytes([b]) for b in self.body] if self.fragment else [self.body]
        for part in parts:
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
        self.responses = iter(responses)
        self.payloads = []
        self.headers = []

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
    monkeypatch.setenv("TEST_MESSAGES_KEY", "fake-secret")
    original_sleep = asyncio.sleep

    async def no_backoff(delay):
        await original_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", no_backoff)

    def make(responses, **kwargs):
        session = FakeSession(responses)
        monkeypatch.setattr(aiohttp, "ClientSession", lambda **kw: session)
        model = SDEMessages(
            model="claude-opus-5-5",
            max_tokens=128000,
            record_dir=str(tmp_path),
            api_key_env="TEST_MESSAGES_KEY",
            **kwargs,
        )
        return model, session

    return make


def request(model, messages=None, doc_id=0, kwargs=None):
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
        ("toy", doc_id, 1),
    )


def test_resume_preserves_failed_sse_and_uses_a_new_attempt_file(setup):
    model, _ = setup([FakeResponse(status=524, body=b"old timeout")], max_attempts=1)
    first = model.generate_until([request(model)])[0]
    path = Path(first.metadata["record_path"])
    old = json.loads(path.read_text())
    old_stream = Path(old["attempts"][0]["body_path"])
    before = old_stream.read_bytes()
    resumed, session = setup([FakeResponse()], resume=True, timeout=120, max_attempts=1)
    again = resumed.generate_until([request(resumed)])[0]
    record = json.loads(path.read_text())
    assert again.metadata["status"] == "ok"
    assert again.metadata["record_path"] == str(path)
    assert record["attempts"][:1] == old["attempts"]
    assert old_stream.read_bytes() == before
    assert Path(record["attempts"][1]["body_path"]).is_file()
    assert record["attempts"][1]["body_path"] != str(old_stream)
    assert len(session.payloads) == 1


def record(output):
    return json.loads(Path(output.metadata["record_path"]).read_text())


def test_message_delta_accepts_nullable_relay_metadata():
    items = events()
    items[-2]["delta"].update(stop_details=None, container=None, stop_sequence=None)
    items[-2]["context_management"] = {"applied_edits": []}
    parsed = parse_stream(stream(items), "claude-opus-5-5")
    assert classify(parsed, "claude-opus-5-5") == ("ok", "<answer>A</answer>")
    assert parsed["message_stop"] is True
    assert parsed["response"]["stop_details"] is None
    assert parsed["response"]["container"] is None


@pytest.mark.parametrize("field", ["container", "stop_details"])
def test_message_delta_rejects_non_object_metadata(field):
    items = events()
    items[-2]["delta"][field] = "invalid"
    parsed = parse_stream(stream(items), "claude-opus-5-5")
    assert classify(parsed, "claude-opus-5-5")[0] == "protocol_error"


def test_message_delta_still_rejects_unknown_fields():
    items = events()
    items[-2]["delta"]["unknown_generation_control"] = None
    parsed = parse_stream(stream(items), "claude-opus-5-5")
    assert classify(parsed, "claude-opus-5-5")[0] == "protocol_error"


def test_stream_preserves_thinking_signature_redaction_and_usage():
    items = events()
    items.insert(
        2,
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "signature-value"},
        },
    )
    parsed = parse_stream(stream(items, crlf=True), "claude-opus-5-5")
    assert classify(parsed, "claude-opus-5-5") == ("ok", "<answer>A</answer>")
    assert parsed["response"]["content"][0]["thinking"] == "推理文本"
    assert parsed["response"]["content"][0]["signature"] == "signature-value"
    assert parsed["response"]["content"][1]["data"] == "opaque"
    assert parsed["response"]["usage"] == {"input_tokens": 12, "output_tokens": 33}


@pytest.mark.parametrize(
    "failure",
    [
        "no_stop",
        "open_block",
        "delta_after_stop",
        "duplicate_start",
        "bad_json",
        "trailing_frame",
        "missing_delta",
    ],
)
def test_incomplete_or_malformed_stream_cannot_be_scored(failure):
    items = events()
    if failure == "no_stop":
        items.pop()
    elif failure == "open_block":
        items = [
            item for item in items if item != {"type": "content_block_stop", "index": 2}
        ]
    elif failure == "delta_after_stop":
        items.insert(
            4,
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "late"},
            },
        )
    elif failure == "duplicate_start":
        items.insert(1, start())
    elif failure == "missing_delta":
        items = [item for item in items if item["type"] != "message_delta"]
    body = stream(items)
    if failure == "bad_json":
        body += "data: {broken}\n\n"
    elif failure == "trailing_frame":
        body += 'data: {"type":"ping"}'
    status, _ = classify(parse_stream(body, "claude-opus-5-5"), "claude-opus-5-5")
    assert status in {"transport_failed", "protocol_error", "generation_incomplete"}


@pytest.mark.parametrize(
    "reason,answer,status",
    [
        ("max_tokens", "partial", "truncated"),
        ("refusal", "Cannot help", "refusal"),
        ("end_turn", "", "empty"),
        ("end_turn", "wrong scientific answer", "ok"),
        ("pause_turn", "partial", "generation_incomplete"),
    ],
)
def test_provider_stop_reasons(reason, answer, status):
    parsed = parse_stream(stream(events(answer, reason)), "claude-opus-5-5")
    assert classify(parsed, "claude-opus-5-5") == (status, answer)


def test_safe_thinking_restart_retains_both_segments_without_combining_answers():
    initial = [start("old"), *block(0, "thinking", "thinking", "unfinished")]
    parsed = parse_stream(stream(initial + events(message_id="new")), "claude-opus-5-5")
    assert classify(parsed, "claude-opus-5-5") == ("ok", "<answer>A</answer>")
    assert [segment["response"]["id"] for segment in parsed["relay_segments"]] == [
        "old",
        "new",
    ]
    assert (
        parsed["relay_segments"][0]["response"]["content"][0]["thinking"]
        == "unfinished"
    )


@pytest.mark.parametrize(
    "prior", ["text", "refusal", "completed", "same_id", "wrong_model"]
)
def test_ambiguous_restart_is_not_accepted(prior):
    initial = [start("old"), *block(0, "thinking", "thinking", "unfinished")]
    if prior == "text":
        initial += block(1, "text", "text", "first answer")
    elif prior == "refusal":
        initial += [
            {"type": "message_delta", "delta": {"stop_reason": "refusal"}, "usage": {}}
        ]
    elif prior == "completed":
        initial = events(message_id="old")
    elif prior == "wrong_model":
        initial[0]["message"]["model"] = "other-model"
    parsed = parse_stream(
        stream(initial + events(message_id="old" if prior == "same_id" else "new")),
        "claude-opus-5-5",
    )
    assert classify(parsed, "claude-opus-5-5")[0] == "protocol_error"


def test_wrong_model_and_tools_are_not_normal_answers():
    items = events()
    items[0]["message"]["model"] = "other-model"
    assert (
        classify(parse_stream(stream(items), "claude-opus-5-5"), "claude-opus-5-5")[0]
        == "model_mismatch"
    )
    items = events()
    items.insert(
        1,
        {
            "type": "content_block_start",
            "index": 3,
            "content_block": {
                "type": "tool_use",
                "id": "tool_1",
                "name": "shell",
                "input": {},
            },
        },
    )
    items.insert(2, {"type": "content_block_stop", "index": 3})
    assert (
        classify(parse_stream(stream(items), "claude-opus-5-5"), "claude-opus-5-5")[0]
        == "protocol_error"
    )


def test_server_tool_json_deltas_require_explicit_agent_mode():
    items = [
        start(),
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "search_1",
                "name": "web_search",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"query":'},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '"science"}'},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        {"type": "message_stop"},
    ]
    body = stream(items)
    assert parse_stream(body, "claude-opus-5-5")["stream_error"] is not None
    parsed = parse_stream(body, "claude-opus-5-5", agent_mode=True)
    assert parsed["stream_error"] is None
    assert parsed["response"]["content"][0]["partial_json"] == '{"query":"science"}'
    assert parsed["message_stop"] and parsed["blocks_closed"]
    assert classify(parsed, "claude-opus-5-5")[0] == "protocol_error"


def test_agent_mode_does_not_relax_delta_type_or_stream_completion():
    items = events()
    items[1]["content_block"]["type"] = "server_tool_use"
    parsed = parse_stream(stream(items), "claude-opus-5-5", agent_mode=True)
    assert parsed["stream_error"] is not None


def test_payload_preserves_messages_and_moves_system_to_top_level(setup):
    model, _ = setup([])
    messages = [
        {"role": "system", "content": "system A"},
        {"role": "system", "content": "system B"},
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
    ]
    kwargs = {
        "max_tokens": 128000,
        "temperature": 0,
        "do_sample": False,
        "until": ["Q:"],
    }
    payload = model.create_payload(request(model, messages, kwargs=kwargs))
    assert payload == {
        "model": "claude-opus-5-5",
        "max_tokens": 128000,
        "messages": messages[2:],
        "system": [
            {"type": "text", "text": "system A"},
            {"type": "text", "text": "system B"},
        ],
        "output_config": {"effort": "max"},
        "stream": True,
        "thinking": {"type": "adaptive", "display": "summarized"},
    }
    assert model.config["api_format"] == "messages"
    assert model.config["thinking_display"] == "summarized"
    assert (
        not {"temperature", "reasoning_effort", "seed", "stop", "tools"}
        & payload.keys()
    )
    with pytest.raises(ValueError, match="leading"):
        model.create_payload(
            request(
                model,
                [
                    {"role": "user", "content": "Q"},
                    {"role": "system", "content": "late"},
                ],
            )
        )
    with pytest.raises(ValueError, match="user"):
        model.create_payload(request(model, [{"role": "system", "content": "only"}]))
    with pytest.raises(ValueError, match="Unsupported"):
        model.create_payload(request(model, kwargs={"tools": []}))


def test_stream_capture_is_durable_redacted_and_reusable(setup):
    model, session = setup(
        [FakeResponse(body=stream(events("fake-secret 中文")).encode(), fragment=True)]
    )
    req = request(model)
    output = model.generate_until([req])[0]
    saved = record(output)
    attempt = saved["attempts"][0]
    assert output.text == "<redacted> 中文"
    assert attempt["response"]["content"][0]["thinking"] == "推理文本"
    assert Path(attempt["body_path"]).read_text() == attempt["body"]
    assert "fake-secret" not in json.dumps(saved)
    assert session.headers[0] == {
        "x-api-key": "fake-secret",
        "anthropic-version": "2023-06-01",
    }
    assert model.generate_until([req]) == [output]
    assert len(session.payloads) == 1


@pytest.mark.parametrize(
    "reason,status",
    [("max_tokens", "truncated"), ("refusal", "refusal"), ("end_turn", "ok")],
)
def test_normal_answer_refusal_and_truncation_do_not_retry(setup, reason, status):
    model, session = setup(
        [FakeResponse(body=stream(events("answer", reason)).encode())]
    )
    output = model.generate_until([request(model)])[0]
    assert output.metadata["status"] == status
    assert len(session.payloads) == 1
    assert output.text == ("answer" if status == "ok" else "")


def test_technical_errors_retry_first_complete_answer_wins(setup):
    model, session = setup(
        [
            FakeResponse(status=524, body=b"timeout"),
            FakeResponse(body=stream(events()[:-1]).encode()),
            FakeResponse(),
        ]
    )
    output = model.generate_until([request(model)])[0]
    assert output.metadata["attempts"] == len(session.payloads) == 3
    assert output.metadata["status"] == "ok"
    assert len(output.metadata["stream_metadata"]) == 3
    assert record(output)["attempts"][1]["response"]["usage"]["output_tokens"] == 33


@pytest.mark.parametrize("blocked", [False, True])
def test_interrupted_stream_retains_partial_thinking_and_usage(setup, blocked):
    body = stream(events()[:4]).encode() + b"data: partial fake-secret"
    model, _ = setup(
        [
            FakeResponse(
                body=body,
                error=None if blocked else aiohttp.ClientPayloadError("dropped"),
                blocked=blocked,
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
    assert attempt["response"]["content"][0]["thinking"] == "推理文本"
    assert attempt["response"]["usage"]["input_tokens"] == 12
    assert attempt["body"].endswith("data: partial <redacted>")
    assert Path(attempt["body_path"]).read_text() == attempt["body"]


@pytest.mark.parametrize("code", [401, 403])
def test_auth_failure_stops_without_retry(setup, code):
    model, session = setup([FakeResponse(status=code, body=b"no permission")])
    with pytest.raises(PermissionError):
        model.generate_until([request(model)])
    assert len(session.payloads) == 1


def test_repeats_have_independent_records_and_sse_files(setup):
    model, _ = setup(
        [
            FakeResponse(body=stream(events("first", message_id="a")).encode()),
            FakeResponse(body=stream(events("second", message_id="b")).encode()),
        ]
    )
    req = request(model)
    outputs = model.generate_until([req, req])
    assert [output.text for output in outputs] == ["first", "second"]
    assert [output.metadata["repeat_index"] for output in outputs] == [0, 1]
    assert (
        record(outputs[0])["attempts"][0]["body_path"]
        != record(outputs[1])["attempts"][0]["body_path"]
    )


def test_original_harness_scores_text_and_logs_execution_metadata(setup):
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
            "task": "toy_messages",
            "output_type": "generate_until",
            "test_split": "test",
            "doc_to_text": "question",
            "doc_to_target": "answer",
            "generation_kwargs": {"temperature": 0, "max_gen_toks": 128000},
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
        model, {"toy_messages": task}, log_samples=True, apply_chat_template=True
    )
    assert evaluated["results"]["toy_messages"]["exact_match,xml"] == 1.0
    assert len(session.payloads) == 10
    assert all(payload["max_tokens"] == 128000 for payload in session.payloads)
    assert all(
        sample["response_metadata"][0][0]["status"] == "ok"
        for sample in evaluated["samples"]["toy_messages"]
    )


@pytest.mark.parametrize("change", ["bad_usage", "changed_model", "bad_error"])
def test_malformed_control_fields_cannot_be_accepted(change):
    items = events()
    if change == "bad_usage":
        items[0]["message"]["usage"] = [12]
        items = [item for item in items if item["type"] != "message_delta"]
        items.insert(
            -1, {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}
        )
    elif change == "changed_model":
        items[0]["message"]["model"] = "wrong-model"
        items[-2]["delta"]["model"] = "claude-opus-5-5"
    else:
        items.insert(-1, {"type": "error", "error": "malformed"})
    parsed = parse_stream(stream(items), "claude-opus-5-5")
    assert classify(parsed, "claude-opus-5-5")[0] == "protocol_error"


def test_overload_error_is_retried_and_raw_error_is_saved(setup):
    error = {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
    model, session = setup(
        [FakeResponse(body=stream([start(), error]).encode()), FakeResponse()]
    )
    output = model.generate_until([request(model)])[0]
    assert output.metadata["status"] == "ok"
    assert len(session.payloads) == 2
    assert record(output)["attempts"][0]["stream_error"] == error


def test_stream_lines_are_saved_before_request_finishes_and_cancel_is_not_success(
    setup, tmp_path
):
    model, session = setup([FakeResponse(blocked=True)])
    attempt: dict[str, Any] = {
        "attempt": 1,
        "record_path": str(tmp_path / "record.json"),
        "status": "running",
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
    assert attempt["status"] == "transport_failed"
    assert attempt["response"]["content"][-1]["text"] == "<answer>A</answer>"


def test_invalid_utf8_cannot_be_scored_as_normal_text(setup):
    model, _ = setup(
        [
            FakeResponse(
                body=stream(events("answer")).encode().replace(b"answer", b"\xffanswer")
            )
        ],
        max_attempts=1,
    )
    output = model.generate_until([request(model)])[0]
    assert output.metadata["status"] == "protocol_error"
    assert output.text == ""


def test_overloaded_before_stream_does_not_consume_attempts(setup):
    model, session = setup(
        [
            FakeResponse(
                body=b'{"type":"error","error":{"type":"overloaded_error"}}', status=529
            ),
            FakeResponse(),
        ],
        max_attempts=1,
    )
    output = model.generate_until([request(model)])[0]
    assert output.metadata["status"] == "ok"
    assert output.metadata["budget_attempts"] == 1
    assert len(session.payloads) == 2
    first = record(output)["attempts"][0]
    assert first["budget_exempt"] is True
    assert Path(first["body_path"]).exists()


def test_thinking_display_can_restore_the_original_payload(setup):
    model, _ = setup([], thinking_display="none")
    assert "thinking" not in model.create_payload(request(model))
    omitted, _ = setup([], thinking_display="omitted")
    assert omitted.create_payload(request(omitted))["thinking"] == {
        "type": "adaptive",
        "display": "omitted",
    }
    with pytest.raises(ValueError, match="thinking_display"):
        setup([], thinking_display="raw")
