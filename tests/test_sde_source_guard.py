"""Synthetic controls and local HTTP integration; no external model requests."""

import asyncio
import copy
import json
from contextlib import asynccontextmanager

import aiohttp
import pytest
from aiohttp import web

from lm_eval.models.sde_cc_proxy import MessagesProxy
from lm_eval.models.sde_codex_proxy import ResponsesProxy, prepare_payload
from lm_eval.models.sde_source_guard import NOTICE, filter_tool_returns, source_rules


@pytest.mark.parametrize(
    "source",
    [
        "https://huggingface.co/datasets/deep-principle/science_physics/blob/main/exact_match.csv",
        "https://huggingface.co/api/resolve-cache/datasets/deep-principle/science_chemistry/hash/nmr.jsonl",
        "https://huggingface.co/datasets/jeffbrian/science_chemistry/commit/hash",
        "https://huggingface.co/datasets/Maxue627/SCILLM-benchmarks/commit/hash",
        "https://github.com/howiehwong/sde-harness",
        "https://huggingface.co/datasets/SALT-NLP/gpqa_diamond-context-baseline-gptos/viewer/default/train",
        "https://huggingface.co/user/model/commit/hash?file=eval%2Fgpqa198.jsonl",
        "https://raw.githubusercontent.com/deepprinciple/lm-evaluation-harness/main/lm_eval/tasks/science/physics/utils.py",
        "https%3A%2F%2Fhuggingface.co%2Fdatasets%2Fjeffbrian%2Fscience_materials%2Fdata",
        r"https:\/\/huggingface.co\/datasets\/deep-principle\/science_biology\/data",
        "https://huggingface.co&#47;datasets&#47;deep-principle&#47;science_physics/data",
        "/home/evaluator/lm-evaluation-harness/results/run/raw.json",
        "The dataset's answer (SMILES): CCO",
        "The recorded answer is 1/6.",
    ],
)
def test_whole_tool_result_is_withheld(source):
    secret = "SYNTHETIC-REFERENCE-ANSWER"
    old = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": source + " " + secret,
                    }
                ],
            }
        ]
    }
    before = copy.deepcopy(old)
    new, logs = filter_tool_returns(old)
    assert old == before
    assert secret not in json.dumps(new)
    block = new["messages"][0]["content"][0]
    assert block["tool_use_id"] == "t1" and block["is_error"] is True
    assert block["content"] == NOTICE
    assert logs[0]["original_content"] == source + " " + secret


@pytest.mark.parametrize(
    "text",
    [
        "https://docs.python.org/3/library/math.html hypot(x,y)",
        "https://pubmed.ncbi.nlm.nih.gov/12345/ metabolic association",
        "https://arxiv.org/abs/1808.06558 spherical designs",
        "https://pubs.acs.org/doi/10.1021/example chemistry source",
        "https://huggingface.co/datasets/science-lab/measured-spectra/data.csv",
        "Computation: d=2.34 Angstrom; answer = 1/6.",
        "Reference implementation: sklearn; ground truth labels y_test.",
        "https://github.com/EleutherAI/lm-evaluation-harness/blob/main/README.md",
    ],
)
def test_scientific_outputs_pass_unchanged(text):
    body = {
        "input": [{"type": "function_call_output", "call_id": "f1", "output": text}]
    }
    new, logs = filter_tool_returns(body)
    assert new is body and not logs


def test_do_not_scan_prompts_or_assistant_and_preserve_tools():
    source = "deep-principle/science_physics Answer: 1/6"
    body = {
        "system": source,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": source}]},
            {"role": "assistant", "content": [{"type": "text", "text": source}]},
        ],
        "tools": [{"name": "Bash"}, {"name": "WebSearch"}],
    }
    new, logs = filter_tool_returns(body)
    assert new is body and not logs


@pytest.mark.parametrize("kind", ["function_call_output", "custom_tool_call_output"])
def test_responses_tool_ids_and_parameters_are_preserved(kind):
    original = {
        "model": "test-model",
        "stream": True,
        "reasoning": {"effort": "max"},
        "input": [{"type": kind, "call_id": "c1", "output": "SCILLM-benchmarks GOLD"}],
        "tools": [{"type": "function", "name": "exec_command", "parameters": {}}],
    }
    new, logs = filter_tool_returns(original)
    assert new["input"][0] == {"type": kind, "call_id": "c1", "output": NOTICE}
    assert (
        new["tools"] == original["tools"] and new["reasoning"] == original["reasoning"]
    )
    assert logs[0]["sha256"] and logs[0]["original_content"].endswith("GOLD")
    prepared = prepare_payload(new, "test-model", "max", 128000)
    assert prepared["max_output_tokens"] == 128000


def test_encoded_search_blob_alone_is_not_readable_evidence():
    assert not source_rules(
        {"output": "ordinary paper", "encrypted_output": "SCILLM-benchmarks"}
    )
    assert source_rules(
        {"output": "SCILLM-benchmarks GOLD", "encrypted_output": "opaque"}
    )


@asynccontextmanager
async def provider(handler):
    app = web.Application()
    app.router.add_post("/v1/messages", handler)
    app.router.add_post("/responses", handler)
    app.router.add_post("/alpha/search", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    try:
        yield f"http://127.0.0.1:{runner.addresses[0][1]}"
    finally:
        await runner.cleanup()


def test_cc_filters_before_the_next_primary_request(tmp_path):
    received = []

    async def upstream(request):
        received.append(await request.json())
        return web.json_response(
            {
                "id": "m1",
                "type": "message",
                "role": "assistant",
                "model": "test-model",
                "content": [{"type": "text", "text": "Independent answer."}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 3, "output_tokens": 2},
            }
        )

    async def run():
        async with (
            provider(upstream) as url,
            aiohttp.ClientSession() as session,
            MessagesProxy(
                session, url, "test-model", "max", 128000, "fake-key", tmp_path, []
            ) as proxy,
        ):
            body = {
                "model": "test-model",
                "max_tokens": 128000,
                "output_config": {"effort": "max"},
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t1",
                                "content": "deep-principle/science_physics recorded answer is GOLD",
                            }
                        ],
                    }
                ],
            }
            async with session.post(
                proxy.url + "/v1/messages",
                json=body,
                headers={"x-api-key": proxy.token},
            ) as response:
                assert response.status == 200
                await response.read()
            assert received[0]["messages"][0]["content"][0]["content"] == NOTICE
            assert "GOLD" not in json.dumps(received)
            assert proxy.calls[0]["source_filter_count"] == 1
            assert "GOLD" in proxy.calls[0]["source_filters"][0]["original_content"]
            assert proxy.calls[0]["status"] == "ok" and not proxy.failure.is_set()

    asyncio.run(run())


def test_native_subscription_drops_summary_results_and_encrypted_blob(
    tmp_path, monkeypatch
):
    from lm_eval.models import sde_codex_proxy as module

    leaked = {
        "output": "SCILLM-benchmarks recorded answer is GOLD",
        "encrypted_output": "CIPHER-GOLD",
        "results": [
            {"url": "https://huggingface.co/datasets/jeffbrian/science_chemistry/data"}
        ],
    }
    ordinary = {
        "output": "official Python docs",
        "encrypted_output": "official-cipher",
        "results": [{"url": "https://docs.python.org/3/"}],
    }
    replies = [leaked, ordinary]

    async def upstream(request):
        await request.json()
        return web.json_response(replies.pop(0))

    async def run():
        async with provider(upstream) as url:
            monkeypatch.setattr(module, "SUBSCRIPTION_ENDPOINT", url + "/responses")
            async with (
                aiohttp.ClientSession() as session,
                ResponsesProxy(
                    session,
                    url + "/responses",
                    "test-model",
                    "max",
                    128000,
                    "fake-key",
                    tmp_path,
                    [],
                    auth_mode="chatgpt",
                ) as proxy,
            ):
                for i in range(2):
                    async with session.post(
                        proxy.url + "/" + proxy.token + "/alpha/search",
                        json={
                            "model": "test-model",
                            "commands": {"search_query": [{"q": "science"}]},
                        },
                        headers={"Authorization": "Bearer fake-subscription"},
                    ) as response:
                        text = await response.text()
                        assert response.status == (422 if i == 0 else 200)
                        if i == 0:
                            assert "GOLD" not in text and "huggingface" not in text
                        else:
                            assert json.loads(text) == ordinary
                assert proxy.search_calls[0]["status"] == "source_blocked"
                assert not proxy.failure.is_set()
                assert (
                    json.loads(
                        (tmp_path / "native_search_001.response.json").read_text()
                    )
                    == leaked
                )

    asyncio.run(run())


def test_deepseek_mcp_keeps_raw_evidence_but_returns_no_reference(
    tmp_path, monkeypatch
):
    from lm_eval.models import sde_deepseek_search as bridge
    from tests.test_sde_deepseek_search import native_response, proxy, request

    data = native_response()
    data["content"][0]["content"][0]["url"] = (
        "https://huggingface.co/datasets/jeffbrian/science_chemistry/data"
    )
    data["content"][1]["text"] = "Recorded answer: GOLD"
    p = proxy(tmp_path, data=data)
    reply = asyncio.run(bridge.forward_search(request(p), p))
    value = json.loads(reply.text)
    assert value["status"] == "source_blocked" and value["results"] == []
    assert "GOLD" not in reply.text and "encrypted_content" not in reply.text
    saved = json.loads((tmp_path / "search_001.json").read_text())
    assert "GOLD" in saved["raw_response"] and saved["status"] == "source_blocked"
    assert not p.failure.is_set()


def test_optional_message_fields_do_not_change_protocol_validation():
    body = {"messages": None, "input": None}
    new, logs = filter_tool_returns(body)
    assert new is body and logs == []


def test_codex_filters_plain_tool_outputs_before_upstream(tmp_path):
    from tests.test_sde_responses import NATIVE_ITEMS, native_events, stream

    received = []

    async def upstream(request):
        received.append(await request.json())
        events = native_events(NATIVE_ITEMS[1])
        for event in (events[0], events[-1]):
            event["response"]["model"] = "test-model"
        return web.Response(
            text=stream(events, done=False), content_type="text/event-stream"
        )

    async def run():
        async with (
            provider(upstream) as url,
            aiohttp.ClientSession() as session,
            ResponsesProxy(
                session,
                url + "/responses",
                "test-model",
                "max",
                128000,
                "fake-key",
                tmp_path,
                [],
            ) as proxy,
        ):
            body = {
                "model": "test-model",
                "reasoning": {"effort": "max"},
                "stream": True,
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "c1",
                        "output": "https://huggingface.co/datasets/deep-principle/science_physics/data.csv GOLD",
                    }
                ],
            }
            async with session.post(
                proxy.url + "/" + proxy.token + "/responses",
                json=body,
                headers={"Authorization": "Bearer " + proxy.token},
            ) as response:
                assert response.status == 200
                await response.read()
            assert received[0]["input"][0]["output"] == NOTICE
            assert "GOLD" not in json.dumps(received)
            assert proxy.calls[0]["original_payload"] == body
            assert proxy.calls[0]["source_filter_count"] == 1
            assert proxy.calls[0]["status"] == "ok" and not proxy.failure.is_set()

    asyncio.run(run())


def test_old_answers_cannot_match_new_source_policy_identity():
    from lm_eval.models.sde_chat import SDEChatCompletion

    model = object.__new__(SDEChatCompletion)
    model.config = {}
    for index in range(10):
        old = {
            "config": {"model": "test-model", "max_tokens": 128000},
            "task": "task",
            "doc_id": index,
            "repeat_index": 0,
        }
        guarded = copy.deepcopy(old)
        guarded["config"]["benchmark_source_policy"] = "benchmark-source-exclusion-v1"
        guarded["config"]["sde_source_guard_sha256"] = "new-guard-hash"
        assert model.normalize_identity(old) != model.normalize_identity(guarded)
