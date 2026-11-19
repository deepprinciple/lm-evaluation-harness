import asyncio
import json
from pathlib import Path

import aiohttp
import pytest
from datasets import Dataset

from lm_eval.api.instance import Instance
from lm_eval.api.model import CacheHook
from lm_eval.api.task import ConfigurableTask
from lm_eval.evaluator import evaluate
from lm_eval.models.sde_chat import SDEChatCompletion, classify, digest


def response(text="<answer>A</answer>", finish="stop"):
    return {
        "model": "deepseek-flash",
        "choices": [
            {
                "message": {"content": text, "reasoning_content": "reasoning text"},
                "finish_reason": finish,
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 9},
    }


class FakeContent:
    def __init__(self, body, error=None, blocked=False, delay=0):
        self.body = body
        self.error = error
        self.blocked = blocked
        self.delay = delay

    async def iter_chunked(self, size):
        yield self.body
        await asyncio.sleep(0)
        if self.delay:
            event = asyncio.Event()
            asyncio.get_running_loop().call_later(self.delay, event.set)
            await event.wait()
        if self.error:
            raise self.error
        if self.blocked:
            await asyncio.Event().wait()


class FakeResponse:
    def __init__(
        self,
        status=200,
        body=None,
        error=None,
        blocked=False,
        delay=0,
        headers=None,
        enter_error=None,
    ):
        self.status = status
        self.headers = headers or {}
        self.enter_error = enter_error
        self.session: FakeSession
        self.content = FakeContent(
            body or json.dumps(response()).encode(), error, blocked, delay
        )

    async def __aenter__(self):
        if self.enter_error:
            raise self.enter_error
        self.session.active += 1
        self.session.max_active = max(self.session.max_active, self.session.active)
        return self

    async def __aexit__(self, *args):
        self.session.active -= 1
        return False


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.payloads = []
        self.active = 0
        self.max_active = 0

    def post(self, url, *, json, headers):
        self.payloads.append(json)
        result = next(self.responses)
        result.session = self
        return result

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_SDE_KEY", "fake-secret")
    original_sleep = asyncio.sleep

    async def no_backoff(delay):
        await original_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", no_backoff)

    def make(responses, **kwargs):
        session = FakeSession(responses)
        monkeypatch.setattr(aiohttp, "ClientSession", lambda **kw: session)
        model = SDEChatCompletion(
            model="deepseek-flash",
            max_tokens=393216,
            record_dir=str(tmp_path),
            api_key_env="TEST_SDE_KEY",
            **kwargs,
        )
        return model, session

    return make


def request(model, doc_id=0, kwargs=None):
    prompt = model.apply_chat_template(
        [{"role": "user", "content": f"Question {doc_id}"}]
    )
    return Instance(
        "generate_until",
        {"question": f"Question {doc_id}"},
        (prompt, kwargs or {}),
        0,
        ("toy", doc_id, 1),
    )


@pytest.mark.parametrize(
    "finish,text,status",
    [
        ("stop", "wrong scientific answer", "ok"),
        ("stop", "", "empty"),
        ("length", "<answer>A</answer>", "truncated"),
        ("content_filter", "I cannot help", "refusal"),
        (None, "partial", "generation_incomplete"),
    ],
)
def test_classification(finish, text, status):
    assert classify(response(text, finish)) == (status, text)


def test_payload_respects_harness_overrides_without_forwarding_sampling_fields(setup):
    model, _ = setup([])
    kwargs = {
        "max_tokens": 393216,
        "until": ["Q:"],
        "temperature": 0,
        "do_sample": False,
    }
    payload = model.create_payload(request(model, kwargs=kwargs))
    assert payload["max_tokens"] == 393216
    assert payload["reasoning_effort"] == "max"
    assert payload["thinking"] == {"type": "enabled"}
    assert (
        not {"temperature", "seed", "stop", "max_completion_tokens", "tools"}
        & payload.keys()
    )
    assert kwargs["until"] == ["Q:"]
    assert model.create_payload(request(model))["max_tokens"] == 393216
    with pytest.raises(ValueError, match="Unsupported"):
        model.create_payload(request(model, kwargs={"unexpected": True}))


def test_reasoning_is_saved_and_resume_avoids_new_network_call(setup):
    model, session = setup([FakeResponse()])
    req = request(model)
    first = model.generate_until([req])[0]
    second = model.generate_until([req])[0]
    assert first == second
    assert len(session.payloads) == 1
    assert first.text == "<answer>A</answer>"
    with open(first.metadata["record_path"]) as file:
        record = json.load(file)
    assert (
        record["attempts"][0]["response"]["choices"][0]["message"]["reasoning_content"]
        == "reasoning text"
    )
    assert "fake-secret" not in json.dumps(record)


def test_repeats_are_independent(setup):
    model, session = setup(
        [
            FakeResponse(body=json.dumps(response("first")).encode()),
            FakeResponse(body=json.dumps(response("second")).encode()),
        ]
    )
    req = request(model)
    outputs = model.generate_until([req, req])
    assert [out.text for out in outputs] == ["first", "second"]
    assert [out.metadata["repeat_index"] for out in outputs] == [0, 1]
    assert outputs[0].metadata["record_path"] != outputs[1].metadata["record_path"]
    assert len(session.payloads) == 2
    assert model.generate_until([req, req]) == outputs


def test_only_technical_failures_are_retried(setup):
    model, session = setup(
        [
            FakeResponse(status=524),
            FakeResponse(body=json.dumps(response("")).encode()),
            FakeResponse(),
        ]
    )
    result = model.generate_until([request(model)])[0]
    assert result.metadata["status"] == "ok"
    assert result.metadata["attempts"] == len(session.payloads) == 3


@pytest.mark.parametrize(
    "finish,text,status",
    [
        ("length", "<answer>A</answer>", "truncated"),
        ("stop", "wrong", "ok"),
        ("content_filter", "refusal", "refusal"),
    ],
)
def test_truncation_refusal_and_wrong_answers_do_not_trigger_retry(
    setup, finish, text, status
):
    model, session = setup(
        [FakeResponse(body=json.dumps(response(text, finish)).encode())]
    )
    result = model.generate_until([request(model)])[0]
    assert result.metadata["status"] == status
    assert len(session.payloads) == 1
    assert result.text == (text if status == "ok" else "")


def test_failed_transport_preserves_partial_body(setup):
    model, _ = setup(
        [
            FakeResponse(
                body=b"partial fake-secret", error=aiohttp.ClientPayloadError("dropped")
            )
        ],
        max_attempts=1,
    )
    result = model.generate_until([request(model)])[0]
    assert result.metadata["status"] == "transport_failed"
    assert result.text == ""
    with open(result.metadata["record_path"]) as file:
        record = json.load(file)
    assert record["attempts"][0]["body"] == "partial <redacted>"


def test_question_timeout_preserves_partial_body(setup):
    model, _ = setup([FakeResponse(body=b"partial", blocked=True)], timeout=0.03)
    result = model.generate_until([request(model)])[0]
    assert result.metadata["status"] == "question_timeout"
    with open(result.metadata["record_path"]) as file:
        record = json.load(file)
    assert record["attempts"][0]["body"] == "partial"


def test_auth_error_stops_batch_without_retry(setup):
    model, session = setup([FakeResponse(status=401)])
    with pytest.raises(PermissionError):
        model.generate_until([request(model)])
    assert len(session.payloads) == 1
    with pytest.raises(PermissionError):
        model.generate_until([request(model)])
    assert len(session.payloads) == 1


def test_response_cache_is_rejected(setup):
    model, _ = setup([])
    hook = CacheHook(None)
    hook.dbdict = {}
    with pytest.raises(ValueError, match="use_cache"):
        model.set_cache_hook(hook)


def score(doc, answers):
    return {"acc": float(answers[0] == doc["answer"])}


class ToyTask(ConfigurableTask):
    def download(self, *args, **kwargs):
        self.dataset = {
            "test": Dataset.from_list(
                [{"question": f"Q{i}", "answer": "A"} for i in range(10)]
            )
        }


def test_adapter_runs_through_original_evaluator_and_filters(setup):
    model, session = setup([FakeResponse() for _ in range(20)])
    task = ToyTask(
        config={
            "task": "toy",
            "test_split": "test",
            "output_type": "generate_until",
            "doc_to_text": "question",
            "doc_to_target": "answer",
            "num_fewshot": 0,
            "repeats": 2,
            "process_results": score,
            "metric_list": [
                {"metric": "acc", "aggregation": "mean", "higher_is_better": True}
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
    result = evaluate(
        model,
        {"toy": task},
        bootstrap_iters=0,
        log_samples=True,
        apply_chat_template=True,
    )
    assert result["results"]["toy"]["acc,xml"] == 1.0
    assert len(session.payloads) == 20
    assert all(sample["filtered_resps"] == ["A"] for sample in result["samples"]["toy"])
    assert all(
        len(sample["response_metadata"][0]) == 2 for sample in result["samples"]["toy"]
    )


def test_concurrency_limit_is_enforced(setup):
    model, session = setup([FakeResponse() for _ in range(10)], num_concurrent=2)
    results = model.generate_until([request(model, i) for i in range(10)])
    assert len(results) == 10
    assert session.max_active == 2
    assert all(result.metadata["status"] == "ok" for result in results)


def test_waiting_for_a_worker_does_not_consume_question_timeout(setup):
    model, _ = setup(
        [FakeResponse(delay=0.03) for _ in range(3)], num_concurrent=1, timeout=0.06
    )
    results = model.generate_until([request(model, i) for i in range(3)])
    assert all(result.metadata["status"] == "ok" for result in results)


def test_backoff_cannot_exceed_question_deadline(setup):
    model, session = setup([FakeResponse(status=524)], timeout=1)
    result = model.generate_until([request(model)])[0]
    assert result.metadata["status"] == "question_timeout"
    assert len(session.payloads) == 1


def test_interrupted_attempt_consumes_one_attempt_when_resuming(setup):
    model, session = setup([FakeResponse(), FakeResponse()])
    req = request(model)
    result = model.generate_until([req])[0]
    from pathlib import Path

    path = Path(result.metadata["record_path"])
    record = json.loads(path.read_text())
    record["finished"] = False
    record["attempts"][-1]["status"] = "running"
    path.write_text(json.dumps(record))
    recovered = model.generate_until([req])[0]
    assert recovered.metadata["attempts"] == 2
    assert len(session.payloads) == 2
    assert json.loads(path.read_text())["attempts"][0]["error"] == "ProcessInterrupted"


def connect_error():
    from types import SimpleNamespace

    key = SimpleNamespace(ssl=None, host="relay", port=443)
    return aiohttp.ClientConnectorError(key, OSError(61, "refused"))


def test_pre_response_failures_do_not_consume_attempts(setup):
    model, _ = setup(
        [
            FakeResponse(status=429, body=b'{"error":"rate"}'),
            FakeResponse(enter_error=connect_error()),
            FakeResponse(status=529, body=b'{"error":"overloaded"}'),
            FakeResponse(),
        ],
        max_attempts=1,
    )
    result = model.generate_until([request(model)])[0]
    assert result.metadata["status"] == "ok"
    assert result.metadata["attempts"] == 4
    assert result.metadata["budget_attempts"] == 1
    with open(result.metadata["record_path"]) as file:
        record = json.load(file)
    assert [a.get("budget_exempt", False) for a in record["attempts"]] == [
        True,
        True,
        True,
        False,
    ]


@pytest.mark.parametrize(
    "status,body",
    [
        (429, b'{"error":{"code":"insufficient_quota"}}'),
        (500, b"server error"),
        (524, b"timeout"),
    ],
)
def test_quota_and_possibly_billed_failures_consume_attempts(setup, status, body):
    model, session = setup([FakeResponse(status=status, body=body)], max_attempts=1)
    result = model.generate_until([request(model)])[0]
    assert result.metadata["status"] == "transport_failed"
    assert result.metadata["budget_attempts"] == len(session.payloads) == 1


def test_retry_delay_honors_retry_after_with_cap():
    from lm_eval.models.sde_stream import retry_delay

    assert retry_delay("7", 3) == 7.0
    assert retry_delay("600", 0) == 60.0
    assert retry_delay(None, 1) == 2.0
    assert retry_delay("Wed, 21 Oct 2026 07:28:00 GMT", 10) == 60.0


def test_resumed_question_gets_a_fresh_deadline_window(setup):
    model, session = setup([FakeResponse(), FakeResponse()], timeout=60)
    req = request(model)
    result = model.generate_until([req])[0]
    from pathlib import Path

    path = Path(result.metadata["record_path"])
    record = json.loads(path.read_text())
    # Simulate a crash long ago: wall-clock time since the first start exceeds
    # the timeout, but the downtime must not be charged to the question.
    record["started_at"] -= 3600
    record["deadline_windows"][0]["started_at"] -= 3600
    record["finished"] = False
    record["attempts"][-1]["status"] = "running"
    path.write_text(json.dumps(record))
    recovered = model.generate_until([req])[0]
    assert recovered.metadata["status"] == "ok"
    assert recovered.metadata["deadline_windows"] == 2
    assert recovered.metadata["budget_attempts"] == 2
    assert len(session.payloads) == 2


def test_record_directory_cannot_have_two_active_runners(setup):
    import fcntl

    model, session = setup([])
    model.record_dir.mkdir(exist_ok=True)
    with (model.record_dir / "runner.lock").open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            model.generate_until([request(model)])
    assert session.payloads == []


def test_changing_concurrency_resumes_existing_records(setup):
    model, _ = setup([FakeResponse()], num_concurrent=4)
    first = model.generate_until([request(model)])[0]
    resumed, resumed_session = setup([], num_concurrent=1)
    again = resumed.generate_until([request(resumed)])[0]
    assert again.metadata["record_path"] == first.metadata["record_path"]
    assert again.text == first.text and resumed_session.payloads == []


@pytest.mark.parametrize("kwargs", [{"max_tokens": 4096}, {"max_gen_toks": 4096}])
def test_task_budget_must_match_the_experiment_budget(setup, kwargs):
    model, session = setup([])
    with pytest.raises(ValueError, match="--gen_kwargs max_tokens=393216"):
        model.generate_until([request(model, kwargs=kwargs)])
    assert session.payloads == []


def test_explicit_resume_reuses_success_with_new_runtime_budgets(setup):
    model, _ = setup([FakeResponse(body=json.dumps(response("wrong answer")).encode())])
    first = model.generate_until([request(model)])[0]
    path = Path(first.metadata["record_path"])
    before = path.read_bytes()
    resumed, session = setup([], resume=True, timeout=120, max_attempts=1)
    again = resumed.generate_until([request(resumed)])[0]
    assert again.text == "wrong answer"
    assert again.metadata["record_path"] == str(path)
    assert path.read_bytes() == before
    assert session.payloads == []


def test_explicit_resume_appends_attempts_with_a_fresh_budget(setup):
    model, _ = setup([FakeResponse(status=524, body=b"old timeout")], max_attempts=1)
    first = model.generate_until([request(model)])[0]
    path = Path(first.metadata["record_path"])
    old = json.loads(path.read_text())
    resumed, session = setup(
        [FakeResponse(status=524), FakeResponse()],
        resume=True,
        timeout=120,
        max_attempts=2,
    )
    again = resumed.generate_until([request(resumed)])[0]
    new = json.loads(path.read_text())
    assert again.metadata["record_path"] == str(path)
    assert again.metadata["status"] == "ok"
    assert again.metadata["budget_attempts"] == 3
    assert new["attempts"][:1] == old["attempts"]
    assert [a["attempt"] for a in new["attempts"]] == [1, 2, 3]
    assert new["deadline_windows"][-1]["attempt_start"] == 1
    assert new["deadline_windows"][-1]["max_attempts"] == 2
    assert new["deadline_windows"][-1]["timeout"] == 120
    assert len(session.payloads) == 2


def test_finished_failure_without_explicit_resume_is_not_retried(setup):
    model, _ = setup([FakeResponse(status=524)], max_attempts=1)
    first = model.generate_until([request(model)])[0]
    reopened, session = setup([], timeout=120, max_attempts=2)
    again = reopened.generate_until([request(reopened)])[0]
    assert again.metadata["record_path"] == first.metadata["record_path"]
    assert again.metadata["status"] == "transport_failed"
    assert session.payloads == []


@pytest.mark.parametrize("finish", ["length", "content_filter"])
def test_explicit_resume_keeps_refusal_and_truncation(setup, finish):
    model, _ = setup([FakeResponse(body=json.dumps(response("text", finish)).encode())])
    first = model.generate_until([request(model)])[0]
    reopened, session = setup([], resume=True)
    again = reopened.generate_until([request(reopened)])[0]
    assert again == first
    assert session.payloads == []


def test_explicit_resume_reopens_a_question_timeout(setup):
    model, _ = setup([FakeResponse(status=524)], timeout=1)
    first = model.generate_until([request(model)])[0]
    assert first.metadata["status"] == "question_timeout"
    reopened, session = setup([FakeResponse()], resume=True, timeout=60)
    again = reopened.generate_until([request(reopened)])[0]
    assert again.metadata["status"] == "ok"
    assert again.metadata["record_path"] == first.metadata["record_path"]
    assert len(session.payloads) == 1


def test_explicit_resume_preserves_interrupted_attempt_and_downtime_is_free(setup):
    model, _ = setup([FakeResponse(status=524)], max_attempts=1)
    first = model.generate_until([request(model)])[0]
    path = Path(first.metadata["record_path"])
    old = json.loads(path.read_text())
    old["finished"] = False
    old["status"] = "running"
    old["attempts"][-1]["status"] = "running"
    old["started_at"] -= 86400
    old["deadline_windows"][0]["started_at"] -= 86400
    path.write_text(json.dumps(old))
    reopened, _ = setup([FakeResponse()], resume=True, max_attempts=1)
    again = reopened.generate_until([request(reopened)])[0]
    record = json.loads(path.read_text())
    assert again.metadata["status"] == "ok"
    assert record["attempts"][0]["error"] == "ProcessInterrupted"
    assert again.metadata["budget_attempts"] == 2
    assert again.metadata["elapsed_seconds"] < 60


def test_explicit_resume_reuses_finished_answer_after_last_checkpoint(setup):
    model, _ = setup([FakeResponse()])
    first = model.generate_until([request(model)])[0]
    path = Path(first.metadata["record_path"])
    record = json.loads(path.read_text())
    record["finished"] = False
    path.write_text(json.dumps(record))
    reopened, session = setup([], resume=True)
    again = reopened.generate_until([request(reopened)])[0]
    assert again.text == first.text
    assert session.payloads == []
    assert json.loads(path.read_text())["finished"]


@pytest.mark.parametrize(
    "setting", ["reasoning_effort", "max_tokens", "adapter_sha256"]
)
def test_explicit_resume_rejects_incompatible_experiment_before_network(setup, setting):
    model, _ = setup([FakeResponse()])
    model.generate_until([request(model)])
    reopened, session = setup([], resume=True)
    reopened.config[setting] = "changed"
    with pytest.raises(ValueError, match="Resume.*mismatch"):
        reopened.generate_until([request(reopened)])
    assert session.payloads == []


def test_explicit_resume_preserves_independent_repeat_indices(setup):
    model, _ = setup(
        [FakeResponse(status=524), FakeResponse()], max_attempts=1, num_concurrent=1
    )
    req = request(model)
    before = model.generate_until([req, req])
    reopened, session = setup([FakeResponse()], resume=True, num_concurrent=1)
    req = request(reopened)
    after = reopened.generate_until([req, req])
    assert [out.metadata["repeat_index"] for out in after] == [0, 1]
    assert [out.metadata["status"] for out in after] == ["ok", "ok"]
    assert after[1].text == before[1].text
    assert len(session.payloads) == 1


def test_resume_reads_pre_change_records_without_renaming_them(setup):
    model, _ = setup([FakeResponse()])
    first = model.generate_until([request(model)])[0]
    path = Path(first.metadata["record_path"])
    record = json.loads(path.read_text())
    record["identity"]["config"].update(
        timeout=3600,
        max_attempts=3,
        adapter_sha256="ef22aedfa4e9f6233429740cfc74f06724c6aedd6b9824816315ce4d22a14221",
    )
    old_path = path.parent / f"{digest(record['identity'])}.json"
    path.unlink()
    old_path.write_text(json.dumps(record))
    before = old_path.read_bytes()
    reopened, session = setup([], resume=True, timeout=120, max_attempts=1)
    again = reopened.generate_until([request(reopened)])[0]
    assert again.metadata["record_path"] == str(old_path)
    assert old_path.read_bytes() == before
    assert session.payloads == []


def test_explicit_resume_after_credentials_are_repaired(setup):
    model, _ = setup([FakeResponse(status=401)], max_attempts=1)
    with pytest.raises(PermissionError):
        model.generate_until([request(model)])
    reopened, session = setup([FakeResponse()], resume=True)
    again = reopened.generate_until([request(reopened)])[0]
    assert again.metadata["status"] == "ok"
    assert len(session.payloads) == 1


def test_resume_rejects_duplicate_question_records_before_network(setup):
    model, _ = setup([FakeResponse()])
    first = model.generate_until([request(model)])[0]
    path = Path(first.metadata["record_path"])
    path.with_name("f" * 64 + ".json").write_bytes(path.read_bytes())
    reopened, session = setup([], resume=True)
    with pytest.raises(ValueError, match="multiple records"):
        reopened.generate_until([request(reopened)])
    assert session.payloads == []


def test_resume_validates_all_saved_questions_before_starting_pending_calls(setup):
    model, _ = setup([FakeResponse()])
    model.generate_until([request(model, doc_id=1)])
    reopened, session = setup([], resume=True)
    changed = request(reopened, doc_id=1)
    changed.doc["new_label"] = "changed dataset"
    with pytest.raises(ValueError, match="Resume.*mismatch"):
        reopened.generate_until([request(reopened, doc_id=2), changed])
    assert session.payloads == []
