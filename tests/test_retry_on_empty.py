"""Some providers return HTTP 200 with empty `content` instead of an error.

An empty string is not a model answer: the filter fails to extract anything and
the item is scored as wrong rather than re-requested. These tests pin down that
`amodel_call` re-requests on empty content and gives up gracefully.
"""

import asyncio

import pytest

from lm_eval.models.api_models import JsonChatStr
from lm_eval.models.openai_completions import DeepSeekChatCompletion


def _response(content):
    return {"choices": [{"index": 0, "message": {"content": content}}]}


class FakeResponse:
    ok = True

    def __init__(self, payload):
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    async def json(self):
        return self._payload


class FakeSession:
    """Returns the queued payloads in order, counting the requests made."""

    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.calls = 0

    def post(self, url, json=None, headers=None):
        self.calls += 1
        payload = self._payloads[min(self.calls - 1, len(self._payloads) - 1)]
        return FakeResponse(payload)


def _call(session, **kwargs):
    lm = DeepSeekChatCompletion(
        model="deepseek-reasoner",
        tokenizer_backend=None,
        tokenized_requests=False,
        max_retries=3,
        **kwargs,
    )
    lm._api_key = "test"
    return asyncio.run(
        lm.amodel_call(
            session=session,
            messages=[JsonChatStr('[{"role": "user", "content": "hi"}]')],
            generate=True,
            cache_keys=None,
            gen_kwargs={"max_tokens": 16, "until": ["</s>"]},
        )
    )


def test_empty_content_is_retried_until_non_empty(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    session = FakeSession([_response(""), _response(""), _response("answer")])
    answers = _call(session)
    assert answers == ["answer"]
    assert session.calls == 3


def test_persistent_empty_content_does_not_raise(monkeypatch):
    """Exhausting the retries must not kill the whole run."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    session = FakeSession([_response("")])
    answers = _call(session)
    assert answers == [""]
    assert session.calls == 1 + 3  # initial request plus max_retries


def test_non_empty_content_makes_a_single_request(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    session = FakeSession([_response("answer")])
    assert _call(session) == ["answer"]
    assert session.calls == 1


def test_retry_on_empty_can_be_disabled(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    session = FakeSession([_response("")])
    assert _call(session, retry_on_empty=False) == [""]
    assert session.calls == 1
