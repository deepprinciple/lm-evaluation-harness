import json
import pickle

import pytest
from datasets import Dataset

from lm_eval.api.instance import GenerationResult, Instance
from lm_eval.api.model import LM
from lm_eval.api.task import ConfigurableTask
from lm_eval.evaluator import evaluate


def make_instance(request_type="generate_until"):
    return Instance(request_type, {}, ("question", {}), 0, ("toy", 0, 2))


def test_metadata_tracks_mixed_repeated_responses():
    request = make_instance()
    request.add_response("first")
    request.add_response(
        GenerationResult("second", {"status": "ok", "usage": {"output_tokens": 7}})
    )
    assert request.resps == ["first", "second"]
    assert request.response_metadata == [
        None,
        {"status": "ok", "usage": {"output_tokens": 7}},
    ]
    restored = pickle.loads(pickle.dumps(request))
    assert restored.resps == request.resps
    assert restored.response_metadata == request.response_metadata


def test_request_cache_without_new_field_remains_usable():
    request = make_instance()
    request.resps = ["old response"]
    del request.response_metadata
    restored = pickle.loads(pickle.dumps(request))
    restored.add_response(GenerationResult("new response", {"status": "ok"}))
    assert restored.resps == ["old response", "new response"]
    assert restored.response_metadata == [None, {"status": "ok"}]


@pytest.mark.parametrize("request_type", ["loglikelihood", "loglikelihood_rolling"])
def test_likelihood_responses_unchanged(request_type):
    request = make_instance(request_type)
    response = (-1.25, False) if request_type == "loglikelihood" else -1.25
    request.add_response(response)
    assert request.resps == [response]
    assert request.response_metadata == [None]
    with pytest.raises(ValueError, match="generate_until"):
        request.add_response(GenerationResult("answer"))
    assert request.resps == [response]


def test_failed_attempt_keeps_diagnostic_metadata():
    request = make_instance()
    request.add_response(
        GenerationResult("", {"status": "incomplete", "trace_path": "raw/attempt.json"})
    )
    assert request.resps == [""]
    assert request.response_metadata == [
        {"status": "incomplete", "trace_path": "raw/attempt.json"}
    ]


def score_answer(doc, results):
    assert isinstance(results[0], str)
    return {"acc": float(results[0] == doc["answer"])}


class ToyTask(ConfigurableTask):
    def download(self, *args, **kwargs):
        self.dataset = {
            "test": Dataset.from_list(
                [{"question": f"Question {i}", "answer": "A"} for i in range(10)]
            )
        }


class ToyLM(LM):
    def __init__(self, structured):
        super().__init__()
        self.structured = structured

    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        outputs = []
        for i, request in enumerate(requests):
            answer = "A" if request.doc_id % 2 == 0 else "B"
            text = f"<answer>{answer}</answer>"
            if self.structured:
                outputs.append(
                    GenerationResult(
                        text,
                        {
                            "status": "ok",
                            "usage": {"output_tokens": i + 1},
                            "trace_path": f"raw/{i}.json",
                        },
                    )
                )
            else:
                outputs.append(text)
        return outputs


def run_toy(structured):
    task = ToyTask(
        config={
            "task": "toy",
            "test_split": "test",
            "output_type": "generate_until",
            "doc_to_text": "question",
            "doc_to_target": "answer",
            "num_fewshot": 0,
            "repeats": 2,
            "process_results": score_answer,
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
    return evaluate(
        ToyLM(structured), {"toy": task}, bootstrap_iters=0, log_samples=True
    )


def test_evaluator_preserves_scores_and_logs_metadata_per_repeat():
    legacy = run_toy(False)
    structured = run_toy(True)
    assert legacy["results"] == structured["results"]
    assert structured["results"]["toy"]["acc,xml"] == 0.5
    assert len(structured["samples"]["toy"]) == 10
    for i, (old, new) in enumerate(
        zip(legacy["samples"]["toy"], structured["samples"]["toy"])
    ):
        assert "response_metadata" not in old
        assert new["resps"] == old["resps"]
        assert new["filtered_resps"] == old["filtered_resps"]
        assert new["response_metadata"] == [
            [
                {
                    "status": "ok",
                    "usage": {"output_tokens": 2 * i + 1},
                    "trace_path": f"raw/{2 * i}.json",
                },
                {
                    "status": "ok",
                    "usage": {"output_tokens": 2 * i + 2},
                    "trace_path": f"raw/{2 * i + 1}.json",
                },
            ]
        ]
        assert (
            json.loads(json.dumps(new))["response_metadata"] == new["response_metadata"]
        )
