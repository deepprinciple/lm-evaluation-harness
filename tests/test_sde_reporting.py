"""Offline checks for independent SDE draws and explicit score denominators."""

import json
from collections import defaultdict
from copy import copy, deepcopy

import pytest
from datasets import Dataset

from lm_eval import evaluator
from lm_eval.__main__ import check_argument_types, setup_parser
from lm_eval.api.instance import GenerationResult
from lm_eval.api.model import LM
from lm_eval.api.task import ConfigurableTask
from lm_eval.evaluator import evaluate
from lm_eval.loggers import EvaluationTracker
from lm_eval.sde_reporting import collect_sde_results


def score(doc, results):
    return {"acc": float(results[0] == doc["answer"])}


class ToyTask(ConfigurableTask):
    def download(self, *args, **kwargs):
        self.dataset = {
            "test": Dataset.from_list(
                [{"question": f"Question {i}", "answer": "A"} for i in range(10)]
            )
        }


def make_task(repeats=6, process_results=score, metric="acc", higher=True):
    return ToyTask(
        config={
            "task": "toy",
            "test_split": "test",
            "output_type": "generate_until",
            "doc_to_text": "question",
            "doc_to_target": "answer",
            "num_fewshot": 0,
            "repeats": repeats,
            "process_results": process_results,
            "metric_list": [
                {"metric": metric, "aggregation": "mean", "higher_is_better": higher}
            ],
            "filter_list": [
                {
                    "name": name,
                    "filter": [
                        {
                            "function": "regex",
                            "regex_pattern": "<answer>(.*?)</answer>",
                        },
                        {"function": "take_first"},
                    ],
                }
                for name in ("xml", "also_xml")
            ],
        }
    )


class ToyLM(LM):
    sde_recorded_results = True

    def __init__(self, failure_status=None):
        super().__init__()
        self.failure_status = failure_status

    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        draws: dict[tuple[int, int], int] = defaultdict(int)
        outputs = []
        for request in requests:
            key = (request.doc_id, request.idx)
            repeat = draws[key]
            draws[key] += 1
            status = self.failure_status or "ok"
            if repeat == 1 and request.doc_id >= 6:
                status = "refusal"
            answer = "A" if request.doc_id < repeat + 4 else "B"
            outputs.append(
                GenerationResult(
                    f"<answer>{answer}</answer>",
                    {"status": status, "repeat_index": repeat},
                )
            )
        return outputs


def run(task=None, model=None, **kwargs):
    return evaluate(
        model or ToyLM(),
        {"toy": task or make_task()},
        bootstrap_iters=0,
        log_samples=True,
        **kwargs,
    )


def test_six_draws_use_original_filters_and_preserve_legacy_table():
    task = make_task()
    result = run(task=task)
    report = result["sde"]["tasks"]["toy"]
    assert result["results"]["toy"]["acc,xml"] == 0.4
    assert [r["metrics_all_documents"]["acc,xml"] for r in report["repeats"]] == [
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    ]
    second = report["repeats"][1]
    assert second["completed_documents"] == 6
    assert second["metrics_completed_documents"]["acc,xml"] == pytest.approx(5 / 6)
    assert second["completed_doc_ids"] == list(range(6))
    assert report["request_status_counts"] == {"ok": 56, "refusal": 4}
    assert report["across_repeats"]["metrics_all_documents"]["acc,xml"][
        "mean"
    ] == pytest.approx(0.65)
    assert report["across_repeats"]["metrics_all_documents"]["acc,xml"]["std"] > 0
    assert len(result["samples"]["toy"][0]["repeat_results"]) == 6
    assert len(task.instances[0].resps) == 6
    assert task.instances[0].filtered_resps["xml"] == "A"
    assert json.loads(json.dumps(result))["sde"] == result["sde"]


@pytest.mark.parametrize(
    "status", ["refusal", "truncated", "question_timeout", "transport_error"]
)
def test_incomplete_correct_text_cannot_earn_credit(status):
    report = run(task=make_task(1), model=ToyLM(status))["sde"]["tasks"]["toy"]
    repeat = report["repeats"][0]
    assert repeat["status_counts"] == {status: 10}
    assert repeat["metrics_all_documents"]["acc,xml"] == 0
    assert repeat["metrics_completed_documents"]["acc,xml"] is None
    assert repeat["completed_doc_ids"] == []


def test_normal_but_unparseable_answer_stays_in_completed_denominator():
    task = make_task(1)
    run(task=task)
    task.instances[0].resps = ["no answer tag"]
    report, rows = collect_sde_results(task)
    assert rows[0][0]["completed"] is True
    assert rows[0][0]["metrics"]["acc,xml"] == 0
    assert report["repeats"][0]["completed_documents"] == 10


@pytest.mark.parametrize(
    "bad_metadata",
    [[], [None], [{"status": "ok", "repeat_index": 2}], [{"repeat_index": 0}]],
)
def test_missing_or_misaligned_metadata_is_rejected(bad_metadata):
    task = make_task(1)
    run(task=task)
    task.instances[0].response_metadata = bad_metadata
    with pytest.raises(ValueError, match="metadata"):
        collect_sde_results(task)


def test_status_summary_is_available_without_sample_logging():
    result = evaluate(
        ToyLM(), {"toy": make_task(1)}, bootstrap_iters=0, log_samples=False
    )
    assert "samples" not in result
    assert result["sde"]["tasks"]["toy"]["repeats"][0]["documents"] == 10


def test_recorded_evaluation_rejects_stale_request_cache_before_generation():
    with pytest.raises(ValueError, match="request cache"):
        run(cache_requests=True)


def test_generic_model_keeps_existing_output_schema():
    model = ToyLM()
    model.sde_recorded_results = False
    assert "sde" not in run(model=model, task=make_task(1))


def test_bypass_reports_status_without_scoring():
    task = make_task(1)
    task.override_metric("bypass")
    report = run(task=task)["sde"]["tasks"]["toy"]
    assert report["repeats"][0]["completed_documents"] == 10
    assert report["repeats"][0]["metrics_all_documents"] == {}


def test_multiple_requests_require_all_responses_to_complete():
    task = make_task(1)
    run(task=task)
    extra = copy(task.instances[0])
    extra.idx = 1
    extra.response_metadata = [{"status": "transport_error", "repeat_index": 0}]
    task._instances.append(extra)
    report, rows = collect_sde_results(task)
    assert report["repeats"][0]["completed_documents"] == 9
    assert report["request_status_counts"] == {"ok": 10, "transport_error": 1}
    assert rows[0][0]["statuses"] == ["ok", "transport_error"]
    assert rows[0][0]["metrics"]["acc,xml"] == 0


def test_custom_metric_name_uses_evaluator_mean_fallback():
    task = make_task(1)
    run(task=task)
    task.config.process_results = lambda doc, results: {
        "custom_acc": float(results[0] == doc["answer"])
    }
    report, _ = collect_sde_results(task)
    assert report["repeats"][0]["metrics_all_documents"]["custom_acc,xml"] == 0.4


def test_original_task_aggregation_is_used():
    task = make_task(1)
    run(task=task)
    task._aggregation_list["acc"] = max
    report, _ = collect_sde_results(task)
    assert report["repeats"][0]["metrics_all_documents"]["acc,xml"] == 1


def test_conditional_metric_keys_cannot_silently_change_denominator():
    task = make_task(1)
    run(task=task)
    task.config.process_results = lambda doc, results: (
        {} if doc["question"] == "Question 0" else {"acc": 1}
    )
    with pytest.raises(ValueError, match="every document"):
        collect_sde_results(task)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), [1, 2]])
def test_non_accuracy_values_fail_explicitly(value):
    task = make_task(1)
    run(task=task)
    task.config.process_results = lambda doc, results: {"acc": value}
    with pytest.raises(ValueError, match="finite scalar"):
        collect_sde_results(task)


def test_report_does_not_mutate_original_responses_or_filters():
    task = make_task(2)
    run(task=task)
    before = deepcopy(
        [
            (req.resps, req.response_metadata, req.filtered_resps, req.doc)
            for req in task.instances
        ]
    )
    collect_sde_results(task)
    assert [
        (req.resps, req.response_metadata, req.filtered_resps, req.doc)
        for req in task.instances
    ] == before


def test_selected_documents_keep_original_ids():
    result = run(task=make_task(1), samples={"toy": [2, 4, 8]})
    report = result["sde"]["tasks"]["toy"]
    assert report["repeats"][0]["completed_doc_ids"] == [2, 4, 8]
    assert [
        row["repeat_results"][0]["doc_id"] for row in result["samples"]["toy"][:3]
    ] == [2, 4, 8]


def test_tracker_writes_summary_and_per_repeat_scores(tmp_path):
    result = run(task=make_task(2))
    tracker = EvaluationTracker(output_path=str(tmp_path))
    tracker.general_config_tracker.model_name_sanitized = "fake_model"
    samples = result.pop("samples")
    tracker.save_results_aggregated(result, samples)
    tracker.save_results_samples("toy", samples["toy"])
    aggregated = list(tmp_path.rglob("results_*.json"))
    sample_files = list(tmp_path.rglob("samples_*.jsonl"))
    assert len(aggregated) == len(sample_files) == 1
    assert json.loads(aggregated[0].read_text())["sde"] == result["sde"]
    rows = [json.loads(line) for line in sample_files[0].read_text().splitlines()]
    assert len(rows[0]["repeat_results"]) == 2


def test_repeats_parser_is_optional_and_typed():
    parser = setup_parser()
    check_argument_types(parser)
    assert parser.parse_args([]).repeats is None
    assert parser.parse_args(["--repeats", "6"]).repeats == 6


@pytest.mark.parametrize("repeats", [0, -1, True, 1.5])
def test_bad_repeats_fail_before_model_initialization(repeats):
    with pytest.raises(ValueError, match="positive integer"):
        evaluator.simple_evaluate(model="missing-model", tasks=["toy"], repeats=repeats)


def test_repeat_override_rejects_cached_request_instances():
    with pytest.raises(ValueError, match="request cache"):
        evaluator.simple_evaluate(
            model="missing-model", tasks=["toy"], repeats=6, cache_requests=True
        )


@pytest.mark.parametrize("override, expected", [(None, 2), (6, 6)])
def test_override_reaches_group_leaves_and_records_effective_config(
    monkeypatch, override, expected
):
    task = make_task(2)
    monkeypatch.setattr(
        evaluator, "get_task_dict", lambda *args: {"group": {"toy": task}}
    )
    monkeypatch.setattr(evaluator, "add_env_info", lambda results: None)
    monkeypatch.setattr(evaluator, "add_tokenizer_info", lambda results, lm: None)
    results = evaluator.simple_evaluate(
        model=ToyLM(),
        tasks=["toy"],
        task_manager=object(),
        repeats=override,
        bootstrap_iters=0,
    )
    assert results["configs"]["toy"]["repeats"] == expected
    assert results["sde"]["tasks"]["toy"]["repeat_count"] == expected
    assert len(results["samples"]["toy"][0]["resps"][0]) == expected
    assert results["config"].get("repeats") == override


def absolute_error(doc, results):
    return {"mae": float(results[0] != doc["answer"])}


def test_lower_is_better_metrics_never_give_incomplete_documents_zero_error():
    task = make_task(2, absolute_error, metric="mae", higher=False)
    report = run(task=task)["sde"]["tasks"]["toy"]
    complete, partial = report["repeats"]
    assert complete["metrics_all_documents"]["mae,xml"] == pytest.approx(0.6)
    assert complete["all_documents_unscorable"] == []
    assert partial["status_counts"] == {"ok": 6, "refusal": 4}
    assert partial["metrics_all_documents"]["mae,xml"] is None
    assert partial["all_documents_unscorable"] == ["mae,also_xml", "mae,xml"]
    assert partial["metrics_completed_documents"]["mae,xml"] == pytest.approx(1 / 6)
    across = report["across_repeats"]["metrics_all_documents"]["mae,xml"]
    assert across["mean"] == pytest.approx(0.6) and across["scored_repeats"] == 1
