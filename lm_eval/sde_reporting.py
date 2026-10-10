"""Independent-draw accuracy summaries for recorded, single-rank SDE models."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from copy import copy, deepcopy
from numbers import Real
from statistics import mean, stdev
from typing import Any

from lm_eval.api.metrics import mean as metric_mean


def _number(value: Any) -> float:
    if not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("SDE repeat reporting requires finite scalar accuracy metrics")
    return float(value)


def collect_sde_results(
    task: Any, doc_id_map: dict[int, int] | None = None
) -> tuple[dict, dict[int, list[dict]]]:
    """Run the task's original filters/scorer once per draw, without mutating it.

    Incomplete documents receive zero in the all-document accuracy denominator;
    the completed-only denominator excludes them. No best-of-N selection occurs.
    This report supports the scalar accuracy/partial-credit metrics in SDE, not
    likelihood tasks or pooled metrics such as perplexity and corpus BLEU.
    """
    instances = task.instances
    if not instances or task.OUTPUT_TYPE != "generate_until":
        raise ValueError("SDE repeat reporting requires generate_until instances")
    repeat_count = instances[0].repeats
    if repeat_count < 1:
        raise ValueError("SDE repeat reporting requires positive repeats")
    for instance in instances:
        if (
            instance.repeats != repeat_count
            or len(instance.resps) != repeat_count
            or len(instance.response_metadata) != repeat_count
        ):
            raise ValueError("SDE response metadata and repeat counts must align")
        for index, metadata in enumerate(instance.response_metadata):
            if (
                not isinstance(metadata, dict)
                or metadata.get("repeat_index") != index
                or not isinstance(metadata.get("status"), str)
                or not metadata["status"]
            ):
                raise ValueError(
                    "SDE response metadata needs ordered repeat_index and status"
                )

    rows: dict[int, list[dict]] = defaultdict(list)
    repeats: list[dict[str, Any]] = []
    bypass = "bypass" in task.aggregation()
    for index in range(repeat_count):
        view = copy(task)
        view._instances = []
        grouped: dict[int, list[Any]] = defaultdict(list)
        for instance in instances:
            draw = copy(instance)
            draw.doc = deepcopy(instance.doc)
            draw.resps = [
                instance.resps[index]
                if instance.response_metadata[index]["status"] == "ok"
                else ""
            ]
            draw.response_metadata = [instance.response_metadata[index]]
            draw.filtered_resps = {}
            draw.repeats = 1
            view._instances.append(draw)
            grouped[draw.doc_id].append(draw)
        view.apply_filters()
        all_scores: dict[str, list[float]] = defaultdict(list)
        completed_scores: dict[str, list[float]] = defaultdict(list)
        document_statuses: Counter = Counter()
        completed_ids = []
        for doc_id, requests in sorted(grouped.items()):
            original_doc_id = doc_id_map[doc_id] if doc_id_map is not None else doc_id
            requests.sort(key=lambda request: request.idx)
            statuses = [request.response_metadata[0]["status"] for request in requests]
            completed = all(status == "ok" for status in statuses)
            failures = set(statuses) - {"ok"}
            status = (
                "ok"
                if completed
                else next(iter(failures))
                if len(failures) == 1
                else "mixed_incomplete"
            )
            document_statuses[status] += 1
            if completed:
                completed_ids.append(original_doc_id)
            row: dict[str, Any] = {
                "doc_id": original_doc_id,
                "repeat_index": index,
                "statuses": statuses,
                "completed": completed,
                "filtered_resps": {},
                "metrics": {},
            }
            for filter_name in requests[0].filtered_resps:
                answers = [request.filtered_resps[filter_name] for request in requests]
                row["filtered_resps"][filter_name] = answers
                if bypass:
                    continue
                scores = view.process_results(requests[0].doc, answers)
                for metric, value in scores.items():
                    key = f"{metric},{filter_name}"
                    value = _number(value) if completed else 0.0
                    row["metrics"][key] = value
                    all_scores[key].append(value)
                    if completed:
                        completed_scores[key].append(value)
            rows[doc_id].append(row)

        if any(len(values) != len(grouped) for values in all_scores.values()):
            raise ValueError("SDE metric keys must cover every document in each repeat")

        def aggregate(scores: dict[str, list[float]], keys: list[str]) -> dict:
            return {
                key: _number(
                    task.aggregation().get(key.split(",", 1)[0], metric_mean)(values)
                )
                if (values := scores.get(key))
                else None
                for key in keys
            }

        repeats.append(
            {
                "repeat_index": index,
                "documents": len(grouped),
                "completed_documents": len(completed_ids),
                "completed_doc_ids": completed_ids,
                "status_counts": dict(sorted(document_statuses.items())),
                "metrics_all_documents": aggregate(all_scores, list(all_scores)),
                "metrics_completed_documents": aggregate(
                    completed_scores, list(all_scores)
                ),
            }
        )
    across: dict[str, dict] = {}
    for denominator in ("metrics_all_documents", "metrics_completed_documents"):
        across[denominator] = {}
        for key in repeats[0][denominator]:
            values = [
                repeat[denominator][key]
                for repeat in repeats
                if repeat[denominator][key] is not None
            ]
            across[denominator][key] = {
                "mean": mean(values) if values else None,
                "std": stdev(values) if len(values) > 1 else None,
                "scored_repeats": len(values),
            }
    summary = {
        "repeat_count": repeat_count,
        "request_status_counts": dict(
            sorted(
                Counter(
                    metadata["status"]
                    for instance in instances
                    for metadata in instance.response_metadata
                ).items()
            )
        ),
        "repeats": repeats,
        "across_repeats": across,
    }
    return summary, dict(rows)
