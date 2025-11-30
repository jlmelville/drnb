from __future__ import annotations

from typing import Any

from drnb.eval.base import evaluate_embedding
from drnb.experiment.common import (
    RUN_STATUS_COMPLETED,
    RUN_STATUS_MISSING,
    RUN_STATUS_PARTIAL_EVALS,
    eval_label_set,
    expected_eval_labels,
    labels_for_evaluator,
    merge_eval_results_by_label,
)
from drnb.experiment.persistence import LazyResult


def result_progress(
    res: dict | LazyResult | None, expected_labels: list[str]
) -> tuple[str, int, int, set[str]]:
    if isinstance(res, LazyResult):
        try:
            res = res.materialize()
        except FileNotFoundError:
            res = None
    expected_set = set(expected_labels)
    expected_count = len(expected_set)
    if res is None or "coords" not in res:
        return RUN_STATUS_MISSING, 0, expected_count, expected_set
    if expected_count == 0:
        return RUN_STATUS_COMPLETED, 0, 0, set()
    actual_labels = eval_label_set(res)
    completed = len(expected_set & actual_labels)
    if completed == expected_count:
        return RUN_STATUS_COMPLETED, completed, expected_count, set()
    missing = expected_set - actual_labels
    return RUN_STATUS_PARTIAL_EVALS, completed, expected_count, missing


def run_missing_evaluations(
    pipeline: Any,
    dataset: str,
    embed_result: dict,
    missing_labels: set[str],
    expected_labels: list[str],
) -> dict:
    x = pipeline.reader.read_data(dataset)[0]
    filtered_evaluators = [
        ev
        for ev in pipeline.evaluators
        if any(label in missing_labels for label in labels_for_evaluator(ev))
    ]
    if not filtered_evaluators:
        return embed_result
    new_evals = evaluate_embedding(
        filtered_evaluators, x, embed_result, ctx=embed_result.get("context")
    )
    existing_evals = embed_result.get("evaluations", [])
    merged = merge_eval_results_by_label(existing_evals, new_evals, expected_labels)
    embed_result["evaluations"] = merged
    return embed_result
