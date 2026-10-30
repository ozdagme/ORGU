from __future__ import annotations

import statistics


def multilabel_metrics(
    gold_sets: list[set[str]],
    predicted_sets: list[set[str]],
    labels: list[str],
) -> dict:
    if len(gold_sets) != len(predicted_sets):
        raise ValueError("Gold and prediction counts differ")
    per_class = {}
    total_tp = total_fp = total_fn = 0
    f1_values = []
    for label in labels:
        tp = sum(label in gold and label in predicted for gold, predicted in zip(gold_sets, predicted_sets))
        fp = sum(label not in gold and label in predicted for gold, predicted in zip(gold_sets, predicted_sets))
        fn = sum(label in gold and label not in predicted for gold, predicted in zip(gold_sets, predicted_sets))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": tp + fn,
            "predicted": tp + fp,
            "true_positive": tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn
        f1_values.append(f1)
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    exact = (
        sum(gold == predicted for gold, predicted in zip(gold_sets, predicted_sets)) / len(gold_sets)
        if gold_sets
        else 0.0
    )
    return {
        "candidate_count": len(gold_sets),
        "accepted_triples": sum(map(len, predicted_sets)),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_f1_fixed_inventory": statistics.mean(f1_values) if f1_values else 0.0,
        "exact_set_match": exact,
        "per_class": per_class,
    }