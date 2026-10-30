from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zipfile

from analyze_loss_family_macro_ap import ARMS, SEEDS, exact_sign_flip_p, read_predictions

POLICY_KEYS = {
    "shared_0.5": "shared_threshold_0.5",
    "validation_macro_f1": "validation_macro_f1",
    "validation_recall_matched": "validation_recall_matched_to_NegDW_BCE_at_0.5",
}


def load_thresholds(bundle_path: Path) -> dict[str, dict[str, float]]:
    with zipfile.ZipFile(bundle_path) as archive:
        policy = json.loads(
            archive.read("runs/m1_m4_loss_family_v1/threshold_policy.json")
        )
    arm_keys = {
        "BCE": "BCE",
        "NegDW-BCE": "NegDW-BCE",
        "Focal": "Focal",
        "ASL": "ASL",
        "pos_weight-BCE": "pos_weight-BCE",
    }
    return {
        policy_name: {
            arm: float(policy["thresholds"][policy_key][arm_keys[arm]])
            for arm in arm_keys
        }
        for policy_name, policy_key in POLICY_KEYS.items()
    }


def macro_f1(targets: np.ndarray, predicted: np.ndarray) -> float:
    true_positive = np.sum(targets * predicted, axis=0)
    false_positive = np.sum((1 - targets) * predicted, axis=0)
    false_negative = np.sum(targets * (1 - predicted), axis=0)
    denominator = 2 * true_positive + false_positive + false_negative
    per_label = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator != 0,
    )
    return float(per_label.mean())


def clustered_counts(
    row_cluster: np.ndarray, values: np.ndarray, cluster_count: int
) -> np.ndarray:
    counts = np.zeros((cluster_count, values.shape[1]), dtype=np.float64)
    np.add.at(counts, row_cluster, values)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("results/m1_m4_loss_family_review_bundle.zip"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/analysis/loss_family_macro_f1_bootstrap.json"),
    )
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    scores, targets, row_cluster = read_predictions(args.bundle)
    thresholds = load_thresholds(args.bundle)
    arm_names = list(ARMS)
    arm_count, seed_count, _, label_count = scores.shape
    cluster_count = int(row_cluster.max()) + 1
    predicted = np.empty(
        (arm_count, seed_count, len(thresholds), targets.shape[0], label_count),
        dtype=np.uint8,
    )
    policy_names = list(thresholds)
    for arm_index, arm_name in enumerate(arm_names):
        for policy_index, policy_name in enumerate(policy_names):
            threshold = thresholds[policy_name][arm_name]
            predicted[arm_index, :, policy_index] = (
                scores[arm_index] >= threshold
            ).astype(np.uint8)

    point_values = np.empty(
        (seed_count, arm_count, len(policy_names)), dtype=np.float64
    )
    clustered_gold = clustered_counts(row_cluster, targets, cluster_count)
    clustered_predictions = np.empty(
        (seed_count, arm_count, len(policy_names), cluster_count, label_count),
        dtype=np.float64,
    )
    clustered_true_positives = np.empty_like(clustered_predictions)
    for seed_index in range(seed_count):
        for arm_index in range(arm_count):
            for policy_index in range(len(policy_names)):
                prediction = predicted[arm_index, seed_index, policy_index]
                clustered_predictions[seed_index, arm_index, policy_index] = (
                    clustered_counts(row_cluster, prediction, cluster_count)
                )
                clustered_true_positives[seed_index, arm_index, policy_index] = (
                    clustered_counts(
                        row_cluster, prediction * targets, cluster_count
                    )
                )
                point_values[seed_index, arm_index, policy_index] = macro_f1(
                    targets, prediction
                )

    rng = np.random.default_rng(args.seed)
    bootstrap_values = np.empty(
        (args.replicates, arm_count, len(policy_names)), dtype=np.float64
    )
    for batch_start in range(0, args.replicates, args.batch_size):
        batch_end = min(batch_start + args.batch_size, args.replicates)
        batch_count = batch_end - batch_start
        cluster_draws = rng.multinomial(
            cluster_count,
            np.full(cluster_count, 1.0 / cluster_count),
            size=batch_count,
        )
        sampled_seeds = rng.integers(0, seed_count, size=(batch_count, seed_count))
        gold_counts = cluster_draws @ clustered_gold
        per_seed = np.empty(
            (batch_count, seed_count, arm_count, len(policy_names)),
            dtype=np.float64,
        )
        for seed_index in range(seed_count):
            for arm_index in range(arm_count):
                for policy_index in range(len(policy_names)):
                    true_positive = cluster_draws @ clustered_true_positives[
                        seed_index, arm_index, policy_index
                    ]
                    predicted_positive = cluster_draws @ clustered_predictions[
                        seed_index, arm_index, policy_index
                    ]
                    denominator = gold_counts + predicted_positive
                    per_label_f1 = np.divide(
                        2 * true_positive,
                        denominator,
                        out=np.zeros_like(true_positive),
                        where=denominator != 0,
                    )
                    per_seed[:, seed_index, arm_index, policy_index] = (
                        per_label_f1.mean(axis=1)
                    )
        rows = np.arange(batch_count)[:, None]
        bootstrap_values[batch_start:batch_end] = per_seed[
            rows, sampled_seeds, :, :
        ].mean(axis=1)

    bce_index = arm_names.index("BCE")
    negdw_index = arm_names.index("NegDW-BCE")
    positive_index = arm_names.index("pos_weight-BCE")
    results = {}
    for policy_index, policy_name in enumerate(policy_names):
        policy_results = {}
        for arm_index, arm_name in enumerate(arm_names):
            values = point_values[:, arm_index, policy_index]
            difference = values - point_values[:, bce_index, policy_index]
            if arm_name == "BCE":
                difference_summary = None
            else:
                bootstrap_difference = (
                    bootstrap_values[:, arm_index, policy_index]
                    - bootstrap_values[:, bce_index, policy_index]
                )
                difference_summary = {
                    "mean": float(difference.mean()),
                    "sample_sd": float(difference.std(ddof=1)),
                    "percentile_95_ci": [
                        float(np.quantile(bootstrap_difference, 0.025)),
                        float(np.quantile(bootstrap_difference, 0.975)),
                    ],
                    "exact_seed_sign_flip_p_unadjusted": exact_sign_flip_p(difference),
                }
            policy_results[arm_name] = {
                "threshold": thresholds[policy_name][arm_name],
                "macro_f1_mean": float(values.mean()),
                "macro_f1_sample_sd": float(values.std(ddof=1)),
                "seed_values": values.tolist(),
                "difference_vs_bce": difference_summary,
            }
        paired_values = point_values[:, positive_index, policy_index] - point_values[
            :, negdw_index, policy_index
        ]
        paired_bootstrap = bootstrap_values[:, positive_index, policy_index] - (
            bootstrap_values[:, negdw_index, policy_index]
        )
        policy_results["pos_weight_minus_negdw"] = {
            "mean": float(paired_values.mean()),
            "sample_sd": float(paired_values.std(ddof=1)),
            "percentile_95_ci": [
                float(np.quantile(paired_bootstrap, 0.025)),
                float(np.quantile(paired_bootstrap, 0.975)),
            ],
            "exact_seed_sign_flip_p_unadjusted": exact_sign_flip_p(paired_values),
        }
        results[policy_name] = policy_results

    output = {
        "dataset": "FinRED entity-disjoint test",
        "candidates": int(targets.shape[0]),
        "source_record_clusters": cluster_count,
        "arms": arm_names,
        "seeds": list(SEEDS),
        "bootstrap": {
            "design": "paired two-way bootstrap over seeds and source-record clusters",
            "replicates": args.replicates,
            "rng": "NumPy default_rng (PCG64)",
            "rng_seed": args.seed,
            "batch_size": args.batch_size,
            "thresholds_frozen_to_validation_selected_values": True,
            "same_seed_and_cluster_draws_across_arms_and_policies": True,
        },
        "interpretation": "Post-hoc agreement with recorded annotations; not verified relation truth.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "results": results}, indent=2))


if __name__ == "__main__":
    main()