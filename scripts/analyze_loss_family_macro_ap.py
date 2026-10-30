from __future__ import annotations

import argparse
import itertools
import json
import zipfile
from pathlib import Path

import numpy as np

ARMS = {
    "BCE": "bce",
    "NegDW-BCE": "negdw_bce",
    "Focal": "focal",
    "ASL": "asl",
    "pos_weight-BCE": "pos_weight_bce",
}
SEEDS = (13, 42, 87, 101, 202, 303, 404, 505, 606, 707)


def average_precision(
    scores: np.ndarray, targets: np.ndarray, weights: np.ndarray | None = None
) -> float:
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_targets = targets[order].astype(np.float64)
    sorted_weights = (
        np.ones(len(order), dtype=np.float64)
        if weights is None
        else weights[order].astype(np.float64)
    )
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    positive_by_group = np.add.reduceat(sorted_weights * sorted_targets, starts)
    weight_by_group = np.add.reduceat(sorted_weights, starts)
    total_positive = positive_by_group.sum()
    if total_positive == 0:
        return 0.0
    return float(
        np.sum(
            (positive_by_group / total_positive)
            * (np.cumsum(positive_by_group) / np.cumsum(weight_by_group))
        )
    )


def exact_sign_flip_p(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    permuted = [
        abs(float(np.mean(values * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(permuted) >= observed - 1e-15))


def read_predictions(archive_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix = "runs/m1_m4_loss_family_v1/test_predictions"
    predictions: dict[str, list[np.ndarray]] = {arm: [] for arm in ARMS}
    reference_ids = None
    reference_targets = None
    reference_records = None
    with zipfile.ZipFile(archive_path) as archive:
        for arm, key in ARMS.items():
            for seed in SEEDS:
                member = f"{prefix}/{key}_seed{seed}.jsonl"
                rows = [
                    json.loads(line)
                    for line in archive.read(member).decode("utf-8").splitlines()
                ]
                rows.sort(key=lambda row: row["candidate_index"])
                candidate_ids = [row["candidate_id"] for row in rows]
                targets = np.asarray([row["gold_targets"] for row in rows], dtype=np.uint8)
                records = [row["record_id"] for row in rows]
                scores = np.asarray([row["scores"] for row in rows], dtype=np.float64)
                if reference_ids is None:
                    reference_ids, reference_targets, reference_records = (
                        candidate_ids,
                        targets,
                        records,
                    )
                elif (
                    candidate_ids != reference_ids
                    or not np.array_equal(targets, reference_targets)
                    or records != reference_records
                ):
                    raise ValueError(f"Evaluation rows differ in {member}")
                predictions[arm].append(scores)
    score_tensor = np.stack(
        [np.stack(predictions[arm], axis=0) for arm in ARMS], axis=0
    )
    record_values, row_cluster = np.unique(reference_records, return_inverse=True)
    if len(reference_ids) != 701 or len(record_values) != 638:
        raise ValueError(
            f"Expected 701 candidates and 638 clusters; got {len(reference_ids)} and "
            f"{len(record_values)}"
        )
    if score_tensor.shape[2:] != reference_targets.shape:
        raise ValueError("Score and target dimensions do not match")
    return score_tensor, reference_targets, row_cluster


def two_way_bootstrap(
    scores: np.ndarray,
    targets: np.ndarray,
    row_cluster: np.ndarray,
    replicates: int,
    random_seed: int,
    batch_size: int,
) -> np.ndarray:
    arm_count, seed_count, _, label_count = scores.shape
    cluster_count = int(row_cluster.max()) + 1
    rng = np.random.default_rng(random_seed)
    bootstrap_means = np.empty((replicates, arm_count), dtype=np.float64)
    ranked = []
    for seed_index in range(seed_count):
        ranked_seed = []
        for arm_index in range(arm_count):
            ranked_arm = []
            for label_index in range(label_count):
                order = np.argsort(
                    -scores[arm_index, seed_index, :, label_index], kind="mergesort"
                )
                sorted_scores = scores[arm_index, seed_index, order, label_index]
                starts = np.r_[
                    0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1
                ]
                ranked_arm.append(
                    (
                        row_cluster[order],
                        targets[order, label_index].astype(np.float64),
                        starts,
                    )
                )
            ranked_seed.append(ranked_arm)
        ranked.append(ranked_seed)

    for batch_start in range(0, replicates, batch_size):
        batch_end = min(batch_start + batch_size, replicates)
        batch_count = batch_end - batch_start
        cluster_draws = rng.multinomial(
            cluster_count,
            np.full(cluster_count, 1.0 / cluster_count),
            size=batch_count,
        )
        sampled_seeds = rng.integers(0, seed_count, size=(batch_count, seed_count))
        per_seed_means = np.zeros((batch_count, seed_count, arm_count), dtype=np.float64)
        for seed_index in range(seed_count):
            for arm_index in range(arm_count):
                arm_macro_ap = per_seed_means[:, seed_index, arm_index]
                for label_index in range(label_count):
                    clusters, sorted_targets, starts = ranked[seed_index][arm_index][
                        label_index
                    ]
                    sorted_weights = cluster_draws[:, clusters]
                    positive_by_group = np.add.reduceat(
                        sorted_weights * sorted_targets[None, :], starts, axis=1
                    )
                    weight_by_group = np.add.reduceat(
                        sorted_weights, starts, axis=1
                    )
                    total_positive = positive_by_group.sum(axis=1)
                    cumulative_positive = np.cumsum(positive_by_group, axis=1)
                    cumulative_weight = np.cumsum(weight_by_group, axis=1)
                    precision = np.divide(
                        cumulative_positive,
                        cumulative_weight,
                        out=np.zeros_like(cumulative_positive, dtype=np.float64),
                        where=cumulative_weight != 0,
                    )
                    label_ap = np.divide(
                        positive_by_group * precision,
                        total_positive[:, None],
                        out=np.zeros_like(positive_by_group, dtype=np.float64),
                        where=total_positive[:, None] != 0,
                    ).sum(axis=1)
                    arm_macro_ap += label_ap / label_count
        rows = np.arange(batch_count)[:, None]
        bootstrap_means[batch_start:batch_end] = per_seed_means[
            rows, sampled_seeds, :
        ].mean(axis=1)
    return bootstrap_means


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
        default=Path("results/analysis/loss_family_macro_ap_bootstrap.json"),
    )
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    scores, targets, row_cluster = read_predictions(args.bundle)
    arm_names = list(ARMS)
    seed_values = np.asarray(
        [
            [
                np.mean(
                    [
                        average_precision(
                            scores[arm_index, seed_index, :, label_index],
                            targets[:, label_index],
                        )
                        for label_index in range(targets.shape[1])
                    ]
                )
                for seed_index in range(len(SEEDS))
            ]
            for arm_index in range(len(arm_names))
        ],
        dtype=np.float64,
    )
    bootstrap = two_way_bootstrap(
        scores,
        targets,
        row_cluster,
        args.replicates,
        args.seed,
        args.batch_size,
    )
    negdw_index = arm_names.index("NegDW-BCE")
    results = {}
    for arm_index, arm_name in enumerate(arm_names):
        difference = seed_values[arm_index] - seed_values[negdw_index]
        difference_result = None
        if arm_name != "NegDW-BCE":
            bootstrap_difference = bootstrap[:, arm_index] - bootstrap[:, negdw_index]
            difference_result = {
                "mean": float(difference.mean()),
                "sample_sd": float(difference.std(ddof=1)),
                "percentile_95_ci": [
                    float(np.quantile(bootstrap_difference, 0.025)),
                    float(np.quantile(bootstrap_difference, 0.975)),
                ],
                "exact_seed_sign_flip_p_unadjusted": exact_sign_flip_p(difference),
                "seed_values": difference.tolist(),
            }
        results[arm_name] = {
            "macro_ap_mean": float(seed_values[arm_index].mean()),
            "macro_ap_sample_sd": float(seed_values[arm_index].std(ddof=1)),
            "seed_values": seed_values[arm_index].tolist(),
            "difference_vs_negdw": difference_result,
        }
    output = {
        "dataset": "FinRED entity-disjoint test",
        "candidates": int(targets.shape[0]),
        "labels": int(targets.shape[1]),
        "source_record_clusters": int(row_cluster.max()) + 1,
        "arms": arm_names,
        "seeds": list(SEEDS),
        "bootstrap": {
            "design": "paired two-way bootstrap over seeds and source-record clusters",
            "replicates": args.replicates,
            "rng": "NumPy default_rng (PCG64)",
            "rng_seed": args.seed,
            "interval": "two-sided percentile 95%",
            "same_seed_and_cluster_draws_for_all_arms": True,
            "thresholds_used": "none; raw scores for threshold-independent macro-AP",
        },
        "interpretation": "Post-hoc agreement with recorded annotations; not verified relation truth.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "results": results}, indent=2))


if __name__ == "__main__":
    main()