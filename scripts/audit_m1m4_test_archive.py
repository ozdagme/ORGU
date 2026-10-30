from __future__ import annotations

import hashlib
import itertools
import json
import math
import statistics
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np


OLD_TEST_PREDICTIONS = Path("results/archives/m1m4_test_predictions.tar.gz")
OLD_RUN_MANIFESTS = Path("results/archives/m1m4_run_manifests.tar.gz")
OLD_VALIDATION_ARCHIVE = Path("results/archives/m1m4_validation_artifacts.tar.gz")
OLD_VALIDATION_MANIFEST = Path("results/archives/m1m4_validation_manifest_2.json")
EXTRA_SEED_ARCHIVE = Path("results/archives/m1m4_extra_seed_artifacts_7seeds.tar.gz")
TRAIN_FILE = Path("data/features/finred_entity_disjoint_v1/C2/train.jsonl")
VALIDATION_FILE = Path("data/features/finred_entity_disjoint_v1/C2/validation.jsonl")
TEST_FILE = Path("data/features/finred_entity_disjoint_v1/C2/test.jsonl")
SPLIT_MANIFEST = Path("data/splits/finred_entity_disjoint_v1/split_manifest.json")
FEATURE_MANIFEST = Path("data/features/finred_entity_disjoint_v1/context_manifest.json")
LABEL_SCHEMA = Path("data/features/finred_entity_disjoint_v1/label_schema.json")
TOKENIZER = Path("data/features/finred_entity_disjoint_v1/tokenizer.json")
TRAINER = Path("src/orgu/train_encoder.py")
NEURAL = Path("src/orgu/neural.py")
METRICS = Path("src/orgu/metrics.py")
EXPORTER = Path("src/orgu/export_m1m4_test_predictions.py")
PRIOR_TEST_REPORT = Path("results/analysis/entity_disjoint/m1_m4_test_analysis.json")
PRIOR_VALIDATION_REPORT = Path("results/analysis/validation/m1_m4_validation_analysis.json")
OUTPUT_DIR = Path("results/analysis/entity_disjoint")

OLD_SEEDS = (13, 42, 87)
NEW_SEEDS = (101, 202, 303, 404, 505, 606, 707)
SEEDS = OLD_SEEDS + NEW_SEEDS
ARMS = ("BCE", "NegDW-BCE")
RUN_NAMES = tuple(f"{arm}_seed{seed}" for arm in ARMS for seed in SEEDS)
OLD_RUN_SPECS = {
    "BCE_seed13": ("BCE", 13, "L0", "9f1ebdeab15b77d56b012a22d606d79043af9524409749e9129ab7a27f45ff50", "1cf1eea6658494f05a7f6851e9b7c29ed44aa8d43a7d8d730ae8641590ab119f"),
    "BCE_seed42": ("BCE", 42, "L0", "64c10cfe7c5ded16d48fd741747f303b9a4fa9295eca0ae74738a968defc2093", "19dcd89af9b6baf456c7cdfd26b6723bbab99612bf291cda206303de121b5a03"),
    "BCE_seed87": ("BCE", 87, "L0", "032f6e9308a894e21b867540e7b1f53a1002d8538b46ae397ca19ff5490a276d", "42bfa490c6b339fa706216c4e60aa02add910cf3356070bb608ff22b7e89c6fb"),
    "NegDW-BCE_seed13": ("NegDW-BCE", 13, "L1", "389f69d5fe8e3e265e16467e84cce88d280e5561fc4c92c2db362faf8831aa2a", "5471ca0d727fe1e18f2a146f7389c0e85e2d8eeeebc5e49a9da40b132d0dc104"),
    "NegDW-BCE_seed42": ("NegDW-BCE", 42, "L1", "5478c7961354c2eb44417cc211ead578a3779ab9cda9d341798765e57cd6ebbe", "24fe0ff95bd0fb45ea7f1ed6cbb40d063f3805145cecaa1192182b78780a1f44"),
    "NegDW-BCE_seed87": ("NegDW-BCE", 87, "L1", "13399789624afc6165835c04fcbed3aba14b478ee057c13bc23525e597d2612c", "10ac359cf7278db12eb5749962e311b4d5a95708cf3c0b3b4eaf8160a6d5405f"),
}
OLD_ARCHIVE_HASHES = {
    OLD_TEST_PREDICTIONS: "7c6434066b1af4f2fb1c8f407c9ed004c5e963db79596f20b7db91f224341cd1",
    OLD_RUN_MANIFESTS: "a4f2babefc6821de312015fbf8c3b337d59f4f78d13965092e9ce0088fb4762d",
    OLD_VALIDATION_ARCHIVE: "7d247ecfd0d3820e1f3a1e4a54ed1acea61e81d4a8e2168f93165b0d7c308881",
}
EXTRA_ARCHIVE_SHA256 = "d6d9c78698bfc7caab0b387019499cc4cb0e568f86926d7a1744d8b0428263a7"
TEST_SHA256 = "bc8cc058fbd0ca75926063bfe7b9d0984746d494c53d406c797464cf246c233f"
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20261028
TWO_WAY_BOOTSTRAP_SEED = 20261029
SENSITIVITY_BOOTSTRAP_SEED = 20261030
TOLERANCE = 1e-12


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical_json_digest(path: Path) -> str:
    return digest(path.read_bytes().rstrip(b"\n") + b"\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(data: bytes, description: str) -> list[dict]:
    rows = []
    for line_number, line in enumerate(data.splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {description}:{line_number}: {exc}") from exc
            check(isinstance(row, dict), f"Expected object at {description}:{line_number}")
            rows.append(row)
    return rows


def archive_blobs(path: Path, expected_names: set[str]) -> dict[str, bytes]:
    blobs = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            item_path = PurePosixPath(member.name)
            check(not item_path.is_absolute() and ".." not in item_path.parts,
                  f"Unsafe archive path: {member.name}")
            check(member.isfile() and member.name not in blobs,
                  f"Unexpected or duplicate archive member: {member.name}")
            stream = archive.extractfile(member)
            check(stream is not None, f"Cannot read archive member: {member.name}")
            blobs[member.name] = stream.read()
    check(set(blobs) == expected_names, f"Unexpected archive contents: {path}")
    return blobs


def summary(values: list[float]) -> dict:
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}


def calculate_metrics(gold: np.ndarray, predicted: np.ndarray, labels: list[str]) -> dict:
    tp = np.logical_and(gold, predicted).sum(axis=0)
    fp = np.logical_and(np.logical_not(gold), predicted).sum(axis=0)
    fn = np.logical_and(gold, np.logical_not(predicted)).sum(axis=0)
    per_class = {}
    class_f1 = []
    for index, label in enumerate(labels):
        precision = tp[index] / (tp[index] + fp[index]) if tp[index] + fp[index] else 0.0
        recall = tp[index] / (tp[index] + fn[index]) if tp[index] + fn[index] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_f1.append(float(f1))
        per_class[label] = {
            "support": int(tp[index] + fn[index]),
            "predicted": int(tp[index] + fp[index]),
            "true_positive": int(tp[index]),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    tp_total, fp_total, fn_total = int(tp.sum()), int(fp.sum()), int(fn.sum())
    micro_p = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0
    micro_r = tp_total / (tp_total + fn_total) if tp_total + fn_total else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    return {
        "candidate_count": int(gold.shape[0]),
        "accepted_triples": int(predicted.sum()),
        "macro_f1_fixed_inventory": statistics.mean(class_f1),
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "exact_set_match": float(np.all(gold == predicted, axis=1).mean()),
        "predicted_labels_per_candidate": float(predicted.sum() / gold.shape[0]),
        "per_class": per_class,
    }


def calculate_macro_average_precision(gold: np.ndarray, scores: np.ndarray,
                                     labels: list[str]) -> dict:
    per_class = {}
    for index, label in enumerate(labels):
        positives = gold[:, index].astype(np.int64)
        support = int(positives.sum())
        if support == 0:
            per_class[label] = 0.0
            continue

        order = np.argsort(-scores[:, index], kind="mergesort")
        ranked_scores = scores[order, index]
        cumulative_positives = np.cumsum(positives[order])
        threshold_ends = np.concatenate((np.flatnonzero(np.diff(ranked_scores)),
                                         [len(ranked_scores) - 1]))
        true_positives = cumulative_positives[threshold_ends]
        predicted_positives = threshold_ends + 1
        precision = true_positives / predicted_positives
        recall = true_positives / support
        recall_increments = np.diff(np.concatenate(([0.0], recall)))
        per_class[label] = float(np.sum(recall_increments * precision))

    return {
        "macro_average_precision_fixed_inventory": statistics.mean(per_class.values()),
        "per_class_average_precision": per_class,
    }


def check_close(expected: float, observed: float, description: str) -> None:
    check(math.isclose(float(expected), float(observed), rel_tol=TOLERANCE, abs_tol=TOLERANCE),
          f"Metric mismatch at {description}: {expected} != {observed}")


def ordered_id_hash(rows: list[dict]) -> str:
    return digest("\n".join(str(row["id"]) for row in rows).encode())


def prediction_arrays(source_rows: list[dict], prediction_bytes: bytes, labels: list[str],
                      split_name: str, split_hash: str, description: str):
    predictions = read_jsonl(prediction_bytes, description)
    check(len(predictions) == len(source_rows), f"Prediction count mismatch: {description}")
    label_set = set(labels)
    gold_matrix = np.zeros((len(source_rows), len(labels)), dtype=np.bool_)
    predicted_matrix = np.zeros_like(gold_matrix)
    score_matrix = np.zeros((len(source_rows), len(labels)), dtype=np.float64)
    prediction_ids = []
    for index, (source, prediction) in enumerate(zip(source_rows, predictions)):
        candidate_id = str(source["id"])
        prediction_ids.append(str(prediction.get("candidate_id")))
        check(str(prediction.get("candidate_id")) == candidate_id,
              f"Candidate ID/order mismatch: {description}:{index}")
        if "candidate_index" in prediction:
            check(prediction["candidate_index"] == index,
                  f"Candidate index mismatch: {description}:{index}")
        check(prediction.get("record_id") == source.get("record_id"),
              f"Record ID mismatch: {description}:{index}")
        check(prediction.get("split") == split_name and prediction.get("split_sha256") == split_hash,
              f"Split/hash mismatch: {description}:{index}")
        gold = set(source.get("labels", []))
        check(gold and gold <= label_set, f"Invalid source gold labels: {description}:{index}")
        check(set(prediction.get("gold_labels", [])) == gold,
              f"Gold-label mismatch: {description}:{index}")
        target = [int(label in gold) for label in labels]
        check(prediction.get("gold_targets") == target,
              f"Gold-target vector mismatch: {description}:{index}")
        scores = prediction.get("scores")
        check(isinstance(scores, list) and len(scores) == len(labels),
              f"Score width mismatch: {description}:{index}")
        check(all(isinstance(score, (int, float)) and math.isfinite(float(score))
                  and 0 <= float(score) <= 1 for score in scores),
              f"Invalid score: {description}:{index}")
        predicted = [float(score) >= 0.5 for score in scores]
        if "threshold" in prediction:
            check(float(prediction["threshold"]) == 0.5,
                  f"Prediction threshold mismatch: {description}:{index}")
        if "predicted_labels" in prediction:
            predicted_labels = {label for label, active in zip(labels, predicted) if active}
            check(set(prediction["predicted_labels"]) == predicted_labels,
                  f"Thresholded-label mismatch: {description}:{index}")
        gold_matrix[index] = target
        predicted_matrix[index] = predicted
        score_matrix[index] = scores
    check(len(set(prediction_ids)) == len(prediction_ids),
          f"Duplicate prediction candidate: {description}")
    order_hash = digest("\n".join(prediction_ids).encode())
    check(order_hash == ordered_id_hash(source_rows), f"Candidate-order hash mismatch: {description}")
    return gold_matrix, predicted_matrix, score_matrix, order_hash


def threshold_candidates(score_arrays: list[np.ndarray]) -> np.ndarray:
    observed = np.unique(np.concatenate([scores.reshape(-1) for scores in score_arrays]))
    return np.unique(np.concatenate(([0.0], observed, [np.nextafter(observed[-1], math.inf)])))


def choose_mean_macro_f1_threshold(run_arrays: dict, arm: str, seeds: tuple[int, ...],
                                   labels: list[str]) -> tuple[float, float]:
    score_arrays = [run_arrays[f"{arm}_seed{seed}"][1] for seed in seeds]
    candidates = threshold_candidates(score_arrays)
    mean_macro_f1 = np.zeros(candidates.shape, dtype=np.float64)
    curve_count = len(seeds) * len(labels)

    for seed in seeds:
        gold, scores = run_arrays[f"{arm}_seed{seed}"]
        for label_index in range(len(labels)):
            column_scores = scores[:, label_index]
            column_gold = gold[:, label_index].astype(np.int64)
            order = np.argsort(column_scores, kind="mergesort")
            sorted_scores = column_scores[order]
            prefix_positive = np.concatenate(([0], np.cumsum(column_gold[order])))
            first_positive = np.searchsorted(sorted_scores, candidates, side="left")
            predicted = len(column_scores) - first_positive
            true_positive = prefix_positive[-1] - prefix_positive[first_positive]
            denominator = predicted + prefix_positive[-1]
            f1 = np.divide(2 * true_positive, denominator,
                           out=np.zeros_like(candidates), where=denominator != 0)
            mean_macro_f1 += f1 / curve_count

    best_value = float(mean_macro_f1.max())
    best_index = int(np.flatnonzero(mean_macro_f1 == best_value)[-1])
    return float(candidates[best_index]), best_value


def mean_micro_recall_at_threshold(run_arrays: dict, arm: str, seeds: tuple[int, ...],
                                   threshold: float) -> float:
    recalls = []
    for seed in seeds:
        gold, scores = run_arrays[f"{arm}_seed{seed}"]
        predicted = scores >= threshold
        true_positive = np.logical_and(gold, predicted).sum()
        support = gold.sum()
        recalls.append(float(true_positive / support) if support else 0.0)
    return statistics.mean(recalls)


def choose_recall_matched_threshold(run_arrays: dict, seeds: tuple[int, ...],
                                    target_recall: float) -> tuple[float, float]:
    score_arrays = [run_arrays[f"BCE_seed{seed}"][1] for seed in seeds]
    candidates = threshold_candidates(score_arrays)
    mean_recall = np.zeros(candidates.shape, dtype=np.float64)

    for seed in seeds:
        gold, scores = run_arrays[f"BCE_seed{seed}"]
        flat_scores = scores.reshape(-1)
        flat_gold = gold.reshape(-1).astype(np.int64)
        order = np.argsort(flat_scores, kind="mergesort")
        sorted_scores = flat_scores[order]
        prefix_positive = np.concatenate(([0], np.cumsum(flat_gold[order])))
        first_positive = np.searchsorted(sorted_scores, candidates, side="left")
        true_positive = prefix_positive[-1] - prefix_positive[first_positive]
        support = prefix_positive[-1]
        recall = true_positive / support if support else np.zeros_like(candidates)
        mean_recall += recall / len(seeds)

    absolute_error = np.abs(mean_recall - target_recall)
    best_error = float(absolute_error.min())
    best_index = int(np.flatnonzero(absolute_error == best_error)[-1])
    return float(candidates[best_index]), float(mean_recall[best_index])


def summarize_threshold_policy(run_arrays: dict, labels: list[str], seeds: tuple[int, ...],
                               thresholds: dict[str, float]) -> dict:
    metric_names = ("macro_f1_fixed_inventory", "micro_f1", "micro_precision", "micro_recall",
                    "exact_set_match", "predicted_labels_per_candidate")
    seed_results = {}
    for arm in ARMS:
        for seed in seeds:
            name = f"{arm}_seed{seed}"
            gold, scores = run_arrays[name]
            metrics = calculate_metrics(gold, scores >= thresholds[arm], labels)
            seed_results[name] = {metric: metrics[metric] for metric in metric_names}

    aggregate_metrics = {}
    for metric in metric_names:
        bce_values = [seed_results[f"BCE_seed{seed}"][metric] for seed in seeds]
        negdw_values = [seed_results[f"NegDW-BCE_seed{seed}"][metric] for seed in seeds]
        differences = [right - left for left, right in zip(bce_values, negdw_values)]
        aggregate_metrics[metric] = {
            "BCE": summary(bce_values),
            "NegDW-BCE": summary(negdw_values),
            "paired_difference_NegDW_minus_BCE": summary(differences),
            "seed_values": {"BCE": dict(zip(seeds, bce_values)),
                            "NegDW-BCE": dict(zip(seeds, negdw_values)),
                            "paired_difference": dict(zip(seeds, differences))},
        }
    return {"thresholds": thresholds, "seed_results": seed_results,
            "aggregate_metrics": aggregate_metrics}


def cluster_counts(run_arrays: dict, record_ids: list[str], labels: list[str]):
    clusters = sorted(set(record_ids))
    cluster_index = {cluster: index for index, cluster in enumerate(clusters)}
    row_cluster = [cluster_index[record_id] for record_id in record_ids]
    counts_by_run = []
    for run_name in RUN_NAMES:
        gold, predicted = run_arrays[run_name]
        counts = [np.zeros((len(clusters), len(labels)), dtype=np.int64) for _ in range(3)]
        for row_index, cluster_id in enumerate(row_cluster):
            counts[0][cluster_id] += np.logical_and(gold[row_index], predicted[row_index])
            counts[1][cluster_id] += np.logical_and(np.logical_not(gold[row_index]), predicted[row_index])
            counts[2][cluster_id] += np.logical_and(gold[row_index], np.logical_not(predicted[row_index]))
        counts_by_run.append(counts)
    tp, fp, fn = (np.stack([run[index] for run in counts_by_run]) for index in range(3))
    return clusters, tp, fp, fn


def macro_f1_by_run(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=np.float64), where=denominator > 0)
    return f1.mean(axis=1)


def bootstrap_intervals(run_arrays: dict, record_ids: list[str], labels: list[str]) -> tuple[dict, dict]:
    clusters, tp_by_run, fp_by_run, fn_by_run = cluster_counts(run_arrays, record_ids, labels)
    cluster_count = len(clusters)
    run_index = {name: index for index, name in enumerate(RUN_NAMES)}
    tp_flat = tp_by_run.transpose(1, 0, 2).reshape(cluster_count, -1)
    fp_flat = fp_by_run.transpose(1, 0, 2).reshape(cluster_count, -1)
    fn_flat = fn_by_run.transpose(1, 0, 2).reshape(cluster_count, -1)

    cluster_rng = np.random.default_rng(BOOTSTRAP_SEED)
    cluster_differences = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        weights = np.bincount(cluster_rng.integers(0, cluster_count, size=cluster_count), minlength=cluster_count)
        tp = (weights @ tp_flat).reshape(len(RUN_NAMES), len(labels))
        fp = (weights @ fp_flat).reshape(len(RUN_NAMES), len(labels))
        fn = (weights @ fn_flat).reshape(len(RUN_NAMES), len(labels))
        macro = macro_f1_by_run(tp, fp, fn)
        differences = [macro[run_index[f"NegDW-BCE_seed{s}"]] - macro[run_index[f"BCE_seed{s}"]] for s in SEEDS]
        cluster_differences[replicate] = statistics.mean(differences)
    cluster_lower, cluster_upper = np.quantile(cluster_differences, [0.025, 0.975], method="linear")
    cluster_result = {
        "unit": "source-record cluster",
        "cluster_count": cluster_count,
        "replicates": BOOTSTRAP_REPLICATES,
        "rng": "NumPy default_rng (PCG64)",
        "rng_seed": BOOTSTRAP_SEED,
        "statistic": "mean paired NegDW-BCE macro-F1 difference across ten seeds",
        "sampling": "clusters sampled with replacement; common resample for all ten seed pairs and both arms",
        "interval_method": "percentile 95% interval",
        "lower_95": float(cluster_lower),
        "upper_95": float(cluster_upper),
        "bootstrap_mean": float(cluster_differences.mean()),
    }

    joint_rng = np.random.default_rng(TWO_WAY_BOOTSTRAP_SEED)
    joint_differences = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        weights = np.bincount(joint_rng.integers(0, cluster_count, size=cluster_count), minlength=cluster_count)
        sampled_seed_indices = joint_rng.integers(0, len(SEEDS), size=len(SEEDS))
        paired = []
        for seed_index in sampled_seed_indices:
            seed = SEEDS[int(seed_index)]
            macro_by_arm = {}
            for arm in ARMS:
                run_name = f"{arm}_seed{seed}"
                index = run_index[run_name]
                tp = weights @ tp_by_run[index]
                fp = weights @ fp_by_run[index]
                fn = weights @ fn_by_run[index]
                macro_by_arm[arm] = float(macro_f1_by_run(tp[None, :], fp[None, :], fn[None, :])[0])
            paired.append(macro_by_arm["NegDW-BCE"] - macro_by_arm["BCE"])
        joint_differences[replicate] = statistics.mean(paired)
    joint_lower, joint_upper = np.quantile(joint_differences, [0.025, 0.975], method="linear")
    joint_result = {
        "units": ["paired training seed", "source-record cluster"],
        "seed_count": len(SEEDS),
        "cluster_count": cluster_count,
        "replicates": BOOTSTRAP_REPLICATES,
        "rng": "NumPy default_rng (PCG64)",
        "rng_seed": TWO_WAY_BOOTSTRAP_SEED,
        "statistic": "mean paired NegDW-BCE macro-F1 difference",
        "sampling": "ten paired seeds and source-record clusters sampled with replacement; shared samples across arms",
        "interval_method": "percentile 95% interval",
        "lower_95": float(joint_lower),
        "upper_95": float(joint_upper),
        "bootstrap_mean": float(joint_differences.mean()),
        "interpretation": "exploratory two-way bootstrap; ten paired seeds are available, but post-hoc test exposure remains",
    }
    return cluster_result, joint_result


def _cluster_confusion_arrays(run_arrays: dict, record_ids: list[str], labels: list[str],
                              thresholds: dict[str, float]) -> tuple[list[str], np.ndarray,
                                                                       np.ndarray, np.ndarray, np.ndarray]:
    clusters = sorted(set(record_ids))
    cluster_index = {cluster: index for index, cluster in enumerate(clusters)}
    row_cluster = np.asarray([cluster_index[record_id] for record_id in record_ids], dtype=np.int64)
    tp_by_run = np.zeros((len(RUN_NAMES), len(clusters), len(labels)), dtype=np.int64)
    fp_by_run = np.zeros_like(tp_by_run)
    fn_by_run = np.zeros_like(tp_by_run)
    for run_index, run_name in enumerate(RUN_NAMES):
        arm = run_name.rsplit("_seed", 1)[0]
        gold, scores = run_arrays[run_name]
        predicted = scores >= thresholds[arm]
        for row_index, cluster_id in enumerate(row_cluster):
            tp_by_run[run_index, cluster_id] += np.logical_and(gold[row_index], predicted[row_index])
            fp_by_run[run_index, cluster_id] += np.logical_and(np.logical_not(gold[row_index]), predicted[row_index])
            fn_by_run[run_index, cluster_id] += np.logical_and(gold[row_index], np.logical_not(predicted[row_index]))
    return clusters, row_cluster, tp_by_run, fp_by_run, fn_by_run


def _weighted_macro_ap(gold: np.ndarray, scores: np.ndarray, row_weights: np.ndarray) -> np.ndarray:
    replicate_count = row_weights.shape[0]
    relation_ap = np.zeros((replicate_count, gold.shape[1]), dtype=np.float64)
    for label_index in range(gold.shape[1]):
        order = np.argsort(-scores[:, label_index], kind="mergesort")
        ranked_scores = scores[order, label_index]
        group_ends = np.concatenate((np.flatnonzero(np.diff(ranked_scores)), [len(order) - 1]))
        ranked_weights = row_weights[:, order]
        cumulative_weights = np.cumsum(ranked_weights, axis=1)
        cumulative_positives = np.cumsum(ranked_weights * gold[order, label_index], axis=1)
        true_positives = cumulative_positives[:, group_ends]
        predicted_positives = cumulative_weights[:, group_ends]
        increments = np.diff(np.concatenate((np.zeros((replicate_count, 1)), true_positives), axis=1), axis=1)
        total_positives = cumulative_positives[:, -1]
        precision = np.divide(true_positives, predicted_positives,
                              out=np.zeros_like(true_positives, dtype=np.float64),
                              where=predicted_positives > 0)
        relation_ap[:, label_index] = np.divide(
            np.sum(increments * precision, axis=1), total_positives,
            out=np.zeros(replicate_count, dtype=np.float64), where=total_positives > 0)
    return relation_ap.mean(axis=1)


def two_way_posthoc_sensitivity_bootstrap(run_arrays: dict, record_ids: list[str], labels: list[str],
                                          policies: dict[str, dict[str, float]]) -> dict:
    """Jointly resample paired seeds and source-record clusters for frozen policies and macro-AP."""
    cluster_count = len(set(record_ids))
    run_index = {name: index for index, name in enumerate(RUN_NAMES)}
    seed_count = len(SEEDS)
    batch_size = 128
    rng = np.random.default_rng(SENSITIVITY_BOOTSTRAP_SEED)
    bootstrap_values = {
        "macro_average_precision_fixed_inventory": np.empty(BOOTSTRAP_REPLICATES),
        **{policy_name: np.empty(BOOTSTRAP_REPLICATES) for policy_name in policies},
    }

    policy_confusions = {}
    for policy_name, thresholds in policies.items():
        clusters, row_cluster, tp, fp, fn = _cluster_confusion_arrays(
            run_arrays, record_ids, labels, thresholds)
        check(len(clusters) == cluster_count, "Sensitivity bootstrap cluster-count mismatch")
        policy_confusions[policy_name] = tuple(
            counts.transpose(1, 0, 2).reshape(cluster_count, -1) for counts in (tp, fp, fn))

    ap_inputs = []
    for run_name in RUN_NAMES:
        gold, scores = run_arrays[run_name]
        ap_inputs.append((gold, scores))

    duplicated_rows = np.repeat(
        np.arange(len(record_ids)), np.where(row_cluster == 0, 2, 1))
    duplication_weights = np.where(row_cluster == 0, 2, 1)[None, :]
    for run_name, (gold, scores) in zip(RUN_NAMES, ap_inputs):
        weighted_ap = _weighted_macro_ap(gold, scores, np.ones((1, len(record_ids))))[0]
        ordinary_ap = calculate_macro_average_precision(gold, scores, labels)[
            "macro_average_precision_fixed_inventory"]
        check_close(ordinary_ap, weighted_ap, f"weighted macro-AP identity {run_name}")
        duplicated_ap = calculate_macro_average_precision(
            gold[duplicated_rows], scores[duplicated_rows], labels)[
                "macro_average_precision_fixed_inventory"]
        cluster_weighted_ap = _weighted_macro_ap(gold, scores, duplication_weights)[0]
        check_close(duplicated_ap, cluster_weighted_ap,
                    f"cluster-duplicated macro-AP identity {run_name}")

    for batch_start in range(0, BOOTSTRAP_REPLICATES, batch_size):
        batch_end = min(batch_start + batch_size, BOOTSTRAP_REPLICATES)
        current_size = batch_end - batch_start
        cluster_counts = rng.multinomial(
            cluster_count, np.full(cluster_count, 1.0 / cluster_count), size=current_size)
        sampled_seed_indices = rng.integers(0, seed_count, size=(current_size, seed_count))
        row_weights = cluster_counts[:, row_cluster]

        for policy_name, (tp_flat, fp_flat, fn_flat) in policy_confusions.items():
            tp = (cluster_counts @ tp_flat).reshape(current_size, len(RUN_NAMES), len(labels))
            fp = (cluster_counts @ fp_flat).reshape(current_size, len(RUN_NAMES), len(labels))
            fn = (cluster_counts @ fn_flat).reshape(current_size, len(RUN_NAMES), len(labels))
            denominator = 2 * tp + fp + fn
            macro_f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=np.float64),
                                 where=denominator > 0).mean(axis=2)
            seed_differences = np.column_stack([
                macro_f1[:, run_index[f"NegDW-BCE_seed{seed}"]]
                - macro_f1[:, run_index[f"BCE_seed{seed}"]]
                for seed in SEEDS
            ])
            bootstrap_values[policy_name][batch_start:batch_end] = np.take_along_axis(
                seed_differences, sampled_seed_indices, axis=1).mean(axis=1)

        ap_by_run = np.empty((current_size, len(RUN_NAMES)), dtype=np.float64)
        for run_position, (gold, scores) in enumerate(ap_inputs):
            ap_by_run[:, run_position] = _weighted_macro_ap(gold, scores, row_weights)
        ap_seed_differences = np.column_stack([
            ap_by_run[:, run_index[f"NegDW-BCE_seed{seed}"]]
            - ap_by_run[:, run_index[f"BCE_seed{seed}"]]
            for seed in SEEDS
        ])
        bootstrap_values["macro_average_precision_fixed_inventory"][batch_start:batch_end] = (
            np.take_along_axis(ap_seed_differences, sampled_seed_indices, axis=1).mean(axis=1))

    point_differences = {}
    for metric_name in bootstrap_values:
        if metric_name == "macro_average_precision_fixed_inventory":
            values_by_seed = [
                calculate_macro_average_precision(*run_arrays[f"NegDW-BCE_seed{seed}"], labels)[
                    "macro_average_precision_fixed_inventory"]
                - calculate_macro_average_precision(*run_arrays[f"BCE_seed{seed}"], labels)[
                    "macro_average_precision_fixed_inventory"]
                for seed in SEEDS
            ]
        else:
            thresholds = policies[metric_name]
            values_by_seed = []
            for seed in SEEDS:
                differences = []
                for arm in ARMS:
                    gold, scores = run_arrays[f"{arm}_seed{seed}"]
                    metrics = calculate_metrics(gold, scores >= thresholds[arm], labels)
                    differences.append(metrics["macro_f1_fixed_inventory"])
                values_by_seed.append(differences[1] - differences[0])
        point_differences[metric_name] = values_by_seed

    results = {}
    for metric_name, replicates in bootstrap_values.items():
        lower, upper = np.quantile(replicates, [0.025, 0.975], method="linear")
        sign_flip = exact_seed_sign_flip(point_differences[metric_name], metric_name)
        results[metric_name] = {
            "point_estimate": float(statistics.mean(point_differences[metric_name])),
            "sample_sd": float(statistics.stdev(point_differences[metric_name])),
            "seed_level_differences": dict(zip(SEEDS, point_differences[metric_name])),
            "two_way_bootstrap_95_percentile_ci": {
                "lower": float(lower), "upper": float(upper),
                "bootstrap_mean": float(replicates.mean()),
            },
            "exact_seed_sign_flip_test": sign_flip,
        }

    return {
        "method": "paired seed-and-source-record-cluster bootstrap with frozen validation-selected thresholds",
        "replicates": BOOTSTRAP_REPLICATES,
        "rng": "NumPy default_rng (PCG64)",
        "rng_seed": SENSITIVITY_BOOTSTRAP_SEED,
        "seed_count": seed_count,
        "cluster_count": cluster_count,
        "sampling": (
            "For each replicate, sample ten paired seed indices and 638 source-record clusters "
            "with replacement; use identical draws for both arms and all three metrics. Repeated "
            "clusters are represented by integer row weights, equivalent to duplicating all rows "
            "in each sampled cluster. Threshold policies remain frozen at their validation-selected values."
        ),
        "interval_method": "two-sided percentile 95% interval",
        "interpretation": "post-hoc exploratory; test previously inspected and seven seeds followed initial three-seed results",
        "metrics": results,
    }


def exact_seed_sign_flip(differences: list[float], metric_name: str = "macro_f1_fixed_inventory") -> dict:
    observed = statistics.mean(differences)
    assignments = 1 << len(differences)
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        randomized = statistics.mean(value * sign for value, sign in zip(differences, signs))
        if abs(randomized) >= abs(observed) - 1e-15:
            extreme += 1
    return {
        "method": f"exact two-sided sign-flip test over paired seed-level {metric_name} differences",
        "assignments": assignments,
        "observed_mean_difference": observed,
        "extreme_assignments": extreme,
        "p_value_two_sided": extreme / assignments,
        "interpretation": "exploratory; added seeds followed initial inspection of test results",
    }


def _manifest_entries(manifest: dict, list_key: str, description: str) -> dict:
    entries = manifest.get(list_key)
    check(isinstance(entries, list), f"Missing run entries: {description}")
    mapped = {(entry["loss"], int(entry["seed"])): entry for entry in entries}
    check(len(mapped) == len(entries), f"Duplicate run entries: {description}")
    return mapped


def _validate_source(source_rows: list[dict], expected_count: int, description: str) -> None:
    check(len(source_rows) == expected_count, f"{description} row-count mismatch")
    ids = [str(row.get("id")) for row in source_rows]
    check(None not in ids and len(ids) == len(set(ids)), f"Duplicate/missing candidate IDs: {description}")
    check(all(row.get("record_id") is not None and row.get("labels") for row in source_rows),
          f"Missing record/gold labels: {description}")


def _validate_run_manifest(run: dict, run_name: str, arm: str, seed: int,
                            expected_labels: list[str], train_hash: str,
                            validation_hash: str, training_code_hashes: dict) -> str:
    check(run.get("status") == "completed" and run.get("arm") == "C2",
          f"Run status/feature arm mismatch: {run_name}")
    check(run.get("seed") == seed and run.get("loss_arm") == ("L0" if arm == "BCE" else "L1"),
          f"Run identity mismatch: {run_name}")
    check(run.get("test_file_read") is False and run.get("official_test_read") is False,
          f"Test lock mismatch: {run_name}")
    check(run.get("labels") == expected_labels, f"Label order mismatch: {run_name}")
    check(run.get("train_sha256") == train_hash and run.get("validation_sha256") == validation_hash,
          f"Training/validation input hash mismatch: {run_name}")
    check(run.get("pretrained_model") == "google-bert/bert-base-cased"
          and run.get("pretrained_model_revision") == "cd5ef92a9fb2f889e972770a36d4ed042daf221e",
          f"Model/revision mismatch: {run_name}")
    config = run.get("config", {})
    check(config.get("epochs") == 3 and config.get("batch_size") == 2
          and config.get("accumulation") == 8 and config.get("eval_batch_size") == 8
          and config.get("encoder_lr") == 2e-5 and config.get("weight_decay") == 0.01
          and config.get("gradient_clip") == 1.0 and config.get("mixed_precision") is True
          and config.get("threshold") == 0.5
          and config.get("selection_metric") == "validation_micro_f1_at_fixed_threshold_0.5",
          f"Training configuration mismatch: {run_name}")
    expected_weight = None if arm == "BCE" else 0.1
    check(config.get("unknown_label_weight") == expected_weight, f"Loss weight mismatch: {run_name}")
    checkpoint_hash = run.get("best_checkpoint", {}).get("sha256")
    check(isinstance(checkpoint_hash, str) and len(checkpoint_hash) == 64,
          f"Missing checkpoint digest: {run_name}")
    for key, filename in (("training_code_sha256", "train_encoder.py"),
                          ("neural_code_sha256", "neural.py"),
                          ("metrics_code_sha256", "metrics.py")):
        check(run.get(key) == training_code_hashes.get(filename),
              f"Training code pin mismatch: {run_name}:{key}")
    return checkpoint_hash


def run_audit() -> dict:
    for path, expected_hash in OLD_ARCHIVE_HASHES.items():
        check(file_digest(path) == expected_hash, f"Historical archive SHA-256 mismatch: {path}")
    check(file_digest(EXTRA_SEED_ARCHIVE) == EXTRA_ARCHIVE_SHA256,
          "Seven-seed archive SHA-256 mismatch")

    prior_test = read_json(PRIOR_TEST_REPORT)
    prior_validation = read_json(PRIOR_VALIDATION_REPORT)
    check(prior_test["archives"]["test_predictions_sha256"] == OLD_ARCHIVE_HASHES[OLD_TEST_PREDICTIONS]
          and prior_test["archives"]["run_manifests_sha256"] == OLD_ARCHIVE_HASHES[OLD_RUN_MANIFESTS],
          "Historical test archives do not match the prior audit")
    check(prior_validation["archive_sha256"] == OLD_ARCHIVE_HASHES[OLD_VALIDATION_ARCHIVE]
          and prior_validation["aggregate_manifest_sha256"] == file_digest(OLD_VALIDATION_MANIFEST),
          "Historical validation archive/manifest do not match the prior audit")

    split = read_json(SPLIT_MANIFEST)
    train_bytes, validation_bytes, test_bytes = TRAIN_FILE.read_bytes(), VALIDATION_FILE.read_bytes(), TEST_FILE.read_bytes()
    train_hash, validation_hash, test_hash = map(digest, (train_bytes, validation_bytes, test_bytes))
    check(test_hash == TEST_SHA256, f"Test SHA-256 mismatch: {test_hash}")
    for name, observed in (("train", train_hash), ("validation", validation_hash), ("test", test_hash)):
        check(split["files"][name]["sha256"] == observed, f"Split-manifest hash mismatch: {name}")
    train_rows = read_jsonl(train_bytes, str(TRAIN_FILE))
    validation_rows = read_jsonl(validation_bytes, str(VALIDATION_FILE))
    test_rows = read_jsonl(test_bytes, str(TEST_FILE))
    _validate_source(train_rows, 5647, "Train")
    _validate_source(validation_rows, 720, "Validation")
    _validate_source(test_rows, 701, "Test")
    check(len({str(row["record_id"]) for row in test_rows}) == 638,
          "Test source-record cluster count mismatch")
    labels = read_json(LABEL_SCHEMA)["labels"]
    check(len(labels) == 29 and labels == prior_test["labels"], "Fixed label inventory mismatch")
    check(set(labels) == set(split["label_counts"]["train"]), "Split label inventory mismatch")
    source_hashes = {"train": ordered_id_hash(train_rows),
                     "validation": ordered_id_hash(validation_rows),
                     "test": ordered_id_hash(test_rows)}
    for split_name, rows in (("train", train_rows), ("validation", validation_rows), ("test", test_rows)):
        support = {label: 0 for label in labels}
        for row in rows:
            for label in row["labels"]:
                check(label in support, f"Unknown source label in {split_name}: {label}")
                support[label] += 1
        check(support == split["label_counts"][split_name], f"Label support mismatch: {split_name}")

    old_test_names = {"test_inference_manifest.json"}
    old_test_names.update(f"{name}/test_predictions.jsonl" for name in OLD_RUN_SPECS)
    old_test_blobs = archive_blobs(OLD_TEST_PREDICTIONS, old_test_names)
    old_run_blobs = archive_blobs(OLD_RUN_MANIFESTS,
                                  {f"{name}/run_manifest.json" for name in OLD_RUN_SPECS})
    old_validation_names = {
        f"runs/m1m4_entity_disjoint_v1/{arm}_seed{seed}/{filename}"
        for arm in ARMS for seed in OLD_SEEDS
        for filename in ("run_manifest.json", "validation_predictions.jsonl", "validation_metrics.json")
    }
    old_validation_blobs = archive_blobs(OLD_VALIDATION_ARCHIVE, old_validation_names)

    new_seeds = set(NEW_SEEDS)
    new_run_names = {f"{arm}_seed{seed}" for arm in ARMS for seed in new_seeds}
    validation_manifest_name = "m1m4_validation_manifest_seeds_101_202_303_404_505_606_707.json"
    inference_manifest_name = "test_inference_manifest_seeds_101_202_303_404_505_606_707.json"
    new_names = {validation_manifest_name, inference_manifest_name}
    for name in new_run_names:
        new_names.update({f"{name}/run_manifest.json", f"{name}/validation_predictions.jsonl",
                          f"{name}/test_predictions.jsonl"})
    new_blobs = archive_blobs(EXTRA_SEED_ARCHIVE, new_names)

    old_validation_manifest = read_json(OLD_VALIDATION_MANIFEST)
    new_validation_manifest = json.loads(new_blobs[validation_manifest_name])
    old_test_inference = json.loads(old_test_blobs["test_inference_manifest.json"])
    new_test_inference = json.loads(new_blobs[inference_manifest_name])
    check(old_validation_manifest.get("status") == "completed"
          and old_validation_manifest.get("test_file_read") is False,
          "Historical validation manifest status/test lock mismatch")
    check(new_validation_manifest.get("status") == "completed"
          and new_validation_manifest.get("test_file_read") is False
          and new_validation_manifest.get("official_test_read") is False,
          "New validation manifest status/test lock mismatch")
    check(old_test_inference.get("test_previously_inspected") is True
          and old_test_inference.get("test_used_for_checkpoint_or_threshold_selection") is False
          and old_test_inference.get("test_metrics_computed") is False,
          "Historical test inference lock mismatch")
    check(new_test_inference.get("status") == "completed"
          and new_test_inference.get("test_previously_inspected") is True
          and new_test_inference.get("test_used_for_checkpoint_or_threshold_selection") is False
          and new_test_inference.get("test_metrics_computed") is False,
          "New test inference lock mismatch")
    check(old_test_inference.get("test_input_sha256") == test_hash
          and new_test_inference.get("test_input_sha256") == test_hash,
          "Test inference input hash mismatch")
    check(new_test_inference.get("test_candidate_count") == 701
          and new_test_inference.get("test_candidate_order_sha256") == source_hashes["test"],
          "New test inference candidate manifest mismatch")
    check(new_test_inference.get("seeds") == list(NEW_SEEDS)
          and new_test_inference.get("labels") == labels,
          "New inference seed/label manifest mismatch")
    check(new_validation_manifest.get("seeds") == list(NEW_SEEDS)
          and new_validation_manifest.get("labels_in_score_vector_order") == labels,
          "New validation seed/label manifest mismatch")

    local_hashes = {
        "feature_manifest_sha256": canonical_json_digest(FEATURE_MANIFEST),
        "label_schema_sha256": canonical_json_digest(LABEL_SCHEMA),
        "tokenizer_sha256": file_digest(TOKENIZER),
        "training_code_sha256": file_digest(TRAINER),
        "neural_code_sha256": file_digest(NEURAL),
        "metrics_code_sha256": file_digest(METRICS),
        "exporter_code_sha256": file_digest(EXPORTER),
    }
    code_manifest = new_validation_manifest["training_code_hashes"]
    check(code_manifest == {"train_encoder.py": local_hashes["training_code_sha256"],
                            "neural.py": local_hashes["neural_code_sha256"],
                            "metrics.py": local_hashes["metrics_code_sha256"]},
          "Local training source hashes differ from the new run manifest")
    for key in ("feature_manifest_sha256", "label_schema_sha256", "tokenizer_sha256"):
        check(new_test_inference.get(key) == local_hashes[key], f"New inference provenance mismatch: {key}")
        check(new_validation_manifest.get({"feature_manifest_sha256": "context_manifest_sha256"}.get(key, key))
              == local_hashes[key], f"New validation provenance mismatch: {key}")
    check(new_test_inference.get("exporter_code_sha256") == local_hashes["exporter_code_sha256"],
          "Current exporter hash differs from new inference manifest")
    check(split["feature_manifest_sha256"] == local_hashes["feature_manifest_sha256"],
          "Feature/split provenance mismatch")
    check(new_validation_manifest["train_input"]["sha256"] == train_hash
          and new_validation_manifest["validation_input"]["sha256"] == validation_hash,
          "New validation manifest data hashes mismatch")
    check(new_validation_manifest["validation_input"]["candidate_order_sha256"] == source_hashes["validation"],
          "New validation candidate-order hash mismatch")
    check(old_validation_manifest["train_input"]["sha256"] == train_hash
          and old_validation_manifest["validation_input"]["sha256"] == validation_hash
          and old_validation_manifest["validation_input"]["candidate_order_sha256"] == source_hashes["validation"],
          "Historical validation manifest data/order mismatch")

    old_validation_entries = _manifest_entries(old_validation_manifest, "runs", "historical validation")
    new_validation_entries = _manifest_entries(new_validation_manifest, "runs", "new validation")
    check(set(old_validation_entries) == {(arm, seed) for arm in ARMS for seed in OLD_SEEDS},
          "Historical validation run inventory mismatch")
    check(set(new_validation_entries) == {(arm, seed) for arm in ARMS for seed in NEW_SEEDS},
          "New validation run inventory mismatch")
    check(set(old_test_inference.get("runs", {})) == set(OLD_RUN_SPECS),
          "Historical test inference run inventory mismatch")
    check(set(new_test_inference.get("runs", {})) == new_run_names,
          "New test inference run inventory mismatch")

    validation_metrics, validation_runs = {}, {}
    test_seed_results, test_run_arrays, run_hashes = {}, {}, {}
    validation_run_arrays, test_score_arrays = {}, {}
    runtime_signatures = set()
    all_checkpoint_links_match = True
    prior_val_runs = {(entry["loss"], int(entry["seed"])): entry for entry in prior_validation["runs"]}

    for arm in ARMS:
        for seed in SEEDS:
            run_name = f"{arm}_seed{seed}"
            is_old = seed in OLD_SEEDS
            if is_old:
                run_bytes = old_run_blobs[f"{run_name}/run_manifest.json"]
                test_prediction_bytes = old_test_blobs[f"{run_name}/test_predictions.jsonl"]
                validation_folder = f"runs/m1m4_entity_disjoint_v1/{run_name}"
                validation_prediction_bytes = old_validation_blobs[f"{validation_folder}/validation_predictions.jsonl"]
                validation_metrics_bytes = old_validation_blobs[f"{validation_folder}/validation_metrics.json"]
                stored_validation_metrics = json.loads(validation_metrics_bytes)
                val_entry = old_validation_entries[(arm, seed)]
                inference_entry = old_test_inference["runs"][run_name]
                expected_run_hash = OLD_RUN_SPECS[run_name][3]
                expected_checkpoint_hash = OLD_RUN_SPECS[run_name][4]
                check(val_entry["run_manifest_sha256"] == expected_run_hash,
                      f"Historical validation/run hash mismatch: {run_name}")
                check(val_entry["validation_predictions_sha256"]
                      == prior_validation_entry_hash(prior_validation, arm, seed),
                      f"Historical validation/prediction hash mismatch: {run_name}")
            else:
                run_bytes = new_blobs[f"{run_name}/run_manifest.json"]
                test_prediction_bytes = new_blobs[f"{run_name}/test_predictions.jsonl"]
                validation_prediction_bytes = new_blobs[f"{run_name}/validation_predictions.jsonl"]
                stored_validation_metrics = None
                val_entry = new_validation_entries[(arm, seed)]
                inference_entry = new_test_inference["runs"][run_name]
                expected_run_hash = val_entry["run_manifest_sha256"]
                expected_checkpoint_hash = None
            run_hash = digest(run_bytes)
            check(run_hash == expected_run_hash, f"Run-manifest hash mismatch: {run_name}")
            run = json.loads(run_bytes)
            checkpoint_hash = _validate_run_manifest(run, run_name, arm, seed, labels,
                                                     train_hash, validation_hash, code_manifest)
            if expected_checkpoint_hash is not None:
                check(checkpoint_hash == expected_checkpoint_hash,
                      f"Historical checkpoint pin mismatch: {run_name}")
            check(inference_entry.get("source_run_manifest_sha256") == run_hash
                  and inference_entry.get("checkpoint_sha256") == checkpoint_hash,
                  f"Test/run/checkpoint provenance mismatch: {run_name}")
            if not is_old:
                check(inference_entry.get("loss_arm") == run.get("loss_arm")
                      and inference_entry.get("seed") == seed,
                      f"New inference run identity mismatch: {run_name}")
            runtime = run.get("runtime", {})
            runtime_signatures.add(json.dumps({key: runtime.get(key) for key in
                ("accelerator_name", "python_version", "torch_version", "transformers_version",
                 "tokenizers_version", "safetensors_version", "numpy_version")}, sort_keys=True))
            if is_old:
                check(digest(validation_metrics_bytes) == run.get("validation_metrics_sha256"),
                      f"Validation-metrics hash mismatch: {run_name}")
            check(digest(validation_prediction_bytes) == val_entry["validation_predictions_sha256"],
                  f"Validation prediction hash mismatch: {run_name}")
            check(run.get("validation_predictions_sha256") == val_entry["validation_predictions_sha256"],
                  f"Run/validation prediction cross-link mismatch: {run_name}")
            check(digest(test_prediction_bytes) == inference_entry["test_predictions_sha256"],
                  f"Test prediction hash mismatch: {run_name}")
            check(inference_entry.get("test_prediction_rows") == 701,
                  f"Test prediction row manifest mismatch: {run_name}")
            check(inference_entry.get("test_candidate_order_sha256") == source_hashes["test"],
                  f"Test order manifest mismatch: {run_name}")

            validation_gold, validation_predicted, validation_scores, val_order = prediction_arrays(
                validation_rows, validation_prediction_bytes, labels, "validation", validation_hash,
                f"{run_name}/validation_predictions.jsonl")
            test_gold, test_predicted, test_scores, test_order = prediction_arrays(
                test_rows, test_prediction_bytes, labels, "test", test_hash,
                f"{run_name}/test_predictions.jsonl")
            check(val_order == source_hashes["validation"] and test_order == source_hashes["test"],
                  f"Candidate-order hash mismatch: {run_name}")
            val_metrics = calculate_metrics(validation_gold, validation_predicted, labels)
            test_metrics = calculate_metrics(test_gold, test_predicted, labels)
            test_metrics.update(calculate_macro_average_precision(test_gold, test_scores, labels))
            if stored_validation_metrics is not None:
                check(stored_validation_metrics["all_candidates"]["candidate_count"] == 720,
                      f"Stored validation metric count mismatch: {run_name}")
                check_close(stored_validation_metrics["all_candidates"]["macro_f1_fixed_inventory"],
                            val_metrics["macro_f1_fixed_inventory"], f"stored validation macro-F1 {run_name}")
                check_close(stored_validation_metrics["all_candidates"]["micro_f1"],
                            val_metrics["micro_f1"], f"stored validation micro-F1 {run_name}")
            elif "validation_macro_f1_at_0_5" in val_entry:
                check_close(val_entry["validation_macro_f1_at_0_5"],
                            val_metrics["macro_f1_fixed_inventory"], f"manifest validation macro-F1 {run_name}")
                check_close(val_entry["validation_micro_f1_at_0_5"],
                            val_metrics["micro_f1"], f"manifest validation micro-F1 {run_name}")
            validation_runs[run_name] = {"arm": arm, "seed": seed, "metrics": val_metrics,
                                         "validation_predictions_sha256": digest(validation_prediction_bytes)}
            test_seed_results[run_name] = {"arm": arm, "seed": seed, "metrics": test_metrics}
            test_run_arrays[run_name] = (test_gold, test_predicted)
            validation_run_arrays[run_name] = (validation_gold, validation_scores)
            test_score_arrays[run_name] = (test_gold, test_scores)
            validation_metrics[run_name] = val_metrics
            run_hashes[run_name] = {
                "run_manifest_sha256": run_hash,
                "checkpoint_sha256_recorded_in_run_manifest": checkpoint_hash,
                "validation_predictions_sha256": digest(validation_prediction_bytes),
                "test_predictions_sha256": digest(test_prediction_bytes),
            }
            all_checkpoint_links_match = all_checkpoint_links_match and (
                inference_entry.get("checkpoint_sha256") == checkpoint_hash)

            if is_old:
                prior_test_run = prior_test["run_hashes"][run_name]
                check(prior_test_run["run_manifest_sha256"] == run_hash
                      and prior_test_run["test_predictions_sha256"] == digest(test_prediction_bytes)
                      and prior_test_run["checkpoint_sha256_recorded_in_run_manifest"] == checkpoint_hash,
                      f"Historical test archive differs from earlier audit: {run_name}")
                previous_val = prior_val_runs[(arm, seed)]["metrics_at_0_5"]
                check_close(previous_val["macro_f1_fixed_inventory"],
                            val_metrics["macro_f1_fixed_inventory"], f"prior validation macro-F1 {run_name}")
                check_close(previous_val["micro_f1"], val_metrics["micro_f1"],
                            f"prior validation micro-F1 {run_name}")
                for metric_name in ("macro_f1_fixed_inventory", "micro_f1", "micro_precision",
                                    "micro_recall", "exact_set_match", "predicted_labels_per_candidate"):
                    prior_seed = prior_test["aggregate_metrics"][metric_name]["seed_values"]
                    check_close(prior_seed["BCE" if arm == "BCE" else "NegDW-BCE"][str(seed)],
                                test_metrics[metric_name], f"prior test {metric_name} {run_name}")

    validation_aggregate = {}
    for metric_name in ("macro_f1_fixed_inventory", "micro_f1"):
        bce = [validation_metrics[f"BCE_seed{seed}"][metric_name] for seed in SEEDS]
        negdw = [validation_metrics[f"NegDW-BCE_seed{seed}"][metric_name] for seed in SEEDS]
        paired = [right - left for left, right in zip(bce, negdw)]
        validation_aggregate[metric_name] = {
            "BCE": summary(bce), "NegDW-BCE": summary(negdw),
            "paired_difference_NegDW_minus_BCE": summary(paired),
            "seed_values": {"BCE": dict(zip(SEEDS, bce)),
                            "NegDW-BCE": dict(zip(SEEDS, negdw)),
                            "paired_difference": dict(zip(SEEDS, paired))},
        }

    aggregate = {}
    metric_names = ("macro_f1_fixed_inventory", "macro_average_precision_fixed_inventory",
                    "micro_f1", "micro_precision", "micro_recall", "exact_set_match",
                    "predicted_labels_per_candidate")
    for metric_name in metric_names:
        bce = [test_seed_results[f"BCE_seed{seed}"]["metrics"][metric_name] for seed in SEEDS]
        negdw = [test_seed_results[f"NegDW-BCE_seed{seed}"]["metrics"][metric_name] for seed in SEEDS]
        paired = [right - left for left, right in zip(bce, negdw)]
        aggregate[metric_name] = {
            "BCE": summary(bce), "NegDW-BCE": summary(negdw),
            "paired_difference_NegDW_minus_BCE": summary(paired),
            "seed_values": {"BCE": dict(zip(SEEDS, bce)),
                            "NegDW-BCE": dict(zip(SEEDS, negdw)),
                            "paired_difference": dict(zip(SEEDS, paired))},
        }

    test_support = split["label_counts"]["test"]
    support_filtered_labels = [label for label in labels if test_support[label] >= 5]
    check(len(support_filtered_labels) == 21,
          f"Expected 21 test relations with support >=5, found {len(support_filtered_labels)}")
    support_filtered_values = {}
    for arm in ARMS:
        support_filtered_values[arm] = [statistics.mean(
            test_seed_results[f"{arm}_seed{seed}"]["metrics"]["per_class"][label]["f1"]
            for label in support_filtered_labels) for seed in SEEDS]
    support_filtered_paired = [right - left for left, right in zip(
        support_filtered_values["BCE"], support_filtered_values["NegDW-BCE"])]
    aggregate["macro_f1_test_support_ge_5"] = {
        "BCE": summary(support_filtered_values["BCE"]),
        "NegDW-BCE": summary(support_filtered_values["NegDW-BCE"]),
        "paired_difference_NegDW_minus_BCE": summary(support_filtered_paired),
        "seed_values": {
            "BCE": dict(zip(SEEDS, support_filtered_values["BCE"])),
            "NegDW-BCE": dict(zip(SEEDS, support_filtered_values["NegDW-BCE"])),
            "paired_difference": dict(zip(SEEDS, support_filtered_paired)),
        },
    }

    train_support = split["label_counts"]["train"]
    strata = {
        "Low": [label for label in labels if train_support[label] <= 66],
        "Middle": [label for label in labels if 67 <= train_support[label] <= 167],
        "High": [label for label in labels if train_support[label] > 167],
    }
    per_relation = {}
    for label in labels:
        bce_values = [test_seed_results[f"BCE_seed{seed}"]["metrics"]["per_class"][label]["f1"]
                      for seed in SEEDS]
        negdw_values = [test_seed_results[f"NegDW-BCE_seed{seed}"]["metrics"]["per_class"][label]["f1"]
                        for seed in SEEDS]
        differences = [right - left for left, right in zip(bce_values, negdw_values)]
        per_relation[label] = {
            "train_support": train_support[label],
            "validation_support": split["label_counts"]["validation"][label],
            "test_support": split["label_counts"]["test"][label],
            "stratum": next(name for name, group in strata.items() if label in group),
            "BCE_f1": summary(bce_values),
            "NegDW-BCE_f1": summary(negdw_values),
            "paired_difference": summary(differences),
        }
    per_relation = dict(sorted(per_relation.items(), key=lambda item: (item[1]["train_support"], item[0])))

    frequency_strata = {}
    for stratum, group in strata.items():
        bce_by_seed = [statistics.mean(
            test_seed_results[f"BCE_seed{seed}"]["metrics"]["per_class"][label]["f1"] for label in group)
            for seed in SEEDS]
        negdw_by_seed = [statistics.mean(
            test_seed_results[f"NegDW-BCE_seed{seed}"]["metrics"]["per_class"][label]["f1"] for label in group)
            for seed in SEEDS]
        differences = [right - left for left, right in zip(bce_by_seed, negdw_by_seed)]
        support_values = [train_support[label] for label in group]
        frequency_strata[stratum] = {
            "relation_count": len(group),
            "training_support_range": [min(support_values), max(support_values)],
            "BCE_macro_f1": summary(bce_by_seed),
            "NegDW-BCE_macro_f1": summary(negdw_by_seed),
            "paired_difference": summary(differences),
            "seed_values": {"BCE": dict(zip(SEEDS, bce_by_seed)),
                            "NegDW-BCE": dict(zip(SEEDS, negdw_by_seed)),
                            "paired_difference": dict(zip(SEEDS, differences))},
        }

    paired_macro_differences = aggregate["macro_f1_fixed_inventory"]["seed_values"]["paired_difference"]
    cluster_bootstrap, joint_bootstrap = bootstrap_intervals(
        test_run_arrays, [str(row["record_id"]) for row in test_rows], labels)
    sign_flip = exact_seed_sign_flip([paired_macro_differences[seed] for seed in SEEDS])
    tuned_thresholds = {
        arm: choose_mean_macro_f1_threshold(validation_run_arrays, arm, SEEDS, labels)
        for arm in ARMS
    }
    validation_macro_f1_tuned = summarize_threshold_policy(
        validation_run_arrays, labels, SEEDS,
        {arm: item[0] for arm, item in tuned_thresholds.items()})
    test_macro_f1_tuned = summarize_threshold_policy(
        test_score_arrays, labels, SEEDS,
        {arm: item[0] for arm, item in tuned_thresholds.items()})
    negdw_validation_recall_at_05 = mean_micro_recall_at_threshold(
        validation_run_arrays, "NegDW-BCE", SEEDS, 0.5)
    recall_matched_bce_threshold, recall_matched_bce_validation_recall = (
        choose_recall_matched_threshold(validation_run_arrays, SEEDS,
                                        negdw_validation_recall_at_05))
    validation_recall_matched = summarize_threshold_policy(
        validation_run_arrays, labels, SEEDS,
        {"BCE": recall_matched_bce_threshold, "NegDW-BCE": 0.5})
    test_recall_matched = summarize_threshold_policy(
        test_score_arrays, labels, SEEDS,
        {"BCE": recall_matched_bce_threshold, "NegDW-BCE": 0.5})
    sensitivity_uncertainty = two_way_posthoc_sensitivity_bootstrap(
        test_score_arrays, [str(row["record_id"]) for row in test_rows], labels,
        {
            "validation_macro_f1_selected_thresholds": {
                arm: item[0] for arm, item in tuned_thresholds.items()},
            "validation_recall_matched_thresholds": {
                "BCE": recall_matched_bce_threshold, "NegDW-BCE": 0.5},
        })
    check_close(
        sensitivity_uncertainty["metrics"]["validation_macro_f1_selected_thresholds"]["point_estimate"],
        test_macro_f1_tuned["aggregate_metrics"]["macro_f1_fixed_inventory"][
            "paired_difference_NegDW_minus_BCE"]["mean"],
        "macro-F1-tuned point estimate")
    check_close(
        sensitivity_uncertainty["metrics"]["validation_recall_matched_thresholds"]["point_estimate"],
        test_recall_matched["aggregate_metrics"]["macro_f1_fixed_inventory"][
            "paired_difference_NegDW_minus_BCE"]["mean"],
        "recall-matched point estimate")
    check_close(
        sensitivity_uncertainty["metrics"]["macro_average_precision_fixed_inventory"]["point_estimate"],
        aggregate["macro_average_precision_fixed_inventory"][
            "paired_difference_NegDW_minus_BCE"]["mean"],
        "macro-AP point estimate")
    archive_paths = (OLD_TEST_PREDICTIONS, OLD_RUN_MANIFESTS, OLD_VALIDATION_ARCHIVE, EXTRA_SEED_ARCHIVE)
    return {
        "status": "completed",
        "interpretation": "post-hoc exploratory analysis; test previously inspected and seven seeds added after initial three-seed results were observed",
        "threshold": 0.5,
        "seeds": list(SEEDS),
        "new_seeds": list(NEW_SEEDS),
        "test_input": {"path": str(TEST_FILE), "sha256": test_hash,
                       "candidate_count": len(test_rows),
                       "source_record_cluster_count": len({str(row["record_id"]) for row in test_rows}),
                       "candidate_order_sha256": source_hashes["test"]},
        "validation_input": {"path": str(VALIDATION_FILE), "sha256": validation_hash,
                              "candidate_count": len(validation_rows),
                              "candidate_order_sha256": source_hashes["validation"]},
        "archives": {str(path): {"sha256": file_digest(path)} for path in archive_paths},
        "prior_archives_rechecked_against_prior_report": True,
        "new_archive": {"path": str(EXTRA_SEED_ARCHIVE), "sha256": EXTRA_ARCHIVE_SHA256,
                        "member_count": len(new_blobs)},
        "validation_checks": {
            "old_archives_match_previous_audit_hashes": True,
            "new_archive_exact_member_set_and_safe_paths": True,
            "old_and_new_candidate_gold_record_order_checks": True,
            "training_validation_test_input_hashes_match_split_manifest": True,
            "all_run_config_model_code_and_checkpoint_crosslinks_match": all_checkpoint_links_match,
            "new_training_and_exporter_code_hashes_match_local_sources": True,
            "test_lock_preserved_in_training_and_inference_manifests": True,
            "checkpoint_blobs_absent_checkpoint_digests_crosslinked_only": True,
        },
        "provenance": {
            **local_hashes,
            "historical_exporter_code_sha256": prior_test["provenance_hashes"]["exporter_code_sha256"],
            "model_id": "google-bert/bert-base-cased",
            "model_revision": "cd5ef92a9fb2f889e972770a36d4ed042daf221e",
            "runtime_signatures": sorted(runtime_signatures),
        },
        "test_support_sensitivity": {
            "support_threshold": 5,
            "included_relation_count": len(support_filtered_labels),
            "included_relations": support_filtered_labels,
            "excluded_relation_count": len(labels) - len(support_filtered_labels),
            "excluded_relations_and_support": {
                label: test_support[label] for label in labels if test_support[label] < 5
            },
            "selection_basis": "Post-hoc filter using recorded positive-label support in the test split.",
        },
        "labels": labels,
        "run_hashes": run_hashes,
        "validation_aggregate_metrics": validation_aggregate,
        "validation_seed_results": validation_runs,
        "threshold_sensitivity": {
            "selection_protocol": (
                "One global threshold per loss selected on validation to maximize mean fixed-inventory "
                "macro-F1 across the ten paired seeds; ties select the higher threshold. Thresholds "
                "are then frozen and applied unchanged to each seed's test predictions."
            ),
            "validation_macro_f1_selected_thresholds": {
                arm: {"threshold": threshold, "selection_objective_mean_validation_macro_f1": objective}
                for arm, (threshold, objective) in tuned_thresholds.items()
            },
            "validation_macro_f1_selected_threshold_policy": {
                "validation": validation_macro_f1_tuned,
                "test": test_macro_f1_tuned,
            },
            "recall_matched_protocol": (
                "Select one BCE threshold on validation that minimizes the absolute difference between "
                "mean seed-wise validation micro-recall and NegDW-BCE micro-recall at 0.5; ties select "
                "the higher threshold. Apply BCE at this threshold and NegDW-BCE at 0.5 on test."
            ),
            "recall_matched_thresholds": {
                "BCE": recall_matched_bce_threshold,
                "NegDW-BCE": 0.5,
                "NegDW_validation_mean_micro_recall_at_0_5": negdw_validation_recall_at_05,
                "BCE_validation_mean_micro_recall_at_selected_threshold": recall_matched_bce_validation_recall,
            },
            "recall_matched_threshold_policy": {
                "validation": validation_recall_matched,
                "test": test_recall_matched,
            },
            "limitations": [
                "Threshold selection reuses validation data that also informed checkpoint selection; validation metrics after threshold selection are in-sample.",
                "The entity-disjoint test was previously inspected and the seven added seeds followed observation of the initial three-seed test results; transferred test metrics are post-hoc and exploratory.",
                "No independent expert-adjudicated FinRED evaluation set is present among the audited project artifacts.",
            ],
        },
        "aggregate_metrics": aggregate,
        "seed_results": test_seed_results,
        "frequency_strata": frequency_strata,
        "per_relation": per_relation,
        "cluster_bootstrap": cluster_bootstrap,
        "seed_cluster_bootstrap": joint_bootstrap,
        "paired_seed_sign_flip_test": sign_flip,
        "posthoc_sensitivity_uncertainty": sensitivity_uncertainty,
        "limitations": [
            "The test split was previously inspected; results are exploratory, not independent confirmation.",
            "Seven additional seeds were run after the initial three-seed test result was observed.",
            "Metrics measure agreement with recorded annotations, not verified relation truth.",
            "Checkpoint blobs were not included; checkpoint hashes are cross-linked across manifests but cannot be independently recomputed.",
            "Threshold sensitivity is assessed with validation-selected operating points, but validation also informed checkpoint selection.",
            "No independent expert-adjudicated FinRED evaluation set is present among the audited project artifacts.",
            "Ten seeds do not resolve controlled label-removal evidence or independent evaluation.",
        ],
    }


def prior_validation_entry_hash(report: dict, arm: str, seed: int) -> str:
    for entry in report["runs"]:
        if entry["loss"] == arm and int(entry["seed"]) == seed:
            return entry["validation_predictions_sha256"]
    raise ValueError(f"No prior validation record for {arm} seed {seed}")


def write_report(result: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result["analysis_code_sha256"] = file_digest(Path(__file__))
    json_path = OUTPUT_DIR / "m1_m4_test_analysis_10seeds.json"
    markdown_path = OUTPUT_DIR / "m1_m4_test_analysis_10seeds.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    fmt = lambda value: f"{value['mean']:.4f} ({value['sample_sd']:.4f})"
    lines = [
        "# Ten-Seed M1/M4 Test and Validation Audit", "",
        f"- Added archive SHA-256: `{result['new_archive']['sha256']}`; {result['new_archive']['member_count']} members.",
        f"- Validation input: {result['validation_input']['candidate_count']} rows; SHA-256 `{result['validation_input']['sha256']}`.",
        f"- Test input: {result['test_input']['candidate_count']} candidates / {result['test_input']['source_record_cluster_count']} source-record clusters; SHA-256 `{result['test_input']['sha256']}`.",
        f"- Ten paired seeds: {', '.join(map(str, result['seeds']))}; stored artifacts use 0.5, while sensitivity analyses re-threshold raw scores using validation-selected policies.",
        "- Historical three-seed archives were rechecked; all new manifests and train/validation/test predictions passed hash and alignment checks.",
        "- Training manifests record no test access; inference manifests record no test-based checkpoint/threshold selection or test-metric computation.",
        "- Checkpoint hashes cross-link across manifests; checkpoint blobs were not supplied for independent verification.",
        "- The test was previously inspected; seven seeds were added after the initial three-seed results, so the expansion remains exploratory.",
        "", "## Validation Results", "",
        "Mean (sample SD) across ten seeds; paired differences are calculated seed-wise at 0.5.", "",
        "| Condition | Seeds | Fixed-inventory macro-F1 | Micro-F1 |",
        "|---|---:|---:|---:|",
    ]
    validation = result["validation_aggregate_metrics"]
    for label, arm in (("BCE", "BCE"), ("NegDW-BCE", "NegDW-BCE")):
        lines.append(f"| {label} | 10 | {fmt(validation['macro_f1_fixed_inventory'][arm])} | {fmt(validation['micro_f1'][arm])} |")
    lines.append(f"| NegDW-BCE - BCE | 10 | {fmt(validation['macro_f1_fixed_inventory']['paired_difference_NegDW_minus_BCE'])} | {fmt(validation['micro_f1']['paired_difference_NegDW_minus_BCE'])} |")
    lines.extend(["", "## Test Metrics", "",
                  "Mean (sample SD) across ten seeds; paired differences are calculated seed-wise. Macro-AP is the unweighted mean of per-relation average precision over the fixed 29-label inventory, computed from raw scores with tied scores grouped.", "",
                  "| Metric | BCE | NegDW-BCE | Paired difference |", "|---|---:|---:|---:|"])
    display_names = {"macro_f1_fixed_inventory": "Fixed-inventory macro-F1",
                     "macro_f1_test_support_ge_5": "Macro-F1; test support >=5 (21 relations)",
                     "macro_average_precision_fixed_inventory": "Fixed-inventory macro-AP",
                     "micro_f1": "Micro-F1",
                     "micro_precision": "Micro-precision", "micro_recall": "Micro-recall",
                     "exact_set_match": "Exact-set match", "predicted_labels_per_candidate": "Predicted labels/candidate"}
    for key, name in display_names.items():
        item = result["aggregate_metrics"][key]
        lines.append(f"| {name} | {fmt(item['BCE'])} | {fmt(item['NegDW-BCE'])} | {fmt(item['paired_difference_NegDW_minus_BCE'])} |")
    support_sensitivity = result["test_support_sensitivity"]
    excluded = ", ".join(
        f"{label} ({count})" for label, count in support_sensitivity["excluded_relations_and_support"].items())
    lines.extend([
        "",
        f"The support-filtered macro-F1 averages per-relation F1 over {support_sensitivity['included_relation_count']} relations with at least {support_sensitivity['support_threshold']} recorded test positives. This post-hoc sensitivity excludes {support_sensitivity['excluded_relation_count']} relations with support below five: {excluded}. It is descriptive and remains agreement with recorded annotations, not verified relation truth.",
    ])
    threshold = result["threshold_sensitivity"]
    tuned = threshold["validation_macro_f1_selected_threshold_policy"]
    tuned_thresholds = threshold["validation_macro_f1_selected_thresholds"]
    lines.extend(["", "## Validation-Selected Threshold Sensitivity", "",
        "One global threshold per loss maximizes mean validation macro-F1 across ten seeds; each is then frozen for test. Validation metrics are selection-in-sample, and test metrics are post hoc because this test was previously inspected.",
        "", "| Loss | Threshold | Validation macro-F1 | Test macro-F1 | Test micro-P | Test micro-R |",
        "|---|---:|---:|---:|---:|---:|"])
    for arm in ("BCE", "NegDW-BCE"):
        validation_metrics = tuned["validation"]["aggregate_metrics"]
        test_metrics = tuned["test"]["aggregate_metrics"]
        lines.append(
            f"| {arm} | {tuned_thresholds[arm]['threshold']:.6f} | "
            f"{fmt(validation_metrics['macro_f1_fixed_inventory'][arm])} | "
            f"{fmt(test_metrics['macro_f1_fixed_inventory'][arm])} | "
            f"{fmt(test_metrics['micro_precision'][arm])} | {fmt(test_metrics['micro_recall'][arm])} |"
        )
    matched = threshold["recall_matched_threshold_policy"]
    matched_thresholds = threshold["recall_matched_thresholds"]
    lines.extend(["", "### BCE Matched to NegDW Validation Recall", "",
        f"NegDW-BCE validation micro-recall at 0.5 is {matched_thresholds['NegDW_validation_mean_micro_recall_at_0_5']:.4f}; BCE threshold {matched_thresholds['BCE']:.6f} gives {matched_thresholds['BCE_validation_mean_micro_recall_at_selected_threshold']:.4f}. The validation-selected BCE threshold is frozen; NegDW-BCE remains at 0.5.",
        "", "| Test operating point | Threshold | Macro-F1 | Micro-P | Micro-R |",
        "|---|---:|---:|---:|---:|"])
    matched_test = matched["test"]["aggregate_metrics"]
    for arm in ("BCE", "NegDW-BCE"):
        lines.append(
            f"| {arm} | {matched_thresholds[arm]:.6f} | "
            f"{fmt(matched_test['macro_f1_fixed_inventory'][arm])} | "
            f"{fmt(matched_test['micro_precision'][arm])} | "
            f"{fmt(matched_test['micro_recall'][arm])} |"
        )
    cluster = result["cluster_bootstrap"]
    joint = result["seed_cluster_bootstrap"]
    sign_flip = result["paired_seed_sign_flip_test"]
    lines.extend(["", "## Uncertainty", "",
        f"- Cluster-only bootstrap 95% percentile interval: [{cluster['lower_95']:.4f}, {cluster['upper_95']:.4f}] ({cluster['replicates']:,} replicates; {cluster['cluster_count']} clusters; conditional on observed seeds).",
        f"- Seed-plus-cluster bootstrap 95% percentile interval: [{joint['lower_95']:.4f}, {joint['upper_95']:.4f}] ({joint['replicates']:,} replicates; {joint['seed_count']} paired seeds; RNG seed {joint['rng_seed']}).",
        f"- Exact two-sided seed-level sign-flip test: p={sign_flip['p_value_two_sided']:.6f} ({sign_flip['assignments']:,} assignments; exploratory and post hoc).",
        "", "### Sensitivity and Ranking Contrasts", "",
        "The following intervals jointly resample paired seeds and source-record clusters. Thresholds remain frozen at their validation-selected values; macro-AP is recomputed from raw scores, with repeated clusters represented as duplicated rows and ties grouped.",
        "", "| Test contrast (NegDW-BCE - BCE) | Mean difference (sample SD) | Two-way bootstrap 95% CI | Exact sign-flip p |",
        "|---|---:|---:|---:|"])
    sensitivity = result["posthoc_sensitivity_uncertainty"]
    sensitivity_labels = {
        "validation_macro_f1_selected_thresholds": "Macro-F1; separately validation-tuned thresholds",
        "validation_recall_matched_thresholds": "Macro-F1; validation-recall-matched thresholds",
        "macro_average_precision_fixed_inventory": "Threshold-independent macro-AP",
    }
    for metric_name, label in sensitivity_labels.items():
        item = sensitivity["metrics"][metric_name]
        interval = item["two_way_bootstrap_95_percentile_ci"]
        p_value = item["exact_seed_sign_flip_test"]["p_value_two_sided"]
        lines.append(
            f"| {label} | {item['point_estimate']:+.4f} ({item['sample_sd']:.4f}) | "
            f"[{interval['lower']:+.4f}, {interval['upper']:+.4f}] | {p_value:.6f} |"
        )
    lines.extend([
        "",
        f"These are exploratory post-hoc summaries (10,000 replicates; {sensitivity['seed_count']} paired seeds; {sensitivity['cluster_count']} source-record clusters; RNG seed {sensitivity['rng_seed']}). The test was previously inspected and seven seeds followed observation of the initial three-seed results; intervals and p-values are not independent confirmation.",
        "", "## Per-Seed Macro-F1", "", "| Seed | BCE | NegDW-BCE | Difference |", "|---:|---:|---:|---:|"])
    macro = result["aggregate_metrics"]["macro_f1_fixed_inventory"]["seed_values"]
    for seed in SEEDS:
        bce, negdw = macro["BCE"][seed], macro["NegDW-BCE"][seed]
        lines.append(f"| {seed} | {bce:.4f} | {negdw:.4f} | {negdw-bce:+.4f} |")
    lines.extend(["", "## Frequency Strata", "",
                  "| Group | Relations | Training support | BCE | NegDW-BCE | Difference |",
                  "|---|---:|---:|---:|---:|---:|"])
    for name in ("Low", "Middle", "High"):
        item = result["frequency_strata"][name]
        lines.append(f"| {name} | {item['relation_count']} | {item['training_support_range'][0]}--{item['training_support_range'][1]} | {fmt(item['BCE_macro_f1'])} | {fmt(item['NegDW-BCE_macro_f1'])} | {fmt(item['paired_difference'])} |")
    lines.extend(["", "## Limitations", "", *[f"- {item}" for item in result["limitations"]],
                  "", "Per-relation scores and complete provenance details are in the adjacent JSON artifact."])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Audit passed. Reports: {markdown_path} and {json_path}")
    for key in ("macro_f1_fixed_inventory", "macro_average_precision_fixed_inventory", "micro_f1",
                "micro_precision", "micro_recall", "exact_set_match"):
        item = result["aggregate_metrics"][key]
        paired = item["paired_difference_NegDW_minus_BCE"]
        print(f"{key}: BCE={fmt(item['BCE'])}; NegDW={fmt(item['NegDW-BCE'])}; paired={fmt(paired)}")
    for key in ("macro_f1_fixed_inventory", "micro_f1"):
        item = result["validation_aggregate_metrics"][key]
        print(f"validation {key}: BCE={fmt(item['BCE'])}; NegDW={fmt(item['NegDW-BCE'])}; paired={fmt(item['paired_difference_NegDW_minus_BCE'])}")
    print(f"cluster-only 95% CI: [{cluster['lower_95']:.6f}, {cluster['upper_95']:.6f}]")
    print(f"seed+cluster 95% CI: [{joint['lower_95']:.6f}, {joint['upper_95']:.6f}]")
    print(f"exact seed-level sign-flip p={sign_flip['p_value_two_sided']:.6f}")


if __name__ == "__main__":
    write_report(run_audit())