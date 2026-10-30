#!/usr/bin/env python3
"""Train and evaluate matched FinRED entity-disjoint loss-family runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd().resolve()
FEATURE_DIR = Path("data/features/finred_entity_disjoint_v1")
SPLIT_MANIFEST = Path("data/splits/finred_entity_disjoint_v1/split_manifest.json")
RUN_ROOT = Path("runs/m1_m4_loss_family_v1")
SEEDS = (13, 42, 87, 101, 202, 303, 404, 505, 606, 707)
ARMS = (
    ("BCE", "L0"),
    ("NegDW-BCE", "L1"),
    ("Focal", "FOCAL"),
    ("ASL", "ASL"),
    ("pos_weight-BCE", "POS_WEIGHT"),
)
TRAINING = {
    "epochs": 3,
    "batch_size": 2,
    "accumulation": 8,
    "eval_batch_size": 8,
    "encoder_lr": 2e-5,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
}
LOSS_PARAMETERS = {
    "BCE": {},
    "NegDW-BCE": {"negative_target_weight": 0.1},
    "Focal": {"alpha": 0.25, "gamma": 2.0},
    "ASL": {
        "gamma_positive": 0.0,
        "gamma_negative": 4.0,
        "negative_clip": 0.05,
        "epsilon": 1e-8,
        "focusing_weight_gradients": "disabled",
    },
    "pos_weight-BCE": {
        "formula": "training_negative_count / training_positive_count per relation",
        "source": "training split only",
        "clipping": "none",
    },
}
METRICS = (
    "macro_f1_fixed_inventory",
    "micro_f1",
    "micro_precision",
    "micro_recall",
    "exact_set_match",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def relative_file(root: Path, value: str, description: str) -> Path:
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts, f"{description} must be repository-relative")
    resolved = (root / path).resolve()
    require(resolved.is_relative_to(root.resolve()), f"{description} escapes the repository")
    return resolved


def run_directory(arm: str, seed: int) -> Path:
    slug = arm.lower().replace("-", "_")
    return RUN_ROOT / f"{slug}_seed{seed}"


def current_source_hashes(root: Path) -> dict[str, str]:
    return {
        "train_encoder.py": sha256_file(root / "src/orgu/train_encoder.py"),
        "neural.py": sha256_file(root / "src/orgu/neural.py"),
        "metrics.py": sha256_file(root / "src/orgu/metrics.py"),
    }


def make_plan(
    context: dict,
    labels: list[str],
    source_paths: dict[str, Path],
    seeds: tuple[int, ...],
    root: Path,
) -> dict:
    train_path = source_paths["train"]
    validation_path = source_paths["validation"]
    return {
        "status": "planned",
        "dataset": "FinRED",
        "protocol": "frozen entity-disjoint split; C2 entity-marker inputs",
        "feature_manifest_sha256": sha256_file(root / FEATURE_DIR / "context_manifest.json"),
        "train_path": str(train_path.relative_to(root)),
        "train_sha256": sha256_file(train_path),
        "validation_path": str(validation_path.relative_to(root)),
        "validation_sha256": sha256_file(validation_path),
        "train_candidates": 5647,
        "validation_candidates": 720,
        "relation_labels": labels,
        "seeds": list(seeds),
        "loss_arms": [
            {"name": name, "loss_code": code, "parameters": LOSS_PARAMETERS[name]}
            for name, code in ARMS
        ],
        "training": TRAINING,
        "checkpoint_selection": {
            "metric": "validation micro-F1 at threshold 0.5",
            "rule": "best epoch; earliest epoch on exact tie",
        },
        "model_id": str(context["model_id"]),
        "model_revision": str(context["model_revision"]),
        "training_source_hashes": current_source_hashes(root),
        "threshold_policies": [
            "shared_threshold_0.5",
            "per_arm_validation_macro_f1_maximizing_threshold",
            "per_arm_validation_recall_matched_to_NegDW_BCE_at_0.5",
        ],
        "test_opened_during_training_or_threshold_selection": False,
        "interpretation": "Post-hoc exploratory comparison; the entity-disjoint test was previously inspected.",
    }


def stage_train(args: argparse.Namespace) -> None:
    require(ROOT.joinpath("src/orgu/train_encoder.py").is_file(), "Run this command from the repository root")
    sys.path.insert(0, str(ROOT))
    import run_m1m4_validation as preparation

    seeds = SEEDS
    context, labels, _, source_paths = preparation.prepare_inputs(
        ROOT,
        {"train": args.train_file, "validation": args.validation_file},
    )
    plan = make_plan(context, labels, source_paths, seeds, ROOT)
    plan_path = ROOT / RUN_ROOT / "experiment_plan.json"
    if plan_path.exists():
        existing = read_json(plan_path)
        comparable = dict(existing)
        comparable["status"] = "planned"
        require(comparable == plan, f"Existing experiment plan differs; preserve and inspect: {plan_path}")
    if args.check_only:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in ("src", environment.get("PYTHONPATH", "")) if part
        )
        for arm, loss_code in ARMS:
            command = [
                sys.executable, "-m", "orgu.train_encoder",
                "--feature-dir", str(FEATURE_DIR), "--arm", "C2", "--loss", loss_code,
                "--seed", str(seeds[0]), "--output-dir", str(RUN_ROOT / "preflight"),
                "--pretrained-model", str(context["model_id"]),
                "--pretrained-model-revision", str(context["model_revision"]),
                "--device", args.device, "--check-only",
            ]
            print(f"=== Preflight: {arm} ===", flush=True)
            subprocess.run(command, env=environment, check=True)
        print("All loss preflights passed; no training or test inference was run.")
        return

    if not plan_path.exists():
        write_json(plan_path, plan)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in ("src", environment.get("PYTHONPATH", "")) if part
    )
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    for arm, loss_code in ARMS:
        for seed in seeds:
            output_dir = run_directory(arm, seed)
            command = [
                sys.executable, "-m", "orgu.train_encoder",
                "--feature-dir", str(FEATURE_DIR), "--arm", "C2", "--loss", loss_code,
                "--seed", str(seed), "--output-dir", str(output_dir),
                "--epochs", str(TRAINING["epochs"]),
                "--batch-size", str(TRAINING["batch_size"]),
                "--accumulation", str(TRAINING["accumulation"]),
                "--eval-batch-size", str(TRAINING["eval_batch_size"]),
                "--encoder-lr", str(TRAINING["encoder_lr"]),
                "--weight-decay", str(TRAINING["weight_decay"]),
                "--gradient-clip", str(TRAINING["gradient_clip"]),
                "--pretrained-model", str(context["model_id"]),
                "--pretrained-model-revision", str(context["model_revision"]),
                "--device", args.device,
            ]
            print(f"\n=== {arm}, seed {seed} ===", flush=True)
            print("+", " ".join(command), flush=True)
            subprocess.run(command, env=environment, check=True)
            manifest = read_json(ROOT / output_dir / "run_manifest.json")
            require(manifest.get("status") == "completed", f"Run did not complete: {output_dir}")
            require(manifest.get("train_sha256") == plan["train_sha256"], f"Training hash mismatch: {output_dir}")
            require(manifest.get("validation_sha256") == plan["validation_sha256"], f"Validation hash mismatch: {output_dir}")
    completed_plan = dict(plan)
    completed_plan["status"] = "training_completed"
    write_json(plan_path, completed_plan)
    print(f"All {len(ARMS) * len(seeds)} matched runs finished; the test was not opened.")
    print(f"Next: python scripts/run_m1m4_loss_family_comparison.py --stage freeze-thresholds")


def prediction_arrays(rows: list[dict], labels: list[str], expected_count: int) -> tuple[np.ndarray, np.ndarray]:
    require(len(rows) == expected_count, f"Expected {expected_count} validation predictions, found {len(rows)}")
    scores = []
    targets = []
    ids = []
    for row in rows:
        score_row = row.get("scores")
        target_row = row.get("gold_targets")
        require(isinstance(score_row, list) and len(score_row) == len(labels), "Invalid prediction score vector")
        require(isinstance(target_row, list) and len(target_row) == len(labels), "Invalid gold target vector")
        require(set(target_row) <= {0, 1}, "Gold targets must be binary")
        scores.append(score_row)
        targets.append(target_row)
        ids.append(row.get("candidate_id"))
    require(all(isinstance(value, str) and value for value in ids), "Missing candidate IDs in predictions")
    require(len(ids) == len(set(ids)), "Duplicate candidate IDs in predictions")
    return np.asarray(scores, dtype=np.float32), np.asarray(targets, dtype=np.uint8)


def threshold_surface(scores: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seed_count, _, label_count = scores.shape
    thresholds = np.unique(np.concatenate((scores.reshape(-1), np.array([0.0, 0.5, 1.0], dtype=np.float32))))
    macro_f1_by_seed = np.zeros((seed_count, thresholds.size), dtype=np.float64)
    recall_by_seed = np.zeros((seed_count, thresholds.size), dtype=np.float64)
    for seed_index in range(seed_count):
        positive_total = int(targets[seed_index].sum())
        require(positive_total > 0, "Validation split must contain recorded positives")
        for label_index in range(label_count):
            all_scores = np.sort(scores[seed_index, :, label_index])
            positive_scores = np.sort(scores[seed_index, targets[seed_index, :, label_index] == 1, label_index])
            predicted_count = all_scores.size - np.searchsorted(all_scores, thresholds, side="left")
            true_positive = positive_scores.size - np.searchsorted(positive_scores, thresholds, side="left")
            false_negative = positive_scores.size - true_positive
            false_positive = predicted_count - true_positive
            denominator = 2 * true_positive + false_positive + false_negative
            f1 = np.divide(
                2 * true_positive,
                denominator,
                out=np.zeros_like(denominator, dtype=np.float64),
                where=denominator > 0,
            )
            macro_f1_by_seed[seed_index] += f1 / label_count
            recall_by_seed[seed_index] += true_positive / positive_total
    return thresholds, macro_f1_by_seed.mean(axis=0), recall_by_seed.mean(axis=0)


def choose_highest_threshold(thresholds: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    best = float(values.max())
    matching = np.flatnonzero(np.isclose(values, best, rtol=0.0, atol=1e-12))
    index = int(matching[-1])
    return float(thresholds[index]), float(values[index])


def choose_recall_threshold(thresholds: np.ndarray, recalls: np.ndarray, target: float) -> tuple[float, float]:
    distances = np.abs(recalls - target)
    closest = float(distances.min())
    matching = np.flatnonzero(np.isclose(distances, closest, rtol=0.0, atol=1e-12))
    index = int(matching[-1])
    return float(thresholds[index]), float(recalls[index])


def load_validation_predictions(root: Path, plan: dict) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict]]:
    label_list = plan["relation_labels"]
    score_arrays = {}
    target_arrays = {}
    run_evidence = {}
    reference_ids = None
    reference_targets = None
    seed_values = plan["seeds"]
    for arm, _ in ARMS:
        per_seed_scores = []
        per_seed_targets = []
        for seed in seed_values:
            output_dir = run_directory(arm, seed)
            manifest_path = root / output_dir / "run_manifest.json"
            predictions_path = root / output_dir / "validation_predictions.jsonl"
            require(manifest_path.is_file() and predictions_path.is_file(), f"Missing completed validation artifacts: {output_dir}")
            manifest = read_json(manifest_path)
            require(manifest.get("status") == "completed", f"Incomplete run: {output_dir}")
            require(manifest.get("loss_method") == arm and manifest.get("loss_arm") == dict(ARMS)[arm], f"Run identity mismatch: {output_dir}")
            require(manifest.get("seed") == seed, f"Run seed mismatch: {output_dir}")
            require(manifest.get("labels") == label_list, f"Relation order mismatch: {output_dir}")
            require(manifest.get("pretrained_model") == plan["model_id"], f"Encoder ID mismatch: {output_dir}")
            require(manifest.get("pretrained_model_revision") == plan["model_revision"], f"Encoder revision mismatch: {output_dir}")
            require(manifest.get("train_sha256") == plan["train_sha256"], f"Train split mismatch: {output_dir}")
            require(manifest.get("validation_sha256") == plan["validation_sha256"], f"Validation split mismatch: {output_dir}")
            require(manifest.get("training_code_sha256") == plan["training_source_hashes"]["train_encoder.py"], f"Training code changed: {output_dir}")
            require(manifest.get("neural_code_sha256") == plan["training_source_hashes"]["neural.py"], f"Encoder code changed: {output_dir}")
            require(manifest.get("metrics_code_sha256") == plan["training_source_hashes"]["metrics.py"], f"Metrics code changed: {output_dir}")
            run_config = manifest.get("config", {})
            for config_key, expected_value in TRAINING.items():
                require(run_config.get(config_key) == expected_value, f"Training setting {config_key} differs: {output_dir}")
            require(run_config.get("loss_parameters") == manifest.get("loss_parameters"), f"Loss configuration mismatch: {output_dir}")
            if arm != "pos_weight-BCE":
                expected_loss_config = {
                    "negative_target_weight": 0.1 if arm == "NegDW-BCE" else None,
                    "focal": LOSS_PARAMETERS[arm] if arm == "Focal" else None,
                    "asl": {
                        "gamma_positive": 0.0,
                        "gamma_negative": 4.0,
                        "negative_clip": 0.05,
                        "epsilon": 1e-8,
                        "focusing_weight_gradients": "disabled",
                    } if arm == "ASL" else None,
                    "pos_weight": None,
                }
                require(manifest.get("loss_parameters") == expected_loss_config, f"Unexpected loss parameters: {output_dir}")
            else:
                pos_config = manifest.get("loss_parameters", {}).get("pos_weight", {})
                values = pos_config.get("values_by_label", {})
                require(pos_config.get("formula") == LOSS_PARAMETERS[arm]["formula"], f"pos_weight formula mismatch: {output_dir}")
                require(set(values) == set(label_list), f"pos_weight label inventory mismatch: {output_dir}")
                require(all(np.isfinite(float(value)) and float(value) > 0.0 for value in values.values()), f"Invalid pos_weight values: {output_dir}")
            checkpoint = manifest.get("best_checkpoint", {})
            checkpoint_path = relative_file(root, checkpoint.get("path", ""), "Selected checkpoint")
            require(checkpoint_path.is_file(), f"Missing selected checkpoint: {checkpoint_path}")
            actual_checkpoint_hash = sha256_file(checkpoint_path)
            require(actual_checkpoint_hash == checkpoint.get("sha256"), f"Checkpoint hash mismatch: {checkpoint_path}")
            require(manifest.get("validation_predictions_sha256") == sha256_file(predictions_path), f"Validation predictions hash mismatch: {output_dir}")
            prediction_rows = read_jsonl(predictions_path)
            scores, targets = prediction_arrays(prediction_rows, label_list, plan["validation_candidates"])
            ids = [row["candidate_id"] for row in prediction_rows]
            if reference_ids is None:
                reference_ids = ids
                reference_targets = targets
            else:
                require(ids == reference_ids, f"Candidate order differs in {output_dir}")
                require(np.array_equal(targets, reference_targets), f"Gold labels differ in {output_dir}")
            per_seed_scores.append(scores)
            per_seed_targets.append(targets)
            run_evidence[f"{arm}_seed{seed}"] = {
                "run_manifest_sha256": sha256_file(manifest_path),
                "checkpoint_sha256": actual_checkpoint_hash,
                "validation_predictions_sha256": sha256_file(predictions_path),
            }
        score_arrays[arm] = np.stack(per_seed_scores)
        target_arrays[arm] = np.stack(per_seed_targets)
    return score_arrays, target_arrays, run_evidence


def stage_freeze_thresholds() -> None:
    root = ROOT
    plan_path = root / RUN_ROOT / "experiment_plan.json"
    require(plan_path.is_file(), f"Missing experiment plan: {plan_path}")
    plan = read_json(plan_path)
    require(plan.get("status") == "training_completed", "All planned training runs must finish before threshold selection")
    require(plan.get("test_opened_during_training_or_threshold_selection") is False, "Experiment plan does not preserve test lock")
    scores, targets, evidence = load_validation_predictions(root, plan)
    validation_summary = {}
    thresholds_by_policy = {
        "shared_threshold_0.5": {arm: 0.5 for arm, _ in ARMS},
        "validation_macro_f1": {},
        "validation_recall_matched_to_NegDW_BCE_at_0.5": {},
    }
    negdw_scores = scores["NegDW-BCE"]
    negdw_targets = targets["NegDW-BCE"]
    negdw_thresholds, _, negdw_recalls = threshold_surface(negdw_scores, negdw_targets)
    index_05 = int(np.flatnonzero(np.isclose(negdw_thresholds, 0.5, rtol=0.0, atol=1e-7))[0])
    recall_target = float(negdw_recalls[index_05])
    for arm, _ in ARMS:
        thresholds, mean_macro_f1, mean_recall = threshold_surface(scores[arm], targets[arm])
        macro_threshold, max_macro_f1 = choose_highest_threshold(thresholds, mean_macro_f1)
        recall_threshold, matched_recall = choose_recall_threshold(thresholds, mean_recall, recall_target)
        if arm == "NegDW-BCE":
            recall_threshold = 0.5
            matched_recall = recall_target
        thresholds_by_policy["validation_macro_f1"][arm] = macro_threshold
        thresholds_by_policy["validation_recall_matched_to_NegDW_BCE_at_0.5"][arm] = recall_threshold
        validation_summary[arm] = {
            "macro_f1_threshold": macro_threshold,
            "validation_mean_macro_f1_at_selected_threshold": max_macro_f1,
            "recall_matched_threshold": recall_threshold,
            "validation_mean_recall_at_recall_threshold": matched_recall,
        }
    policy = {
        "status": "frozen_before_test_evaluation",
        "experiment_plan_sha256": sha256_file(plan_path),
        "seeds": plan["seeds"],
        "relation_labels": plan["relation_labels"],
        "train_sha256": plan["train_sha256"],
        "validation_sha256": plan["validation_sha256"],
        "validation_run_evidence": evidence,
        "thresholds": thresholds_by_policy,
        "selection_details": {
            "macro_f1": "One global threshold per method maximizes mean fixed-inventory validation macro-F1 over seeds; ties choose the highest threshold.",
            "recall_matched": "Each method is matched to the mean validation micro-recall of NegDW-BCE at 0.5; NegDW-BCE remains fixed at 0.5; ties choose the highest threshold.",
            "recall_target": recall_target,
            "validation_diagnostics": validation_summary,
        },
        "test_file_read": False,
        "interpretation": "Thresholds are validation-selected; validation also selected checkpoints, and test outcomes remain post-hoc.",
    }
    output_path = root / RUN_ROOT / "threshold_policy.json"
    if output_path.exists():
        require(read_json(output_path) == policy, f"Frozen threshold policy differs; preserve and inspect: {output_path}")
    else:
        write_json(output_path, policy)
    print(f"Frozen all three operating points without opening test: {output_path}")
    print(f"Validation recall target: {recall_target:.6f}")


def validate_test_rows(rows: list[dict], labels: list[str], expected_count: int) -> None:
    require(len(rows) == expected_count, f"Expected {expected_count} test candidates, found {len(rows)}")
    ids = [row.get("id") for row in rows]
    require(all(isinstance(value, str) and value for value in ids), "Test candidates must have IDs")
    require(len(ids) == len(set(ids)), "Test candidate IDs are not unique")
    label_set = set(labels)
    for row in rows:
        require(row.get("split") == "test", f"Candidate {row['id']} is not marked as test")
        require(isinstance(row.get("record_id"), str) and row["record_id"], f"Candidate {row['id']} lacks record_id")
        row_labels = row.get("labels")
        require(isinstance(row_labels, list) and set(row_labels) <= label_set, f"Invalid labels for test candidate {row['id']}")
        model_inputs = row.get("model_inputs", row)
        require(isinstance(model_inputs.get("input_ids"), list), f"Test candidate {row['id']} lacks encoded input_ids")


def metric_at_threshold(predictions: list[dict], labels: list[str], threshold: float) -> dict:
    from orgu.metrics import multilabel_metrics

    gold_sets = [set(row["gold_labels"]) for row in predictions]
    predicted_sets = [
        {label for label, score in zip(labels, row["scores"]) if float(score) >= threshold}
        for row in predictions
    ]
    return multilabel_metrics(gold_sets, predicted_sets, labels)


def aggregate_seed_metrics(per_seed: dict[str, dict]) -> dict:
    result = {}
    for metric in METRICS:
        values = [float(per_seed[seed][metric]) for seed in sorted(per_seed, key=int)]
        result[metric] = {
            "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "seed_values": {seed: float(per_seed[seed][metric]) for seed in sorted(per_seed, key=int)},
        }
    return result


def stage_evaluate_test(args: argparse.Namespace) -> None:
    root = ROOT
    plan_path = root / RUN_ROOT / "experiment_plan.json"
    policy_path = root / RUN_ROOT / "threshold_policy.json"
    evaluation_manifest_path = root / RUN_ROOT / "test_evaluation_manifest.json"
    require(plan_path.is_file() and policy_path.is_file(), "Train runs and freeze validation thresholds before test inference")
    require(not evaluation_manifest_path.exists(), f"Test evaluation already exists; preserve it: {evaluation_manifest_path}")
    plan = read_json(plan_path)
    policy = read_json(policy_path)
    require(plan.get("status") == "training_completed", "Training is not complete")
    require(policy.get("status") == "frozen_before_test_evaluation", "Threshold policy is not frozen")
    require(policy.get("experiment_plan_sha256") == sha256_file(plan_path), "Threshold policy belongs to a different experiment plan")
    scores, targets, current_evidence = load_validation_predictions(root, plan)
    require(policy.get("validation_run_evidence") == current_evidence, "Validation runs changed after threshold freeze")
    require(policy.get("test_file_read") is False, "Threshold policy was created after test access")

    sys.path.insert(0, str(root / "src"))
    import torch
    from safetensors.torch import load_file
    from orgu import neural, train_encoder

    feature_data = train_encoder.load_feature_data(FEATURE_DIR)
    require(feature_data["labels"] == plan["relation_labels"], "Test labels differ from frozen relation inventory")
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        require(device.type != "cuda" or torch.cuda.is_available(), "CUDA was selected but is unavailable")

    predictions_root = root / RUN_ROOT / "test_predictions"
    require(not predictions_root.exists(), f"Partial test predictions exist; preserve and inspect: {predictions_root}")
    require(not (root / RUN_ROOT / "test_metrics.json").exists(), "Test summary already exists; preserve it")
    test_path = relative_file(root, args.test_file or "data/splits/finred_entity_disjoint_v1/test.jsonl", "Test file")
    split_manifest_path = root / SPLIT_MANIFEST
    require(split_manifest_path.is_file(), f"Missing frozen split manifest: {split_manifest_path}")
    split_manifest = read_json(split_manifest_path)
    test_info = split_manifest.get("files", {}).get("test", {})
    expected_test_hash = test_info.get("sha256")
    expected_test_count = split_manifest.get("counts", {}).get("test")
    require(isinstance(expected_test_hash, str) and isinstance(expected_test_count, int), "Split manifest lacks frozen test identity")
    require(test_path.is_file(), f"Missing frozen test candidates: {test_path}")
    actual_test_hash = sha256_file(test_path)
    require(actual_test_hash == expected_test_hash, "Test SHA-256 differs from the frozen entity-disjoint manifest")
    test_rows = read_jsonl(test_path)
    validate_test_rows(test_rows, plan["relation_labels"], expected_test_count)
    training_ids = {row.get("id") for row in feature_data["train_rows"]}
    validation_ids = {row.get("id") for row in feature_data["validation_rows"]}
    test_ids = {row.get("id") for row in test_rows}
    require(not (training_ids & test_ids), "Candidate IDs overlap between train and test")
    require(not (validation_ids & test_ids), "Candidate IDs overlap between validation and test")
    predictions_root.mkdir(parents=True)
    test_seed_metrics = {policy_name: {arm: {} for arm, _ in ARMS} for policy_name in policy["thresholds"]}
    prediction_artifacts = {}
    checkpoint_evidence = {}
    for arm, _ in ARMS:
        for seed in plan["seeds"]:
            run_dir = root / run_directory(arm, seed)
            run_manifest_path = run_dir / "run_manifest.json"
            run_manifest = read_json(run_manifest_path)
            checkpoint = run_manifest.get("best_checkpoint", {})
            checkpoint_path = relative_file(root, checkpoint.get("path", ""), "Selected checkpoint")
            require(checkpoint_path.is_file(), f"Missing selected checkpoint: {checkpoint_path}")
            checkpoint_hash = sha256_file(checkpoint_path)
            require(checkpoint_hash == checkpoint.get("sha256"), f"Checkpoint hash mismatch: {checkpoint_path}")
            require(run_manifest.get("labels") == plan["relation_labels"], f"Run label order mismatch: {run_dir}")
            require(run_manifest.get("train_sha256") == plan["train_sha256"], f"Run train hash mismatch: {run_dir}")
            require(run_manifest.get("validation_sha256") == plan["validation_sha256"], f"Run validation hash mismatch: {run_dir}")
            model = neural.EntityMarkerEncoder(
                run_manifest["pretrained_model"],
                feature_data["tokenizer"].get_vocab_size(with_added_tokens=True),
                len(plan["relation_labels"]),
                revision=run_manifest["pretrained_model_revision"],
            ).to(device)
            model.load_state_dict(load_file(str(checkpoint_path), device="cpu"), strict=True)
            _, predictions = train_encoder.evaluate(
                model,
                test_rows,
                plan["relation_labels"],
                feature_data["tokenizer"],
                feature_data["marker_ids"],
                device,
                int(run_manifest["config"]["eval_batch_size"]),
                split_sha256=actual_test_hash,
            )
            output_name = f"{arm.lower().replace('-', '_')}_seed{seed}.jsonl"
            output_path = predictions_root / output_name
            write_jsonl(output_path, predictions)
            prediction_artifacts[f"{arm}_seed{seed}"] = {
                "path": str(output_path.relative_to(root)),
                "sha256": sha256_file(output_path),
                "checkpoint_sha256": checkpoint_hash,
            }
            checkpoint_evidence[f"{arm}_seed{seed}"] = checkpoint_hash
            for policy_name, threshold_by_arm in policy["thresholds"].items():
                metrics = metric_at_threshold(predictions, plan["relation_labels"], float(threshold_by_arm[arm]))
                test_seed_metrics[policy_name][arm][str(seed)] = {
                    "threshold": float(threshold_by_arm[arm]),
                    **metrics,
                }
            del model

    summaries = {}
    paired_differences = {}
    for policy_name, arms_metrics in test_seed_metrics.items():
        summaries[policy_name] = {
            arm: aggregate_seed_metrics(arms_metrics[arm]) for arm, _ in ARMS
        }
        paired_differences[policy_name] = {}
        for arm, _ in ARMS:
            if arm == "BCE":
                continue
            paired_differences[policy_name][arm] = {}
            for baseline in ("BCE", "NegDW-BCE"):
                if arm == baseline:
                    continue
                paired_differences[policy_name][arm][f"minus_{baseline}"] = {}
                for metric in METRICS:
                    deltas = [
                        float(arms_metrics[arm][str(seed)][metric])
                        - float(arms_metrics[baseline][str(seed)][metric])
                        for seed in plan["seeds"]
                    ]
                    paired_differences[policy_name][arm][f"minus_{baseline}"][metric] = {
                        "mean": statistics.mean(deltas),
                        "sample_sd": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                        "seed_differences": {str(seed): delta for seed, delta in zip(plan["seeds"], deltas)},
                    }
    report = {
        "status": "posthoc_exploratory",
        "dataset": "FinRED entity-disjoint test",
        "test_file": str(test_path.relative_to(root)),
        "test_sha256": actual_test_hash,
        "test_candidates": len(test_rows),
        "test_previously_inspected": True,
        "test_read_after_thresholds_frozen": True,
        "seeds": plan["seeds"],
        "loss_arms": [name for name, _ in ARMS],
        "threshold_policy_sha256": sha256_file(policy_path),
        "thresholds": policy["thresholds"],
        "metrics_mean_sample_sd": summaries,
        "paired_seed_differences": paired_differences,
        "per_seed_metrics": test_seed_metrics,
        "prediction_artifacts": prediction_artifacts,
        "checkpoint_hashes": checkpoint_evidence,
        "interpretation": "Metrics are agreement with recorded annotations on a previously inspected test, not independent confirmation or verified relation truth.",
    }
    summary_path = root / RUN_ROOT / "test_metrics.json"
    write_json(summary_path, report)
    evaluation_manifest = {
        "status": "completed_posthoc_test_evaluation",
        "test_sha256": actual_test_hash,
        "threshold_policy_sha256": sha256_file(policy_path),
        "experiment_plan_sha256": sha256_file(plan_path),
        "summary_path": str(summary_path.relative_to(root)),
        "summary_sha256": sha256_file(summary_path),
        "prediction_artifacts": prediction_artifacts,
        "test_previously_inspected": True,
    }
    write_json(evaluation_manifest_path, evaluation_manifest)
    print(f"Post-hoc test report: {summary_path}")
    print("Test was opened only after the validation threshold policy was verified frozen.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Matched 10-seed FinRED loss-family experiment")
    parser.add_argument("--stage", choices=("train", "freeze-thresholds", "evaluate-test"), required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "cuda:0"), default="auto")
    parser.add_argument("--train-file", help="Repository-relative frozen train JSONL override.")
    parser.add_argument("--validation-file", help="Repository-relative frozen validation JSONL override.")
    parser.add_argument("--test-file", help="Repository-relative frozen test JSONL; accepted only during evaluate-test.")
    parser.add_argument("--check-only", action="store_true", help="Stage and preflight train/dev inputs without training.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.stage == "train":
            require(not args.test_file, "Test path is accepted only during evaluate-test")
            stage_train(args)
        elif args.stage == "freeze-thresholds":
            require(not args.check_only and not args.test_file and not args.train_file and not args.validation_file,
                    "Threshold freezing does not accept training or test overrides")
            stage_freeze_thresholds()
        else:
            require(not args.check_only and not args.train_file and not args.validation_file,
                    "Test evaluation uses the frozen experiment plan only")
            stage_evaluate_test(args)
    except Exception as exc:
        print(f"[loss-family] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())