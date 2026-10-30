#!/usr/bin/env python3
"""Run paired controlled-positive-label-removal experiments on local C2 files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path.cwd().resolve()
DEFAULT_FEATURE_DIR = Path("data/features/finred_entity_disjoint_v1")
DEFAULT_OUTPUT_DIR = Path("runs/controlled_label_removal_local_v1")
DEFAULT_SEEDS = (13, 42, 87)
REMOVAL_FRACTIONS = (0.0, 0.1, 0.3, 0.5)
LOSS_ARMS = (("BCE", "L0"), ("NegDW-BCE", "L1"))
METRICS = (
    "macro_f1_fixed_inventory",
    "micro_f1",
    "micro_precision",
    "micro_recall",
    "exact_set_match",
)
TRAINING_DEFAULTS = {
    "epochs": 3,
    "batch_size": 2,
    "accumulation": 8,
    "eval_batch_size": 8,
    "encoder_lr": 2e-5,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def reject_upload_path(path: Path, description: str) -> None:
    require(
        all(part.casefold() != "private-uploads" for part in path.parts),
        f"{description} must not use a private staging directory: {path}",
    )


def project_path(
    root: Path,
    value: str | Path,
    description: str,
    *,
    must_exist: bool,
) -> Path:
    path = Path(value)
    require(not path.is_absolute(), f"{description} must be repository-relative: {path}")
    require(".." not in path.parts, f"{description} cannot contain '..': {path}")
    reject_upload_path(path, description)
    try:
        resolved = (root / path).resolve(strict=must_exist)
    except FileNotFoundError as exc:
        raise RuntimeError(f"{description} does not exist: {path}") from exc
    require(resolved.is_relative_to(root.resolve()), f"{description} escapes the repository: {path}")
    reject_upload_path(resolved, description)
    return resolved


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )


def write_bytes_preserving(path: Path, content: bytes) -> None:
    if path.is_file():
        require(path.read_bytes() == content, f"Existing artifact differs; refusing to overwrite: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def label_inventory(schema_path: Path) -> list[str]:
    schema = read_json(schema_path)
    labels = schema.get("labels")
    require(isinstance(labels, list) and labels, "Label schema must contain a nonempty labels list")
    if isinstance(labels[0], dict):
        labels = [entry.get("name", entry.get("label")) for entry in labels]
    labels = [str(label) for label in labels]
    require(len(labels) == len(set(labels)), "Label schema contains duplicate labels")
    return labels


def validate_rows(rows: list[dict], labels: list[str], split_name: str) -> list[str]:
    require(bool(rows), f"{split_name} file is empty")
    label_set = set(labels)
    candidate_ids = []
    for index, row in enumerate(rows):
        candidate_id = row.get("id")
        require(isinstance(candidate_id, str) and candidate_id, f"Missing candidate id in {split_name} row {index}")
        candidate_ids.append(candidate_id)
        row_labels = row.get("labels")
        require(isinstance(row_labels, list), f"Missing labels list in {split_name} row {index}")
        require(len(row_labels) == len(set(row_labels)), f"Duplicate labels in {split_name} row {index}")
        unknown = set(row_labels) - label_set
        require(not unknown, f"Unknown labels in {split_name} row {index}: {sorted(unknown)}")
        model_inputs = row.get("model_inputs", row)
        require(isinstance(model_inputs, dict), f"Invalid model_inputs in {split_name} row {index}")
        require(isinstance(model_inputs.get("input_ids"), list), f"Missing encoded input_ids in {split_name} row {index}")
    require(len(candidate_ids) == len(set(candidate_ids)), f"Duplicate candidate ids in {split_name}")
    return candidate_ids


def resolve_sources(args: argparse.Namespace) -> dict:
    feature_dir = project_path(ROOT, args.feature_dir, "Feature directory", must_exist=True)
    require(feature_dir.is_dir(), f"Feature directory is not a directory: {feature_dir}")
    manifest_path = feature_dir / "context_manifest.json"
    tokenizer_path = feature_dir / "tokenizer.json"
    schema_path = feature_dir / "label_schema.json"
    for path in (manifest_path, tokenizer_path, schema_path):
        require(path.is_file(), f"Missing local feature asset: {path.relative_to(ROOT)}")

    manifest = read_json(manifest_path)
    require(manifest.get("test_inference_run") is False, "Feature manifest must preserve the test lock")
    files = manifest.get("arms", {}).get("C2", {}).get("files", {})
    require("train" in files and "validation" in files, "Feature manifest lacks C2 train/validation entries")

    train_value = args.train_file or files["train"].get("path")
    validation_value = args.validation_file or files["validation"].get("path")
    require(train_value and validation_value, "Train and validation paths are required")
    train_path = project_path(ROOT, train_value, "Train file", must_exist=True)
    validation_path = project_path(ROOT, validation_value, "Validation file", must_exist=True)
    for split_name, path in (("train", train_path), ("validation", validation_path)):
        require(path.is_file(), f"Missing local {split_name} JSONL: {path.relative_to(ROOT)}")
        expected_hash = files[split_name].get("sha256")
        require(isinstance(expected_hash, str), f"Manifest lacks the frozen {split_name} hash")
        require(sha256_file(path) == expected_hash, f"Frozen {split_name} SHA-256 mismatch: {path.relative_to(ROOT)}")
    require(train_path != validation_path, "Train and validation files must differ")

    labels = label_inventory(schema_path)
    expected_schema_hash = manifest.get("label_schema_sha256")
    require(expected_schema_hash == sha256_file(schema_path), "Label-schema SHA-256 differs from the feature manifest")
    train_rows = read_jsonl(train_path)
    validation_rows = read_jsonl(validation_path)
    train_ids = validate_rows(train_rows, labels, "train")
    validation_ids = validate_rows(validation_rows, labels, "validation")
    require(not (set(train_ids) & set(validation_ids)), "Train and validation candidate IDs overlap")
    require(len(train_rows) == 5647, f"Expected the frozen entity-disjoint train count 5647, found {len(train_rows)}")
    require(len(validation_rows) == 720, f"Expected the frozen entity-disjoint validation count 720, found {len(validation_rows)}")
    require(sum(bool(row.get("eligible_for_training", True)) for row in train_rows) == 5647,
            "Expected all 5,647 frozen train candidates to be eligible")
    require(sum(len(row["labels"]) for row in train_rows) > 0, "Training file has no recorded positive labels")

    return {
        "feature_dir": feature_dir,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "tokenizer_path": tokenizer_path,
        "schema_path": schema_path,
        "labels": labels,
        "train_path": train_path,
        "validation_path": validation_path,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "train_ids": train_ids,
        "validation_ids": validation_ids,
    }


def build_mask_schedule(train_rows: list[dict], labels: list[str], seed: int) -> dict[float, tuple[list[dict], dict]]:
    positive_slots = [
        (row_index, label)
        for row_index, row in enumerate(train_rows)
        if row.get("eligible_for_training", True)
        for label in labels
        if label in row["labels"]
    ]
    require(bool(positive_slots), "No eligible positive label assignments to mask")
    permutation = np.random.default_rng(seed).permutation(len(positive_slots)).tolist()
    schedule = {}
    previous_removed: set[tuple[int, str]] = set()
    for fraction in REMOVAL_FRACTIONS:
        remove_count = int(math.floor(len(positive_slots) * fraction + 0.5))
        selected = {positive_slots[index] for index in permutation[:remove_count]}
        require(previous_removed.issubset(selected), "Generated omission masks are not nested")
        masked_rows = []
        for row_index, row in enumerate(train_rows):
            updated = dict(row)
            updated["labels"] = [
                label for label in row["labels"] if (row_index, label) not in selected
            ]
            masked_rows.append(updated)
        dropped_by_label = {
            label: sum(slot_label == label for _, slot_label in selected)
            for label in labels
        }
        mask_payload = json.dumps(
            [[row_index, label] for row_index, label in sorted(selected)],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        info = {
            "fraction_requested": fraction,
            "positive_assignments_before": len(positive_slots),
            "positive_assignments_removed": remove_count,
            "fraction_realized": remove_count / len(positive_slots),
            "dropped_by_label": dropped_by_label,
            "mask_sha256": sha256_bytes(mask_payload),
            "rng": "numpy.default_rng(seed).permutation over row-order and schema-label-order slots",
        }
        schedule[fraction] = (masked_rows, info)
        previous_removed = selected
    return schedule


def removal_slug(fraction: float) -> str:
    return f"remove_{int(round(fraction * 100)):02d}pct"


def arm_slug(arm: str) -> str:
    return arm.lower().replace("-", "_")


def prepare_feature_dir(
    root: Path,
    output_dir: Path,
    sources: dict,
    fraction: float,
    seed: int,
    masked_rows: list[dict],
    mask_info: dict,
) -> tuple[Path, dict]:
    feature_dir = output_dir / "prepared" / removal_slug(fraction) / f"seed_{seed}" / "feature"
    c2_dir = feature_dir / "C2"
    train_out = c2_dir / "train.jsonl"
    write_bytes_preserving(train_out, jsonl_bytes(masked_rows))
    write_bytes_preserving(feature_dir / "tokenizer.json", sources["tokenizer_path"].read_bytes())
    write_bytes_preserving(feature_dir / "label_schema.json", sources["schema_path"].read_bytes())

    train_counts = {label: 0 for label in sources["labels"]}
    for row in masked_rows:
        if row.get("eligible_for_training", True):
            for label in row["labels"]:
                train_counts[label] += 1
    validation_counts = {label: 0 for label in sources["labels"]}
    for row in sources["validation_rows"]:
        for label in row["labels"]:
            validation_counts[label] += 1

    train_relative = train_out.relative_to(root).as_posix()
    validation_relative = sources["validation_path"].relative_to(root).as_posix()
    generated_manifest = {
        "format_version": sources["manifest"].get("format_version", 1),
        "test_inference_run": False,
        "model_id": sources["manifest"].get("model_id"),
        "model_revision": sources["manifest"].get("model_revision"),
        "max_length": sources["manifest"].get("max_length", 512),
        "label_schema_sha256": sha256_file(sources["schema_path"]),
        "arms": {
            "C2": {
                "files": {
                    "train": {
                        "path": train_relative,
                        "sha256": sha256_file(train_out),
                        "candidates": len(masked_rows),
                        "eligible_for_training": sum(
                            bool(row.get("eligible_for_training", True)) for row in masked_rows
                        ),
                        "unique_label_assignments": sum(train_counts.values()),
                        "label_counts": train_counts,
                    },
                    "validation": {
                        "path": validation_relative,
                        "sha256": sha256_file(sources["validation_path"]),
                        "candidates": len(sources["validation_rows"]),
                        "unique_label_assignments": sum(validation_counts.values()),
                        "label_counts": validation_counts,
                    },
                }
            }
        },
        "controlled_positive_label_removal": {
            "fraction": fraction,
            "seed": seed,
            **mask_info,
        },
    }
    write_bytes_preserving(feature_dir / "context_manifest.json", json_bytes(generated_manifest))
    return feature_dir, generated_manifest


def job_directory(output_dir: Path, fraction: float, seed: int, arm: str) -> Path:
    return output_dir / removal_slug(fraction) / f"seed_{seed}" / arm_slug(arm)


def build_training_command(
    feature_dir: Path,
    output_dir: Path,
    arm_code: str,
    seed: int,
    args: argparse.Namespace,
    sources: dict,
    *,
    check_only: bool,
) -> list[str]:
    manifest = sources["manifest"]
    model_id = args.pretrained_model or manifest.get("model_id")
    require(isinstance(model_id, str) and model_id, "No pretrained model ID is configured")
    model_path = Path(model_id)
    require(not model_path.is_absolute(), "Pretrained model paths must be repository-relative")
    require(".." not in model_path.parts, "Pretrained model path cannot contain '..'")
    reject_upload_path(model_path, "Pretrained model")
    if model_path.exists():
        resolved_model_path = model_path.resolve()
        require(resolved_model_path.is_relative_to(ROOT), "Local pretrained model must remain inside the repository")
        reject_upload_path(resolved_model_path, "Pretrained model")
    revision = args.pretrained_model_revision
    if revision is None and model_id == manifest.get("model_id"):
        revision = manifest.get("model_revision")
    command = [
        sys.executable,
        "-m",
        "orgu.train_encoder",
        "--feature-dir",
        feature_dir.relative_to(ROOT).as_posix(),
        "--arm",
        "C2",
        "--loss",
        arm_code,
        "--seed",
        str(seed),
        "--output-dir",
        output_dir.relative_to(ROOT).as_posix(),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--accumulation",
        str(args.accumulation),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--encoder-lr",
        str(args.encoder_lr),
        "--weight-decay",
        str(args.weight_decay),
        "--gradient-clip",
        str(args.gradient_clip),
        "--pretrained-model",
        model_id,
        "--device",
        args.device,
    ]
    if revision:
        command.extend(("--pretrained-model-revision", str(revision)))
    if check_only:
        command.append("--check-only")
    return command


def run_trainer(command: list[str], root: Path) -> None:
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in ("src", existing_pythonpath) if part
    )
    result = subprocess.run(command, cwd=root, env=environment, check=False)
    require(result.returncode == 0, f"Training subprocess failed with exit code {result.returncode}")


def load_prediction_arrays(path: Path, labels: list[str], expected_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    rows = read_jsonl(path)
    require(len(rows) == len(expected_ids), f"Unexpected prediction count in {path}")
    scores = []
    targets = []
    ids = []
    for row in rows:
        score_vector = row.get("scores")
        target_vector = row.get("gold_targets")
        require(isinstance(score_vector, list) and len(score_vector) == len(labels), f"Invalid scores in {path}")
        require(isinstance(target_vector, list) and len(target_vector) == len(labels), f"Invalid gold targets in {path}")
        require(set(target_vector) <= {0, 1}, f"Non-binary gold targets in {path}")
        scores.append(score_vector)
        targets.append(target_vector)
        ids.append(row.get("candidate_id"))
    require(ids == expected_ids, f"Candidate order differs from the frozen validation set in {path}")
    return np.asarray(scores, dtype=np.float64), np.asarray(targets, dtype=np.uint8)


def threshold_surface(scores: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    require(scores.ndim == 3 and scores.shape == targets.shape, "Threshold arrays must be [seed, candidate, label]")
    seed_count, _, label_count = scores.shape
    thresholds = np.unique(
        np.concatenate(
            (scores.reshape(-1), np.asarray([0.0, 0.5, 1.0, 1.000001], dtype=np.float64))
        )
    )
    macro_f1_by_seed = np.zeros((seed_count, thresholds.size), dtype=np.float64)
    for seed_index in range(seed_count):
        for label_index in range(label_count):
            class_scores = scores[seed_index, :, label_index]
            class_targets = targets[seed_index, :, label_index]
            sorted_scores = np.sort(class_scores)
            sorted_positive_scores = np.sort(class_scores[class_targets == 1])
            predicted = sorted_scores.size - np.searchsorted(sorted_scores, thresholds, side="left")
            true_positive = sorted_positive_scores.size - np.searchsorted(
                sorted_positive_scores, thresholds, side="left"
            )
            false_negative = int(class_targets.sum()) - true_positive
            false_positive = predicted - true_positive
            denominator = 2 * true_positive + false_positive + false_negative
            class_f1 = np.divide(
                2 * true_positive,
                denominator,
                out=np.zeros_like(denominator, dtype=np.float64),
                where=denominator > 0,
            )
            macro_f1_by_seed[seed_index] += class_f1 / label_count
    return thresholds, macro_f1_by_seed.mean(axis=0)


def select_macro_f1_threshold(scores: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    thresholds, macro_f1 = threshold_surface(scores, targets)
    best_value = float(macro_f1.max())
    best_indices = np.flatnonzero(np.isclose(macro_f1, best_value, rtol=0.0, atol=1e-12))
    index = int(best_indices[-1])
    return float(thresholds[index]), best_value


def evaluate_scores(scores: np.ndarray, targets: np.ndarray, labels: list[str], threshold: float) -> dict:
    adjusted = np.asarray(scores >= threshold, dtype=np.uint8)
    gold_sets = [
        {labels[index] for index, value in enumerate(row) if value}
        for row in targets
    ]
    predicted_sets = [
        {labels[index] for index, value in enumerate(row) if value}
        for row in adjusted
    ]
    class_f1_values = []
    total_tp = total_fp = total_fn = 0
    for label_index in range(len(labels)):
        true_positive = int(np.sum((targets[:, label_index] == 1) & (adjusted[:, label_index] == 1)))
        false_positive = int(np.sum((targets[:, label_index] == 0) & (adjusted[:, label_index] == 1)))
        false_negative = int(np.sum((targets[:, label_index] == 1) & (adjusted[:, label_index] == 0)))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        class_f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        total_tp += true_positive
        total_fp += false_positive
        total_fn += false_negative
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    return {
        "candidate_count": int(targets.shape[0]),
        "macro_f1_fixed_inventory": float(np.mean(class_f1_values)) if class_f1_values else 0.0,
        "micro_f1": micro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "exact_set_match": (
            sum(gold == predicted for gold, predicted in zip(gold_sets, predicted_sets)) / len(gold_sets)
            if gold_sets
            else 0.0
        ),
    }


def summarize_seed_metrics(seed_metrics: dict[str, dict]) -> dict:
    summary = {}
    for metric in METRICS:
        values = [float(entry[metric]) for entry in seed_metrics.values()]
        summary[metric] = {
            "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "seed_values": {seed: float(entry[metric]) for seed, entry in seed_metrics.items()},
        }
    return summary


def paired_metric_difference(left: dict[str, dict], right: dict[str, dict]) -> dict:
    differences = {}
    for metric in METRICS:
        per_seed = {
            seed: float(left[seed][metric]) - float(right[seed][metric])
            for seed in left
        }
        values = list(per_seed.values())
        differences[metric] = {
            "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "seed_differences": per_seed,
        }
    return differences


def build_configuration(args: argparse.Namespace, sources: dict, prepared: dict) -> dict:
    masks = {}
    for (fraction, seed), entry in sorted(prepared.items()):
        mask_key = f"{removal_slug(fraction)}_seed{seed}"
        train_path = entry["feature_dir"] / "C2" / "train.jsonl"
        masks[mask_key] = {
            **entry["mask_info"],
            "masked_train_sha256": sha256_file(train_path),
            "masked_train_path": train_path.relative_to(ROOT).as_posix(),
        }
    model_id = args.pretrained_model or sources["manifest"].get("model_id")
    model_revision = args.pretrained_model_revision
    if model_revision is None and model_id == sources["manifest"].get("model_id"):
        model_revision = sources["manifest"].get("model_revision")
    return {
        "dataset": "FinRED entity-disjoint frozen C2 split",
        "train_path": sources["train_path"].relative_to(ROOT).as_posix(),
        "train_sha256": sha256_file(sources["train_path"]),
        "validation_path": sources["validation_path"].relative_to(ROOT).as_posix(),
        "validation_sha256": sha256_file(sources["validation_path"]),
        "feature_manifest_sha256": sha256_file(sources["manifest_path"]),
        "tokenizer_sha256": sha256_file(sources["tokenizer_path"]),
        "label_schema_sha256": sha256_file(sources["schema_path"]),
        "candidate_counts": {"train": len(sources["train_rows"]), "validation": len(sources["validation_rows"])},
        "labels": sources["labels"],
        "seeds": list(args.seeds),
        "removal_fractions": list(REMOVAL_FRACTIONS),
        "masking": {
            "method": "uniform positive-assignment removal; per-seed nested masks, shared across losses",
            "count_rule": "round half up on eligible positive assignments",
            "rng": "numpy.default_rng(seed).permutation",
            "numpy_version": np.__version__,
        },
        "runner_code_sha256": sha256_file(Path(__file__).resolve()),
        "masks": masks,
        "loss_arms": [
            {"name": name, "code": code, "negative_target_weight": 0.1 if code == "L1" else None}
            for name, code in LOSS_ARMS
        ],
        "threshold_policies": {
            "fixed": 0.5,
            "validation_tuned": "one global threshold per removal fraction and loss, maximizing mean seed validation fixed-inventory macro-F1; highest threshold breaks ties",
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "accumulation": args.accumulation,
            "eval_batch_size": args.eval_batch_size,
            "encoder_lr": args.encoder_lr,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "device": args.device,
            "pretrained_model": model_id,
            "pretrained_model_revision": model_revision,
        },
        "evaluation_is_opened_only_after_thresholds_are_frozen": True,
    }


def write_plan(path: Path, configuration: dict, updates: dict | None = None) -> dict:
    if path.is_file():
        plan = read_json(path)
        require(plan.get("configuration") == configuration,
                f"Existing experiment plan has a different configuration: {path.relative_to(ROOT)}")
    else:
        plan = {
            "status": "planned",
            "experiment": "controlled positive-label removal",
            "interpretation": "Synthetic training-label omissions measure robustness to simulated omissions, not recovery of naturally missing relations.",
            "configuration": configuration,
            "completed_runs": [],
        }
    if updates:
        plan.update(updates)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(plan))
    return plan


def validate_evaluation_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = project_path(ROOT, value, "Evaluation file", must_exist=True)
    require(path.is_file(), f"Evaluation file is not a file: {path.relative_to(ROOT)}")
    return path


def prepare_all_jobs(args: argparse.Namespace, sources: dict, output_dir: Path) -> dict:
    prepared = {}
    for seed in args.seeds:
        schedule = build_mask_schedule(sources["train_rows"], sources["labels"], seed)
        for fraction in REMOVAL_FRACTIONS:
            masked_rows, mask_info = schedule[fraction]
            feature_dir, manifest = prepare_feature_dir(
                ROOT, output_dir, sources, fraction, seed, masked_rows, mask_info
            )
            prepared[(fraction, seed)] = {
                "feature_dir": feature_dir,
                "manifest": manifest,
                "mask_info": mask_info,
            }
    return prepared


def run_preflight(args: argparse.Namespace, sources: dict, prepared: dict, output_dir: Path) -> None:
    fraction = REMOVAL_FRACTIONS[0]
    seed = args.seeds[0]
    feature_dir = prepared[(fraction, seed)]["feature_dir"]
    run_dir = job_directory(output_dir, fraction, seed, "BCE")
    command = build_training_command(feature_dir, run_dir, "L0", seed, args, sources, check_only=True)
    run_trainer(command, ROOT)
    print("Preflight passed. No model was trained and no evaluation split was read.")


def train_all(args: argparse.Namespace, sources: dict, prepared: dict, output_dir: Path, plan: dict) -> dict:
    completed = set(plan.get("completed_runs", []))
    for fraction in REMOVAL_FRACTIONS:
        for seed in args.seeds:
            feature_dir = prepared[(fraction, seed)]["feature_dir"]
            for arm_name, arm_code in LOSS_ARMS:
                run_dir = job_directory(output_dir, fraction, seed, arm_name)
                job_key = f"{removal_slug(fraction)}:{seed}:{arm_name}"
                command = build_training_command(feature_dir, run_dir, arm_code, seed, args, sources, check_only=False)
                print(f"[label-removal] {job_key}", flush=True)
                run_trainer(command, ROOT)
                manifest_path = run_dir / "run_manifest.json"
                require(manifest_path.is_file(), f"Trainer did not produce {manifest_path.relative_to(ROOT)}")
                run_manifest = read_json(manifest_path)
                require(run_manifest.get("status") == "completed", f"Run is incomplete: {job_key}")
                require(run_manifest.get("train_sha256") == sha256_file(feature_dir / "C2" / "train.jsonl"),
                        f"Training hash mismatch for {job_key}")
                require(run_manifest.get("validation_sha256") == sha256_file(sources["validation_path"]),
                        f"Validation hash mismatch for {job_key}")
                completed.add(job_key)
                plan = write_plan(
                    output_dir / "experiment_plan.json",
                    plan["configuration"],
                    {"status": "training", "completed_runs": sorted(completed)},
                )
    plan = write_plan(
        output_dir / "experiment_plan.json",
        plan["configuration"],
        {"status": "training_completed", "completed_runs": sorted(completed)},
    )
    return plan


def freeze_thresholds(args: argparse.Namespace, sources: dict, output_dir: Path, plan: dict) -> dict:
    expected_ids = sources["validation_ids"]
    evidence = {}
    threshold_results = {}
    validation_metrics = {}
    for fraction in REMOVAL_FRACTIONS:
        condition_key = removal_slug(fraction)
        threshold_results[condition_key] = {}
        validation_metrics[condition_key] = {}
        for arm_name, _ in LOSS_ARMS:
            score_by_seed = []
            target_by_seed = []
            run_evidence = {}
            per_seed_arrays = {}
            for seed in args.seeds:
                run_dir = job_directory(output_dir, fraction, seed, arm_name)
                manifest_path = run_dir / "run_manifest.json"
                predictions_path = run_dir / "validation_predictions.jsonl"
                require(manifest_path.is_file() and predictions_path.is_file(),
                        f"Missing completed validation outputs for {condition_key}, {arm_name}, seed {seed}")
                run_manifest = read_json(manifest_path)
                require(run_manifest.get("status") == "completed", f"Incomplete run: {manifest_path.relative_to(ROOT)}")
                require(run_manifest.get("validation_predictions_sha256") == sha256_file(predictions_path),
                        f"Validation prediction hash mismatch: {predictions_path.relative_to(ROOT)}")
                scores, targets = load_prediction_arrays(predictions_path, sources["labels"], expected_ids)
                score_by_seed.append(scores)
                target_by_seed.append(targets)
                per_seed_arrays[seed] = (scores, targets)
                run_evidence[str(seed)] = {
                    "run_manifest_sha256": sha256_file(manifest_path),
                    "validation_predictions_sha256": sha256_file(predictions_path),
                    "checkpoint_sha256": run_manifest.get("best_checkpoint", {}).get("sha256"),
                }
            scores_array = np.stack(score_by_seed)
            targets_array = np.stack(target_by_seed)
            threshold, validation_macro = select_macro_f1_threshold(scores_array, targets_array)
            threshold_results[condition_key][arm_name] = {
                "threshold": threshold,
                "validation_mean_macro_f1_at_threshold": validation_macro,
                "selection": "mean across seeds of fixed-inventory validation macro-F1",
                "run_evidence": run_evidence,
            }
            policy_name = "validation_macro_f1_tuned"
            validation_metrics[condition_key][arm_name] = {
                "fixed_0.5": {},
                policy_name: {},
            }
            for seed, (scores, targets) in per_seed_arrays.items():
                validation_metrics[condition_key][arm_name]["fixed_0.5"][str(seed)] = evaluate_scores(
                    scores, targets, sources["labels"], 0.5
                )
                validation_metrics[condition_key][arm_name][policy_name][str(seed)] = evaluate_scores(
                    scores, targets, sources["labels"], threshold
                )

    policy = {
        "status": "frozen_before_optional_evaluation",
        "dataset": "FinRED entity-disjoint validation",
        "validation_sha256": sha256_file(sources["validation_path"]),
        "seeds": list(args.seeds),
        "thresholds": threshold_results,
        "validation_metrics": validation_metrics,
        "selection_warning": "Threshold-tuned validation metrics reuse the validation labels for threshold selection; use an explicitly supplied held-out evaluation file for test metrics.",
        "bce_adjustment_definition": "Validation-tuned BCE uses the BCE checkpoint with a validation-selected global threshold; it is not a separately trained model.",
    }
    policy_path = output_dir / "threshold_policy.json"
    policy_path.write_bytes(json_bytes(policy))
    plan = write_plan(
        output_dir / "experiment_plan.json",
        plan["configuration"],
        {
            "status": "thresholds_frozen",
            "threshold_policy_sha256": sha256_file(policy_path),
            "evaluation_split_read": False,
        },
    )
    write_validation_summary(output_dir / "validation_summary.json", validation_metrics, args.seeds)
    return {"policy": policy, "policy_path": policy_path, "plan": plan}


def write_validation_summary(path: Path, metrics: dict, seeds: tuple[int, ...]) -> None:
    report = {
        "status": "validation_only_selection_contaminated",
        "seeds": list(seeds),
        "metrics": {},
        "interpretation": "These are validation results; macro-F1-tuned values are selected on the same validation labels and are not held-out estimates.",
    }
    for fraction_key, arms in metrics.items():
        report["metrics"][fraction_key] = {}
        for arm_name, policies in arms.items():
            report["metrics"][fraction_key][arm_name] = {}
            for policy_name, seed_values in policies.items():
                report["metrics"][fraction_key][arm_name][policy_name] = summarize_seed_metrics(seed_values)
        report.setdefault("paired_differences", {})[fraction_key] = {
            "NegDW-BCE_fixed_0.5_minus_BCE_fixed_0.5": paired_metric_difference(
                arms["NegDW-BCE"]["fixed_0.5"], arms["BCE"]["fixed_0.5"]
            ),
            "NegDW-BCE_tuned_minus_BCE_tuned": paired_metric_difference(
                arms["NegDW-BCE"]["validation_macro_f1_tuned"],
                arms["BCE"]["validation_macro_f1_tuned"],
            ),
            "BCE_tuned_minus_BCE_fixed_0.5": paired_metric_difference(
                arms["BCE"]["validation_macro_f1_tuned"], arms["BCE"]["fixed_0.5"]
            ),
        }
    path.write_bytes(json_bytes(report))


def evaluate_holdout(
    args: argparse.Namespace,
    sources: dict,
    output_dir: Path,
    prepared: dict,
    frozen: dict,
    evaluation_path: Path,
) -> dict:
    existing_plan = read_json(output_dir / "experiment_plan.json")
    if existing_plan.get("evaluation_file") is not None:
        require(
            existing_plan.get("evaluation_file") == evaluation_path.relative_to(ROOT).as_posix()
            and existing_plan.get("evaluation_sha256") == sha256_file(evaluation_path),
            "A different evaluation file was already recorded; use a new --output-dir for another evaluation.",
        )

    from safetensors.torch import load_file
    import torch

    sys.path.insert(0, str(ROOT / "src"))
    from orgu import neural, train_encoder

    split_manifest_path = project_path(
        ROOT,
        "data/splits/finred_entity_disjoint_v1/split_manifest.json",
        "Frozen split manifest",
        must_exist=True,
    )
    split_manifest = read_json(split_manifest_path)
    test_info = split_manifest.get("files", {}).get("test", {})
    expected_test_hash = test_info.get("sha256")
    expected_test_count = split_manifest.get("counts", {}).get("test")
    require(isinstance(expected_test_hash, str), "Frozen split manifest lacks the test SHA-256")
    require(expected_test_count == 701, "Frozen split manifest does not describe the 701-candidate test")
    evaluation_hash = sha256_file(evaluation_path)
    require(evaluation_hash == expected_test_hash,
            "Evaluation file SHA-256 differs from the frozen entity-disjoint test")
    evaluation_rows = read_jsonl(evaluation_path)
    require(len(evaluation_rows) == expected_test_count,
            f"Expected {expected_test_count} held-out candidates, found {len(evaluation_rows)}")
    evaluation_ids = validate_rows(evaluation_rows, sources["labels"], "test")
    require(not (set(evaluation_ids) & set(sources["train_ids"])), "Evaluation candidates overlap train")
    require(not (set(evaluation_ids) & set(sources["validation_ids"])), "Evaluation candidates overlap validation")
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        require(device.type != "cuda" or torch.cuda.is_available(), "CUDA was selected but is unavailable")

    seed_results = {}
    predictions_root = output_dir / "evaluation_predictions"
    predictions_root.mkdir(parents=True, exist_ok=True)
    for fraction in REMOVAL_FRACTIONS:
        condition_key = removal_slug(fraction)
        seed_results[condition_key] = {}
        for arm_name, _ in LOSS_ARMS:
            seed_results[condition_key][arm_name] = {
                "fixed_0.5": {},
                "validation_macro_f1_tuned": {},
            }
            threshold = float(frozen["policy"]["thresholds"][condition_key][arm_name]["threshold"])
            for seed in args.seeds:
                run_dir = job_directory(output_dir, fraction, seed, arm_name)
                run_manifest = read_json(run_dir / "run_manifest.json")
                feature_dir = prepared[(fraction, seed)]["feature_dir"]
                tokenizer_path = feature_dir / "tokenizer.json"
                from tokenizers import Tokenizer

                tokenizer = Tokenizer.from_file(str(tokenizer_path))
                marker_ids = {
                    name: tokenizer.token_to_id(token)
                    for name, token in train_encoder.MARKER_NAMES.items()
                }
                require(None not in marker_ids.values(), "Entity markers are missing from the local tokenizer")
                model = neural.EntityMarkerEncoder(
                    run_manifest["pretrained_model"],
                    tokenizer.get_vocab_size(with_added_tokens=True),
                    len(sources["labels"]),
                    revision=run_manifest.get("pretrained_model_revision"),
                ).to(device)
                checkpoint_value = run_manifest.get("best_checkpoint", {}).get("path")
                require(isinstance(checkpoint_value, str), "Run manifest lacks its selected checkpoint path")
                checkpoint_path = project_path(ROOT, checkpoint_value, "Selected checkpoint", must_exist=True)
                checkpoint_hash = sha256_file(checkpoint_path)
                require(checkpoint_hash == run_manifest["best_checkpoint"].get("sha256"), "Checkpoint SHA-256 mismatch")
                model.load_state_dict(load_file(str(checkpoint_path), device="cpu"), strict=True)
                _, predictions = train_encoder.evaluate(
                    model,
                    evaluation_rows,
                    sources["labels"],
                    tokenizer,
                    marker_ids,
                    device,
                    args.eval_batch_size,
                    split_sha256=evaluation_hash,
                )
                scores = np.asarray([row["scores"] for row in predictions], dtype=np.float64)
                targets = np.asarray([row["gold_targets"] for row in predictions], dtype=np.uint8)
                output_name = f"{condition_key}_{arm_slug(arm_name)}_seed{seed}.jsonl"
                prediction_path = predictions_root / output_name
                enriched_predictions = []
                for row in predictions:
                    enriched = dict(row)
                    enriched["validation_macro_f1_threshold"] = threshold
                    enriched["predicted_labels_fixed_0.5"] = row["predicted_labels"]
                    enriched["predicted_labels_validation_macro_f1_tuned"] = sorted(
                        label
                        for label, score in zip(sources["labels"], row["scores"])
                        if float(score) >= threshold
                    )
                    enriched_predictions.append(enriched)
                write_bytes_preserving(prediction_path, jsonl_bytes(enriched_predictions))
                seed_results[condition_key][arm_name]["fixed_0.5"][str(seed)] = evaluate_scores(
                    scores, targets, sources["labels"], 0.5
                )
                seed_results[condition_key][arm_name]["validation_macro_f1_tuned"][str(seed)] = evaluate_scores(
                    scores, targets, sources["labels"], threshold
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    aggregate = {}
    paired_differences = {}
    for fraction_key, arms in seed_results.items():
        aggregate[fraction_key] = {}
        for arm_name, policies in arms.items():
            aggregate[fraction_key][arm_name] = {
                policy_name: summarize_seed_metrics(seed_values)
                for policy_name, seed_values in policies.items()
            }
        paired_differences[fraction_key] = {
            "NegDW-BCE_fixed_0.5_minus_BCE_fixed_0.5": paired_metric_difference(
                arms["NegDW-BCE"]["fixed_0.5"], arms["BCE"]["fixed_0.5"]
            ),
            "NegDW-BCE_tuned_minus_BCE_tuned": paired_metric_difference(
                arms["NegDW-BCE"]["validation_macro_f1_tuned"],
                arms["BCE"]["validation_macro_f1_tuned"],
            ),
            "BCE_tuned_minus_BCE_fixed_0.5": paired_metric_difference(
                arms["BCE"]["validation_macro_f1_tuned"], arms["BCE"]["fixed_0.5"]
            ),
        }
    report = {
        "status": "posthoc_exploratory",
        "dataset": "FinRED entity-disjoint held-out evaluation",
        "evaluation_file": evaluation_path.relative_to(ROOT).as_posix(),
        "evaluation_sha256": evaluation_hash,
        "evaluation_candidates": len(evaluation_rows),
        "evaluation_previously_inspected": True,
        "evaluation_read_after_thresholds_frozen": True,
        "seeds": list(args.seeds),
        "removal_fractions": list(REMOVAL_FRACTIONS),
        "threshold_policy_sha256": sha256_file(frozen["policy_path"]),
        "metrics_mean_sample_sd": aggregate,
        "paired_differences": paired_differences,
        "per_seed_metrics": seed_results,
        "interpretation": "Synthetic removal tests robustness to controlled omissions, not recovery of naturally missing relations. Scores measure agreement with recorded annotations; this previously inspected split is exploratory.",
    }
    report_path = output_dir / "evaluation_summary.json"
    write_bytes_preserving(report_path, json_bytes(report))
    write_metrics_csv(output_dir / "evaluation_summary.csv", seed_results)
    plan = write_plan(
        output_dir / "experiment_plan.json",
        frozen["plan"]["configuration"],
        {
            "status": "evaluation_completed_posthoc",
            "evaluation_file": evaluation_path.relative_to(ROOT).as_posix(),
            "evaluation_sha256": evaluation_hash,
            "evaluation_split_read": True,
            "evaluation_summary_sha256": sha256_file(report_path),
        },
    )
    return report


def write_metrics_csv(path: Path, results: dict) -> None:
    rows = []
    for condition_key, arms in results.items():
        for arm_name, policies in arms.items():
            for policy_name, seed_metrics in policies.items():
                for metric in METRICS:
                    values = [float(entry[metric]) for entry in seed_metrics.values()]
                    rows.append(
                        {
                            "condition": condition_key,
                            "loss": arm_name,
                            "operating_point": policy_name,
                            "metric": metric,
                            "mean": statistics.mean(values),
                            "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                            "n_seeds": len(values),
                        }
                    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["condition"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired controlled removal of recorded positive labels using local C2 JSONL files."
    )
    parser.add_argument("--stage", choices=("train", "evaluate"), default="train")
    parser.add_argument("--feature-dir", default=DEFAULT_FEATURE_DIR.as_posix())
    parser.add_argument("--train-file", help="Optional repository-relative local C2 train JSONL override; hash must match the frozen manifest.")
    parser.add_argument("--validation-file", help="Optional repository-relative local C2 validation JSONL override; hash must match the frozen manifest.")
    parser.add_argument("--evaluation-file", help="Optional repository-relative local C2 held-out JSONL; read only after thresholds are frozen.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--pretrained-model", help="Defaults to the model pinned in context_manifest.json.")
    parser.add_argument("--pretrained-model-revision")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "cuda:0"), default="auto")
    parser.add_argument("--epochs", type=int, default=TRAINING_DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=TRAINING_DEFAULTS["batch_size"])
    parser.add_argument("--accumulation", type=int, default=TRAINING_DEFAULTS["accumulation"])
    parser.add_argument("--eval-batch-size", type=int, default=TRAINING_DEFAULTS["eval_batch_size"])
    parser.add_argument("--encoder-lr", type=float, default=TRAINING_DEFAULTS["encoder_lr"])
    parser.add_argument("--weight-decay", type=float, default=TRAINING_DEFAULTS["weight_decay"])
    parser.add_argument("--gradient-clip", type=float, default=TRAINING_DEFAULTS["gradient_clip"])
    parser.add_argument("--check-only", action="store_true", help="Prepare masks and run one trainer preflight; does not train or read evaluation data.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        require(len(args.seeds) >= 3, "Use at least three paired seeds")
        require(len(args.seeds) == len(set(args.seeds)), "Seed values must be unique")
        require(all(seed >= 0 for seed in args.seeds), "Seeds must be nonnegative")
        for name in ("epochs", "batch_size", "accumulation", "eval_batch_size"):
            require(getattr(args, name) > 0, f"--{name.replace('_', '-')} must be positive")
        require(args.encoder_lr > 0 and args.weight_decay >= 0 and args.gradient_clip > 0,
                "Learning rate and gradient clip must be positive; weight decay must be nonnegative")
        require(not args.check_only or args.stage == "train", "--check-only is available only during --stage train")
        require(args.stage != "evaluate" or args.evaluation_file,
                "--stage evaluate requires --evaluation-file")
        output_dir = project_path(ROOT, args.output_dir, "Output directory", must_exist=False)
        require(output_dir != ROOT, "Output directory cannot be the repository root")
        sources = resolve_sources(args)
        evaluation_path = validate_evaluation_path(args.evaluation_file)
        if evaluation_path is not None:
            require(evaluation_path not in (sources["train_path"], sources["validation_path"]),
                    "Evaluation file must differ from train and validation files")
        require(not output_dir.is_relative_to(sources["feature_dir"]),
                "Output directory must not be inside the source feature directory")
        require(not sources["train_path"].is_relative_to(output_dir)
                and not sources["validation_path"].is_relative_to(output_dir),
                "Output directory must not contain source train/validation files")
        prepared = prepare_all_jobs(args, sources, output_dir)
        configuration = build_configuration(args, sources, prepared)
        plan_path = output_dir / "experiment_plan.json"
        plan = write_plan(plan_path, configuration)
        if args.check_only:
            run_preflight(args, sources, prepared, output_dir)
            print(f"Prepared plan: {plan_path.relative_to(ROOT)}")
            return 0

        if args.stage == "train":
            require(plan.get("status") != "evaluation_completed_posthoc",
                    "This output already contains held-out results; use --stage evaluate or choose a new --output-dir.")
            plan = write_plan(plan_path, configuration, {"status": "training"})
            plan = train_all(args, sources, prepared, output_dir, plan)
        else:
            require(plan.get("status") in {"thresholds_frozen", "evaluation_completed_posthoc"},
                    "Train all jobs and freeze validation thresholds before held-out evaluation")
        frozen = freeze_thresholds(args, sources, output_dir, plan)
        if evaluation_path is None:
            print(f"Validation-only results: {(output_dir / 'validation_summary.json').relative_to(ROOT)}")
            print("No held-out evaluation was supplied; tuned validation scores are selection-contaminated.")
        else:
            report = evaluate_holdout(args, sources, output_dir, prepared, frozen, evaluation_path)
            print(f"Evaluation results: {(output_dir / 'evaluation_summary.json').relative_to(ROOT)}")
            print(f"Candidates evaluated: {report['evaluation_candidates']}")
        return 0
    except Exception as exc:
        print(f"[controlled-label-removal] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())